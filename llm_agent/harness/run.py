"""Stage the MC-RC images, run one fresh agent per case, and validate its output.

  python harness/run.py stage                  copy T1 images only, headers cleared (run once)
  python harness/run.py run --run 1 [--jobs 2] run the agent on every staged case not done yet
  python harness/run.py smoke                  one run on pilot/image.nii.gz (debugging)

Each case: a new container (no network, only that case's image mounted) serves the tools,
and a new `claude -p` session (no built-in tools, no settings, no memory) is the agent.
"""
import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path.home() / "MMIV/data/VS_MCRC_T1_evaluation_mm"
STAGING = ROOT / "staging"
KEY = ROOT / "private/case_key.csv"
MODEL = "claude-opus-5-5"
EFFORT = "high"
IMAGE = "llm-seg-tools:v1"
TIMEOUT_S = 60 * 60
SHUFFLE_SEED = 2026
EXPECTED_OUTPUTS = {"label.nii.gz", "tool_log.jsonl", "views"}


def read_prompt():
    text = (ROOT / "harness/prompt.md").read_text()
    system, task = text.split("# Task")
    return system.replace("# System prompt", "").strip(), task.strip()


def sanitized_copy(src, dst):
    """Copy a NIfTI with identical voxels and geometry but empty free-text header fields."""
    im = nib.load(src)
    data = np.asarray(im.dataobj)
    hdr = im.header.copy()
    for field in ("descrip", "aux_file", "db_name", "intent_name"):
        hdr[field] = b""
    slope, inter = im.dataobj.slope, im.dataobj.inter
    if slope != 1 or inter != 0:
        data = data.astype(np.float32)
        hdr.set_data_dtype(np.float32)
    hdr.set_slope_inter(1.0, 0.0)
    nib.save(nib.Nifti1Image(data.astype(hdr.get_data_dtype()), im.affine, hdr), dst)
    check = nib.load(dst)
    assert np.array_equal(np.asarray(check.dataobj), np.asarray(im.dataobj)), src
    assert np.allclose(check.affine, im.affine), src


def stage():
    if KEY.exists():
        sys.exit(f"{KEY} exists; staging is done once. Delete staging/ and private/ to redo it.")
    patients = sorted(p.name for p in DATA.iterdir() if (p / "T1.nii.gz").exists())
    random.Random(SHUFFLE_SEED).shuffle(patients)
    KEY.parent.mkdir(exist_ok=True)
    with KEY.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "patient_id", "t1_sha256"])
        for n, pid in enumerate(patients, start=1):
            case = f"case_{n:02d}"  # position in the (shuffled) run order
            (STAGING / pid).mkdir(parents=True)
            src = DATA / pid / "T1.nii.gz"
            sanitized_copy(src, STAGING / pid / "image.nii.gz")
            w.writerow([case, pid, hashlib.sha256(src.read_bytes()).hexdigest()])
    print(f"staged {len(patients)} cases (images only) into {STAGING}")


def run_order():
    """Patient IDs in the shuffled order recorded in the key (folder names are never shown to the agent)."""
    return [r["patient_id"] for r in csv.DictReader(KEY.open())]


def validate(out, image_path):
    """Return (status, detail). Status: ok | no_save | invalid."""
    extra = ({p.name for p in out.iterdir()} - EXPECTED_OUTPUTS
             - {"transcript.jsonl", "stderr.log", "status.json", ".DS_Store"})
    label = out / "label.nii.gz"
    if not label.exists():
        return "no_save", {"unexpected_files": sorted(extra)}
    if label.is_symlink():
        return "invalid", {"reason": "label is a symlink"}
    try:
        lab, ref = nib.load(label), nib.load(image_path)
        vals = np.unique(np.asarray(lab.dataobj))
        if lab.shape != ref.shape or not np.allclose(lab.affine, ref.affine, atol=1e-4):
            return "invalid", {"reason": "grid mismatch"}
        if not set(vals.tolist()) <= {0, 1}:
            return "invalid", {"reason": f"values {vals[:5].tolist()}"}
    except Exception as e:
        return "invalid", {"reason": str(e)}
    return "ok", {"foreground_voxels": int((np.asarray(lab.dataobj) > 0).sum()), "unexpected_files": sorted(extra)}


def run_case(case, image_dir, out):
    out.mkdir(parents=True, exist_ok=True)
    os.chmod(out, 0o777)  # the container user (uid 1000) writes here
    system, task = read_prompt()
    container = f"seg-{out.parent.name}-{case}"
    server = {"command": "docker", "args": [
        "run", "-i", "--rm", "--name", container, "--network", "none", "--read-only",
        "--tmpfs", "/tmp:size=512m", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--pids-limit", "256", "--memory", "4g", "--cpus", "2",
        "-v", f"{image_dir}:/data:ro", "-v", f"{out}:/out", IMAGE]}
    # minimal environment: subscription login from the keychain, no API keys, no inherited
    # Claude session variables, no auto-memory
    env = {k: os.environ[k] for k in ("HOME", "USER", "LOGNAME", "TMPDIR", "LANG") if k in os.environ}
    env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
    env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] = "1"
    start = time.time()
    with tempfile.TemporaryDirectory(prefix="agent-") as cwd:  # empty working dir, deleted afterwards
        cfg = Path(cwd) / "mcp.json"
        cfg.write_text(json.dumps({"mcpServers": {"seg": server}}))
        cmd = ["claude", "-p", task, "--system-prompt", system, "--model", MODEL, "--effort", EFFORT,
               "--tools", "", "--strict-mcp-config", "--mcp-config", str(cfg), "--allowedTools", "mcp__seg",
               "--setting-sources", "", "--no-session-persistence", "--output-format", "stream-json", "--verbose"]
        with (out / "transcript.jsonl").open("w") as so, (out / "stderr.log").open("w") as se:
            try:
                rc = subprocess.run(cmd, cwd=cwd, env=env, stdout=so, stderr=se, timeout=TIMEOUT_S).returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
            finally:
                subprocess.run(["docker", "rm", "-f", container], capture_output=True)
        cfg.unlink()
    status, detail = validate(out, image_dir / "image.nii.gz")
    result = {}
    for line in (out / "transcript.jsonl").read_text().splitlines():
        if '"type":"result"' in line.replace(" ", ""):
            result = json.loads(line)
    cli = subprocess.run(["claude", "--version"], capture_output=True, text=True, env=env).stdout.strip()
    record = {"patient_id": case, "status": status, **detail, "returncode": rc, "model": MODEL, "effort": EFFORT,
              "image": IMAGE, "claude_cli": cli, "wall_s": round(time.time() - start), "num_turns": result.get("num_turns"),
              "is_error": result.get("is_error"), "usage": result.get("usage"),
              "cost_usd_equivalent": result.get("total_cost_usd")}
    (out / "status.json").write_text(json.dumps(record, indent=1))
    print(f"{case}: {status} ({record['wall_s']} s, {record['num_turns']} turns)", flush=True)
    return record


def run(run_id, jobs):
    cases = run_order()
    todo = [c for c in cases if not (ROOT / f"runs/run{run_id}/{c}/status.json").exists()]
    print(f"run{run_id}: {len(todo)} of {len(cases)} cases to do", flush=True)
    with ThreadPoolExecutor(jobs) as ex:
        list(ex.map(lambda c: run_case(c, STAGING / c, ROOT / f"runs/run{run_id}/{c}"), todo))


def smoke():
    out = ROOT / "runs/smoke"
    if out.exists():
        shutil.rmtree(out)
    with tempfile.TemporaryDirectory(prefix="smoke-") as d:
        sanitized_copy(ROOT / "pilot/image.nii.gz", Path(d) / "image.nii.gz")
        run_case("pilot", Path(d), out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stage", "run", "smoke"])
    ap.add_argument("--run", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    {"stage": stage, "smoke": smoke, "run": lambda: run(a.run, a.jobs)}[a.cmd]()
