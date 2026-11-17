"""Fable 5.2 high on the 22 MC-RC patients: the original harness with only the model changed.

  python3 harness/fable/fable_run.py prepare   check parity with run1 and write the manifest
  python3 harness/fable/fable_run.py launch    run every case not done yet (two at a time)

Reuses harness/run.py unchanged (prompt, 12 tools, Docker sandbox, staged images, shuffled
case order, 40-call / 20-image / 60-minute limits, `--jobs 2`, Claude subscription login).
Outputs go to runs/fable_high/<patient_id>/. A case that ends in a CLI error, or that was
answered by a model other than the one requested, stops the arm; nothing is retried
automatically.
"""
import argparse
import csv
import fcntl
import hashlib
import importlib.util
import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL = "claude-fable-5-2"
EFFORT = "high"
JOBS = 2
ARM = ROOT / "runs/fable_high"
# values recorded for run1 (Opus 5.5) and the Astra arms; the run refuses to start if any differ
EXPECT = {
    "prompt_sha256": "aa9e523489703a4f84152735b939622ecccb130e3faeb77cbf251a942688c975",
    "tools_sha256": "658e5160a4e7a8a749c94c393264b35c2f14a83d1358cbd6331f35232f189de5",
    "docker_image_id": "sha256:9560b625afffe5228b50f09a1a67795848029f6bc12ee0543087bd0f698a5b8f",
    "claude_cli": "2.1.311 (Claude Code)",
}

spec = importlib.util.spec_from_file_location("original_harness", ROOT / "harness/run.py")
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
TAG = original.IMAGE  # llm-seg-tools:v1, as in run1
original.MODEL, original.EFFORT = MODEL, EFFORT


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1) + "\n")


def prepare():
    ARM.mkdir(parents=True, exist_ok=True)
    path = ARM / "manifest.json"
    if path.exists():
        return json.loads(path.read_text())
    records = list(csv.DictReader(original.KEY.open()))
    assert len(records) == 22
    found = {
        "prompt_sha256": sha(ROOT / "harness/prompt.md"),
        "tools_sha256": sha(ROOT / "harness/tools.py"),
        "docker_image_id": subprocess.check_output(
            ["docker", "image", "inspect", TAG, "--format", "{{.Id}}"], text=True).strip(),
        "claude_cli": subprocess.check_output(["claude", "--version"], text=True).strip(),
    }
    assert found == EXPECT, f"setup differs from run1: {found}"
    for r in records:  # same source scans as run1, and a staged copy for each
        pid = r["patient_id"]
        assert sha(original.DATA / pid / "T1.nii.gz") == r["t1_sha256"], pid
        r["staged_sha256"] = sha(original.STAGING / pid / "image.nii.gz")
    manifest = {"created_at": now(), "cohort": "VS_MCRC_T1", "model": MODEL, "effort": EFFORT, "total": 22,
                "jobs": JOBS, "shuffle_seed": original.SHUFFLE_SEED, "records": records,
                "parity": {**found, "docker_image_tag": TAG, "original_harness_sha256": sha(ROOT / "harness/run.py"),
                           "authentication": "Claude subscription; no API keys inherited",
                           "tool_calls": 40, "images": 20, "timeout_s": original.TIMEOUT_S,
                           "same_case_order_as_run1": True}}
    dump(path, manifest)
    return manifest


def models_in(transcript):
    """Model names the CLI reports for the session: the init line and every assistant message."""
    seen = set()
    for line in transcript.read_text().splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "system" and ev.get("subtype") == "init":
            seen.add(ev.get("model"))
        if ev.get("type") == "assistant":
            seen.add(ev.get("message", {}).get("model"))
    return sorted(m for m in seen if m and m != "<synthetic>")


def launch():
    manifest = prepare()
    lock = (ARM / "run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    order = [r["patient_id"] for r in manifest["records"]]
    by_pid = {r["patient_id"]: r for r in manifest["records"]}
    todo = []
    for pid in order:
        out = ARM / pid
        if (out / "status.json").exists():
            continue
        if out.exists() and any(out.iterdir()):
            sys.exit(f"Partial attempt for {pid}; inspect it before resuming (no automatic rerun)")
        todo.append(pid)
    print(f"[{now()}] fable_high: {len(todo)} of {len(order)} cases to do", flush=True)
    stop = threading.Event()
    errors = []

    def one(pid):
        if stop.is_set():
            return
        out = ARM / pid
        print(f"[{now()}] START {pid}", flush=True)
        record = original.run_case(pid, original.STAGING / pid, out)
        seen = models_in(out / "transcript.jsonl")
        record.update(run_order_case=by_pid[pid]["case"], models_seen=seen,
                      input_sha256=by_pid[pid]["t1_sha256"], staged_sha256=by_pid[pid]["staged_sha256"],
                      prompt_sha256=manifest["parity"]["prompt_sha256"],
                      tools_sha256=manifest["parity"]["tools_sha256"],
                      image_id=manifest["parity"]["docker_image_id"],
                      authentication="Claude subscription", finished_at=now())
        bad = record["returncode"] not in (0, "timeout") or record.get("is_error") or seen != [MODEL]
        if bad:  # keep the transcript, do not count the case as done, start no further cases
            (out / "status.json").unlink()
            dump(out / "infrastructure_error.json", record)
            errors.append(pid)
            stop.set()
            print(f"[{now()}] ERROR {pid}: returncode {record['returncode']}, is_error {record.get('is_error')}, "
                  f"models {seen}", flush=True)
            return
        dump(out / "status.json", record)
        print(f"[{now()}] DONE {pid}: {record['status']}, {record['wall_s']} s", flush=True)

    with ThreadPoolExecutor(JOBS) as ex:
        list(ex.map(one, todo))
    done = sum((ARM / pid / "status.json").exists() for pid in order)
    if errors:
        sys.exit(f"BLOCKED after {done}/22: error in {errors}; no automatic retry")
    print(f"[{now()}] COMPLETE: {done}/22 cases have a status", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["prepare", "launch"])
    a = ap.parse_args()
    {"prepare": prepare, "launch": launch}[a.command]()
