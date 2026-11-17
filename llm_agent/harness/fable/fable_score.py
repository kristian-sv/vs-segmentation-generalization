"""Score the Fable 5.2 high arm with the original metrics, beside the other arms.

  python3 harness/fable/fable_score.py

Uses the functions in harness/score.py unchanged (Dice, volume error, HD95, bootstrap).
Writes results/fable_high/scores.csv, summary.md and figures/dice_by_patient.png.
A case without a valid saved mask counts as an empty prediction. Nothing is exported
to the reviewer bundle.
"""
import csv
import importlib.util
import json
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("original_score", ROOT / "harness/score.py")
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)

TARGET = "Fable 5.2 high"
LLM = {TARGET: ROOT / "runs/fable_high", "Opus 5.5 high": ROOT / "runs/run1",
       "Astra high": ROOT / "runs/astra_high", "Astra xhigh": ROOT / "runs/astra_xhigh"}
OUT = ROOT / "results/fable_high"


def main():
    records = list(csv.DictReader((ROOT / "private/case_key.csv").open()))
    assert len(records) == 22
    missing = [r["patient_id"] for r in records if not (LLM[TARGET] / r["patient_id"] / "status.json").exists()]
    assert not missing, f"arm is incomplete: {missing}"
    rows = []
    for r in sorted(records, key=lambda r: r["patient_id"]):
        pid = r["patient_id"]
        ref = nib.load(original.DATA / pid / "label.nii.gz")
        gt = np.asarray(ref.dataobj) > 0
        sp = ref.header.get_zooms()[:3]
        sources = {}
        for name, folder in LLM.items():
            status = json.loads((folder / pid / "status.json").read_text())["status"]
            mask = original.load_mask(folder / pid / "label.nii.gz", ref) if status == "ok" else np.zeros_like(gt)
            sources[name] = (mask, status)
        for filename, name in original.COMPARATORS.items():
            sources[name] = (original.load_mask(original.BUNDLE / pid / f"{filename}.nii.gz", ref), "ok")
        for name, (mask, status) in sources.items():
            rows.append({"patient": pid, "run_order_case": r["case"], "stratum": original.stratum(pid),
                         "source": name, "status": status,
                         **{k: round(float(v), 4) for k, v in original.metrics(mask, gt, sp).items()}})
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "scores.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    patients = sorted({r["patient"] for r in rows})
    look = {(r["patient"], r["source"]): r for r in rows}
    arms = {**{name: [name] for name in LLM}, "plain arm": ["plain s42", "plain s43"],
            "augmented arm": ["augmented s42", "augmented s43"], "nnU-Net initial": ["nnU-Net initial"]}
    dice = {a: np.array([np.mean([look[(p, s)]["dice"] for s in ss]) for p in patients]) for a, ss in arms.items()}

    lines = ["# Fable 5.2 high: 22 MC-RC patients", "",
             "Model: `claude-fable-5-2`, effort high, Claude Code CLI on the Claude subscription login. One "
             "independent session per patient. Same prompt, 12 tools, Docker image, staged images, case order and "
             "40-call / 20-image / 60-minute limits as run1 (Opus 5.5); only the model name differs. Invalid or "
             "unsaved masks count as empty predictions. nnU-Net arms average their two seeds within each patient. "
             "Paired 95% intervals: patient bootstrap, 10,000 resamples, seed 42. This arm was added after the "
             "Opus and Astra results were known and is not in PROTOCOL.md. Patient 028 was the pilot image.", ""]
    groups = {"All patients": None, "Primary slab cases (13)": original.SLAB13, **original.STRATA}
    for group, ids in groups.items():
        sel = np.array([ids is None or p.split("-")[-1] in ids for p in patients])
        n = int(sel.sum())
        lines += [f"## {group} (n = {n})", "", "| Source | Mean Dice | Median Dice | Empty outputs |",
                  "|---|---:|---:|---:|"]
        for a, ss in arms.items():
            d = dice[a][sel]
            empty = sum(look[(p, s)]["empty"] for p, use in zip(patients, sel) if use for s in ss)
            lines.append(f"| {a} | {d.mean():.3f} | {np.median(d):.3f} | {int(empty)}/{n * len(ss)} |")
        lines += ["", "| Paired difference (Fable minus) | Mean | 95% CI | Fable higher in |", "|---|---:|---|---:|"]
        for a in arms:
            if a == TARGET:
                continue
            diff = (dice[TARGET] - dice[a])[sel]
            lo, hi = original.boot_ci(diff)
            lines.append(f"| {a} | {diff.mean():+.3f} | [{lo:+.3f}, {hi:+.3f}] | {int((diff > 0).sum())}/{n} |")
        lines.append("")

    lines += ["## Volume error and HD95, all patients (LLM arms)", "",
              "| Source | Median abs. volume error (%) | Median signed volume error (%) | Under-segmented | "
              "Median HD95 (mm) | Dice = 0 | Dice < 0.5 |", "|---|---:|---:|---:|---:|---:|---|"]
    for a in LLM:
        v = np.array([look[(p, a)]["vol_err_pct"] for p in patients])
        h = np.array([look[(p, a)]["hd95_mm"] for p in patients])
        d = dice[a]
        low = ", ".join(p.split("-")[-1] for p, x in zip(patients, d) if x < 0.5) or "none"
        lines.append(f"| {a} | {np.median(np.abs(v)):.1f} | {np.median(v):+.1f} | {int((v < 0).sum())}/22 | "
                     f"{np.nanmedian(h):.2f} | {int((d == 0).sum())} | {low} |")

    lines += ["", "## Per-patient Dice", "", "| Patient | Stratum | " + " | ".join(arms) + " |",
              "|---|---|" + "---:|" * len(arms)]
    for i, p in enumerate(patients):
        lines.append(f"| {p} | {original.stratum(p)} | " + " | ".join(f"{dice[a][i]:.3f}" for a in arms) + " |")
    counts = {}
    for p in patients:
        s = look[(p, TARGET)]["status"]
        counts[s] = counts.get(s, 0) + 1
    lines += ["", f"Run outcomes (Fable 5.2 high): {counts}"]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")

    (OUT / "figures").mkdir(exist_ok=True)
    order = [p for s in original.STRATA for p in patients if original.stratum(p) == s]
    x = np.arange(len(order))
    fig, ax = original.plt.subplots(figsize=(12, 4))
    for a, mk, c in ((TARGET, "o", "#d62728"), ("Opus 5.5 high", "x", "#ff7f0e"), ("Astra xhigh", "D", "#2ca02c"),
                     ("augmented arm", "s", "#1f77b4"), ("plain arm", "^", "#7f7f7f")):
        ax.scatter(x, [dice[a][patients.index(p)] for p in order], marker=mk, color=c, label=a, s=28, zorder=3)
    for b in np.cumsum([len(v) for v in original.STRATA.values()])[:-1]:
        ax.axvline(b - 0.5, color="k", lw=0.6, ls=":")
    ax.set_xticks(x, [p.split("-")[-1] for p in order], fontsize=8)
    ax.set_ylabel("Dice")
    ax.set_ylim(-0.03, 1.0)
    ax.set_title("Per-patient Dice by stratum: axial whole-head | axial slab | non-axial")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "figures/dice_by_patient.png", dpi=150)
    print((OUT / "summary.md").read_text())


if __name__ == "__main__":
    main()
