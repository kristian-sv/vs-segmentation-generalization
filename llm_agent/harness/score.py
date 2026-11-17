"""Score the agent runs against the reference labels (host only, after all runs).

  python harness/score.py             score and write results/
  python harness/score.py --reviewer  also copy the predictions into the MMIV reviewer bundle

Reads runs/run*/<patient_id>/label.nii.gz,
scores against the evaluation labels and the existing nnU-Net predictions, and writes
results/scores.csv, results/summary.md and results/figures/dice_by_patient.png.
A run without a valid saved mask counts as an empty prediction (and is exported as an empty mask).
"""
import sys
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
DATA = Path.home() / "MMIV/data/VS_MCRC_T1_evaluation_mm"
BUNDLE = Path.home() / "MMIV/data/vs_viewer_bundle/data/mcrc"
OUT = ROOT / "results"
# acquisition strata as defined in the thesis (Table 4.6)
STRATA = {
    "axial whole-head": ["021", "028", "044", "063", "082", "085", "099"],
    "axial slab": ["009", "016", "024", "026", "030", "045", "057", "076", "081", "097", "098"],
    "non-axial": ["012", "043", "060", "100"],
}
SLAB13 = STRATA["axial slab"] + ["012", "100"]
COMPARATORS = {  # bundle file -> label (the follow-up models used re-stacked inputs)
    "pred_nnunet_s42": "nnU-Net initial",
    "pred_plain_s42": "plain s42", "pred_plain_s43": "plain s43",
    "pred_slabthick_s42": "augmented s42", "pred_slabthick_s43": "augmented s43",
}


def stratum(pid):
    return next(s for s, ids in STRATA.items() if pid.split("-")[-1] in ids)


def surface(m):
    return m & ~ndi.binary_erosion(m, ndi.generate_binary_structure(3, 1))


def hd95(p, g, sp):
    if not p.any():
        return np.nan
    d1 = ndi.distance_transform_edt(~surface(g), sampling=sp)[surface(p)]
    d2 = ndi.distance_transform_edt(~surface(p), sampling=sp)[surface(g)]
    return float(max(np.percentile(d1, 95), np.percentile(d2, 95)))


def metrics(p, g, sp):
    vox_ml = float(np.prod(sp)) / 1000
    inter = (p & g).sum()
    return {"dice": 2 * inter / (p.sum() + g.sum()) if (p.sum() + g.sum()) else 1.0,
            "empty": int(not p.any()),
            "pred_ml": p.sum() * vox_ml, "ref_ml": g.sum() * vox_ml,
            "vol_err_ml": (p.sum() - g.sum()) * vox_ml,
            "vol_err_pct": 100 * (p.sum() - g.sum()) / g.sum(),
            "hd95_mm": hd95(p, g, sp)}


def load_mask(path, ref):
    im = nib.load(path)
    if im.shape != ref.shape or not np.allclose(im.affine, ref.affine, atol=1e-3):
        raise ValueError(f"grid mismatch: {path}")
    return np.asarray(im.dataobj) > 0


def boot_ci(diff, seed=42, n=10000):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), (n, len(diff)))
    return np.percentile(diff[idx].mean(1), [2.5, 97.5])


def main():
    key = {r["patient_id"]: r["case"] for r in csv.DictReader((ROOT / "private/case_key.csv").open())}
    runs = sorted(p.name for p in (ROOT / "runs").glob("run[0-9]*") if p.is_dir())
    rows = []
    for pid, case in sorted(key.items()):
        gt_img = nib.load(DATA / pid / "label.nii.gz")
        g = np.asarray(gt_img.dataobj) > 0
        sp = gt_img.header.get_zooms()[:3]
        sources = {}
        for run in runs:
            st = ROOT / "runs" / run / pid / "status.json"
            status = json.loads(st.read_text())["status"] if st.exists() else "missing"
            p = load_mask(ROOT / "runs" / run / pid / "label.nii.gz", gt_img) if status == "ok" else np.zeros_like(g)
            sources[f"LLM {run}"] = (p, status)
        for f, label in COMPARATORS.items():
            path = BUNDLE / pid / f"{f}.nii.gz"
            if path.exists():
                sources[label] = (load_mask(path, gt_img), "ok")
        if "--reviewer" in sys.argv:  # reviewer shows pred_*.nii.gz files in the bundle's patient folder
            for run in runs:
                p, _ = sources[f"LLM {run}"]
                out = nib.Nifti1Image(p.astype(np.uint8), gt_img.affine, gt_img.header)
                out.set_data_dtype(np.uint8)
                nib.save(out, BUNDLE / pid / f"pred_llm_opus55_{run}.nii.gz")
        for source, (p, status) in sources.items():
            rows.append({"patient": pid, "run_order_case": case, "stratum": stratum(pid), "source": source,
                         "status": status, **{k: round(float(v), 4) for k, v in metrics(p, g, sp).items()}})
        print(pid, " ".join(f"{s}={r['dice']:.3f}" for s, r in ((r["source"], r) for r in rows if r["patient"] == pid)))

    OUT.mkdir(exist_ok=True)
    with (OUT / "scores.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    # patient-level tables: one Dice per patient and source; arms average their two seeds/runs
    pts = sorted({r["patient"] for r in rows})
    dice = {(r["patient"], r["source"]): r["dice"] for r in rows}
    empty = {(r["patient"], r["source"]): r["empty"] for r in rows}
    llm_arm = "LLM agent" if len(runs) == 1 else f"LLM ({len(runs)}-run mean)"
    arms = {llm_arm: [f"LLM {r}" for r in runs],
            "plain arm": ["plain s42", "plain s43"], "augmented arm": ["augmented s42", "augmented s43"],
            "nnU-Net initial": ["nnU-Net initial"]}
    singles = [f"LLM {r}" for r in runs]
    arm_d = {a: np.array([np.mean([dice[(p, s)] for s in ss]) for p in pts]) for a, ss in arms.items()}
    arm_e = {a: np.array([np.mean([empty[(p, s)] for s in ss]) * len(ss) for p in pts]) for a, ss in arms.items()}

    def block(title, sel):
        m = np.array([p.split("-")[-1] in sel for p in pts]) if sel else np.ones(len(pts), bool)
        lines = [f"### {title} (n = {m.sum()})", "", "| Source | Mean Dice | Median Dice | Empty outputs |",
                 "|---|---|---|---|"]
        for s in (singles if len(singles) > 1 else []):
            d = np.array([dice[(p, s)] for p in pts])[m]
            e = int(sum(empty[(p, s)] for p, k in zip(pts, m) if k))
            lines.append(f"| {s} | {d.mean():.3f} | {np.median(d):.3f} | {e}/{m.sum()} |")
        for a in arms:
            n_models = len(arms[a])
            lines.append(f"| {a} | {arm_d[a][m].mean():.3f} | {np.median(arm_d[a][m]):.3f} | "
                         f"{int(arm_e[a][m].sum())}/{m.sum() * n_models} |")
        llm = arm_d[llm_arm][m]
        lines += ["", "| Paired difference (LLM minus) | Mean | 95% CI |", "|---|---|---|"]
        for a in ("augmented arm", "plain arm", "nnU-Net initial"):
            diff = llm - arm_d[a][m]
            lo, hi = boot_ci(diff)
            lines.append(f"| {a} | {diff.mean():+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
        return lines + [""]

    md = ["# LLM agent vs nnU-Net on the 22 MC-RC patients", "",
          f"Model: claude-opus-5-5, effort high. Runs: {', '.join(runs)}. Arms average their models within each "
          "patient (no mask ensembling). Empty outputs count individual model predictions. Paired 95% CIs: "
          "patient bootstrap, 10,000 resamples, seed 42. A run that never saved a valid mask counts as empty.", ""]
    md += block("All patients", None)
    md += block("Primary slab cases (13 slabs, as in RQ3)", SLAB13)
    for s, ids in STRATA.items():
        md += block(s.capitalize(), ids)
    md += ["## Per patient (Dice)", "", "| Patient | Stratum | " + " | ".join(arms) + " |",
           "|---|---|" + "---|" * len(arms)]
    for i, p in enumerate(pts):
        md.append(f"| {p} | {stratum(p)} | " + " | ".join(f"{arm_d[a][i]:.3f}" for a in arms) + " |")
    status_counts = {}
    for r in rows:
        if r["source"].startswith("LLM"):
            status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1
    md += ["", f"Run outcomes: {status_counts}"]
    (OUT / "summary.md").write_text("\n".join(md) + "\n")

    # figure: per-patient Dice, grouped by stratum
    (OUT / "figures").mkdir(exist_ok=True)
    order = [p for s in STRATA for p in pts if stratum(p) == s]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(12, 4))
    for (a, mk, c) in ((llm_arm, "o", "#d62728"), ("augmented arm", "s", "#1f77b4"),
                       ("plain arm", "^", "#7f7f7f")):
        ax.scatter(x, [arm_d[a][pts.index(p)] for p in order], marker=mk, color=c, label=a, s=28, zorder=3)
    for b in np.cumsum([len(v) for v in STRATA.values()])[:-1]:
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
