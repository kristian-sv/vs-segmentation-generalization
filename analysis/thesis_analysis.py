"""Reproducible descriptive analysis of frozen scores; no inference or new scoring."""
import csv
import json
from pathlib import Path


def analyse(evidence, destination):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader((evidence / "analysis/reports/t1_evaluation_20260912/patient_scores.csv").open()))
    runs = sorted({r["run"] for r in rows})
    assert len(runs) == 7
    colors = ["#2166ac" if r.startswith("baseline") else "#d95f02" if r.startswith("dropout") else "#238b45" for r in runs]
    names = [r.replace("baseline_", "Baseline ").replace("dropout01_", "Dropout 0.1 ").replace("nnunet_", "nnU-Net ") for r in runs]
    output = {"bootstrap": {"method": "patient percentile bootstrap", "seed": 42, "replicates": 10000,
                           "shared_indices_across_runs": True, "confidence": 0.95,
                           "scope": "Conditional on fitted models and evaluated cohort; not training or site uncertainty."}, "cohorts": {}}
    matrices = {}
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for cohort, n, offset, marker in [("validation", 24, -.12, "o"), ("mcrc", 22, .12, "s")]:
        subset = [r for r in rows if r["cohort"] == cohort]
        cases = sorted({r["case_id"] for r in subset})
        assert len(cases) == n and len(subset) == n * len(runs)
        by = {(r["case_id"], r["run"]): r for r in subset}
        assert len(by) == n * len(runs)
        d = np.array([[float(by[c, r]["dice"]) for r in runs] for c in cases])
        ref = np.array([[float(by[c, r]["reference_ml"]) for r in runs] for c in cases])
        pred = np.array([[float(by[c, r]["prediction_ml"]) for r in runs] for c in cases])
        assert np.isfinite(d).all() and ((d >= 0) & (d <= 1)).all() and (ref > 0).all()
        assert np.allclose(ref, ref[:, :1])
        indices = np.random.default_rng(42).integers(0, n, size=(10000, n))
        ci = np.quantile(d[indices].mean(axis=1), [.025, .975], axis=0)
        stats = {}
        for j, run in enumerate(runs):
            stats[run] = {"n": n, "mean_dice": float(d[:, j].mean()), "median_dice": float(np.median(d[:, j])),
                          "patient_sd": float(d[:, j].std(ddof=1)), "mean_dice_ci95": ci[:, j].tolist(),
                          "dice_below_0_1": int((d[:, j] < .1).sum()), "empty_predictions": int((pred[:, j] == 0).sum()),
                          "median_absolute_volume_error_ml": float(np.median(abs(pred[:, j] - ref[:, j]))),
                          "median_absolute_relative_volume_error_percent": float(np.median(100 * abs(pred[:, j] - ref[:, j]) / ref[:, j]))}
            ax.errorbar(j+offset, d[:, j].mean(), yerr=[[d[:, j].mean()-ci[0,j]], [ci[1,j]-d[:, j].mean()]],
                        color=colors[j], fmt=marker, capsize=3, label=cohort if j == 0 else None)
        output["cohorts"][cohort] = {"runs": stats, "reference_volume_ml": {"median": float(np.median(ref[:,0])), "min": float(ref.min()), "max": float(ref.max())}}
        matrices[cohort] = cases, d, ref[:,0], pred
    ax.set(xticks=range(len(runs)), xticklabels=names, ylabel="Mean patient Dice", ylim=(-.02,1.02))
    ax.tick_params(axis="x", rotation=25); ax.legend(); ax.grid(axis="y", alpha=.2)
    fig.tight_layout(); fig.savefig(destination / "v003_cohort_means.pdf"); plt.close(fig)
    cases,d,ref,pred = matrices["mcrc"]
    order = np.argsort(ref)
    fig, ax = plt.subplots(figsize=(8.3,9))
    im=ax.imshow(d[order], vmin=0, vmax=1, cmap="viridis", aspect="auto")
    for i,k in enumerate(order):
        for j in range(len(runs)):
            if pred[k,j] == 0: ax.text(j,i,"×",ha="center",va="center",color="white",fontsize=12)
    ax.set(xticks=range(len(runs)),xticklabels=names,yticks=range(len(cases)),
           yticklabels=[f"{cases[k]} ({ref[k]:.2f} ml)" for k in order])
    ax.tick_params(axis="x", rotation=35); fig.colorbar(im,ax=ax,label="Dice")
    fig.tight_layout(); fig.savefig(destination / "v003_patient_heatmap.pdf"); plt.close(fig)
    manual = [i for i,r in enumerate(runs) if not r.startswith("nnunet")]
    nn = runs.index("nnunet_s42")
    low = d < .1
    output["patient_patterns"] = {"all_runs_below_0_1": [c for c,k in zip(cases,low.all(axis=1)) if k],
        "any_run_below_0_1": int(low.any(axis=1).sum()), "low_dice_patient_run_pairs": int(low.sum()),
        "empty_low_dice_pairs": int((low & (pred==0)).sum()),
        "small_nonempty_low_dice_pairs": int((low & (pred>0) & (pred < .25*ref[:,None])).sum()),
        "nnunet_empty_cases": [c for c,k in zip(cases,pred[:,nn]==0) if k]}
    output["patients"] = [{"case_id":c,"reference_ml":float(ref[i]),"dice":dict(zip(runs,map(float,d[i]))),
                           "prediction_ml":dict(zip(runs,map(float,pred[i])))} for i,c in enumerate(cases)]
    scanner_rows=list(csv.DictReader((evidence / "analysis/reports/per_case_metrics.csv").open()))
    scanner={r["case_id"]:r["scanner_group"] for r in scanner_rows}
    output["scanner_groups"]={g:{"n":sum(scanner[c]==g for c in cases),
       "manual_six_run_mean":float(d[[scanner[c]==g for c in cases]][:,manual].mean()),
       "nnunet_mean":float(d[[scanner[c]==g for c in cases],nn].mean())} for g in sorted({scanner[c] for c in cases})}
    output["volume_subgroups"]={}
    for name, mask in [("below_median",ref<np.median(ref)),("at_or_above_median",ref>=np.median(ref))]:
        output["volume_subgroups"][name]={"n":int(mask.sum()),"manual_six_run_mean":float(d[mask][:,manual].mean()),"nnunet_mean":float(d[mask,nn].mean())}
    fig,ax=plt.subplots(figsize=(8,4.7))
    vref=matrices["validation"][2]
    ax.axvspan(vref.min(),vref.max(),color="grey",alpha=.12,label="Validation reference-volume range")
    for prefix,color,label,mark in [("baseline",colors[0],"Baseline mean over three seeds","o"),("dropout", "#d95f02","Dropout 0.1 mean over three seeds","^"),("nnunet","#238b45","nnU-Net seed 42","s")]:
        cols=[j for j,r in enumerate(runs) if r.startswith(prefix)]
        ax.scatter(ref,d[:,cols].mean(axis=1),color=color,label=label,marker=mark,alpha=.85)
    ax.set(xscale="log",xlabel="Reference tumor volume (ml)",ylabel="Dice",ylim=(-.03,1.03))
    ax.legend(fontsize=8); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(destination / "v003_volume_dice.pdf"); plt.close(fig)
    folds=[]; all_cv=[]
    for path in sorted(evidence.glob("outputs/t1/nnunet_cv/Dataset90*/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation/summary.json")):
        records=json.loads(path.read_text())["metric_per_case"]
        vals=[float(c["metrics"]["1"]["Dice"]) for c in records]
        assert np.isfinite(vals).all()
        folds.append({"dataset":path.parts[-5],"n":len(vals),"mean":float(np.mean(vals)),"min":min(vals),"below_0_1":sum(x<.1 for x in vals)})
        all_cv.extend({"case_id":Path(c["reference_file"]).name.removesuffix(".nii.gz"),**c["metrics"]["1"]} for c in records)
    assert len(folds)==5 and len(all_cv)==217 and len({c['case_id'] for c in all_cv})==217
    vals=[c["Dice"] for c in all_cv]
    output["cross_validation"]={"folds":folds,"n":len(vals),"mean":float(np.mean(vals)),"median":float(np.median(vals)),"patient_sd":float(np.std(vals,ddof=1)),"lowest_five":sorted(all_cv,key=lambda c:c["Dice"])[:5],"protocol":"Native nnU-Net evaluator, final checkpoint, default mirroring; distinct from seven-run common scorer."}
    (destination / "statistics.json").write_text(json.dumps(output,indent=2)+"\n")
    table=[r"\begin{table}[htbp]",r"\centering\small",r"\caption{Per-patient Dice on the 22 retained MC-RC cases, ordered by reference volume. B and D denote baseline and dropout 0.1, followed by the training seed; N denotes nnU-Net seed 42.}",r"\label{tab:per-patient-dice}",r"\begin{tabular}{lrrrrrrrr}",r"\hline",r"Case & ml & B42 & B43 & B44 & D42 & D43 & D44 & N42 \\",r"\hline"]
    for k in order:
        table.append(cases[k].replace("VS-MC-RC-","")+" & "+f"{ref[k]:.3f}"+" & "+" & ".join(f"{v:.3f}" for v in d[k])+r" \\")
    table += [r"\hline",r"\end{tabular}",r"\end{table}", ""]
    (destination / "appendix_per_patient.tex").write_text("\n".join(table))
    return output


def paired_bootstrap(evidence, destination, replicates=10000, seed=42):
    """Paired patient bootstrap of mean-Dice differences: one index draw per replicate, shared by both arms."""
    import numpy as np

    rows = list(csv.DictReader((evidence / "analysis/reports/t1_evaluation_20260912/patient_scores.csv").open()))
    runs = sorted({r["run"] for r in rows})
    manual = [r for r in runs if not r.startswith("nnunet")]
    contrasts = {"nnunet_s42 - manual_six_run_mean": (["nnunet_s42"], manual)}
    contrasts.update({f"dropout01_s{s} - baseline_s{s}": ([f"dropout01_s{s}"], [f"baseline_s{s}"]) for s in (42, 43, 44)})
    output = {"method": "paired patient percentile bootstrap", "seed": seed, "replicates": replicates,
              "confidence": 0.95, "unit": "patient; same resampled patients for both arms of every contrast",
              "scope": "Conditional on fitted models and evaluated cohort; not training or site uncertainty.",
              "cohorts": {}}
    for cohort in ("validation", "mcrc"):
        by = {(r["case_id"], r["run"]): float(r["dice"]) for r in rows if r["cohort"] == cohort}
        cases = sorted({c for c, _ in by})
        assert len(by) == len(cases) * len(runs)
        d = np.array([[by[c, r] for r in runs] for c in cases])
        indices = np.random.default_rng(seed).integers(0, len(cases), size=(replicates, len(cases)))
        output["cohorts"][cohort] = {}
        for name, (a, b) in contrasts.items():
            diff = d[:, [runs.index(r) for r in a]].mean(axis=1) - d[:, [runs.index(r) for r in b]].mean(axis=1)
            boot = diff[indices].mean(axis=1)
            output["cohorts"][cohort][name] = {"n": len(cases), "mean_difference": float(diff.mean()),
                                               "ci95": np.quantile(boot, [.025, .975]).tolist(),
                                               "fraction_replicates_above_zero": float((boot > 0).mean()),
                                               "patients_a_better": int((diff > 0).sum()), "patients_b_better": int((diff < 0).sum())}
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "paired_bootstrap.json").write_text(json.dumps(output, indent=2) + "\n")
    return output
