# Trustworthy T1 testing MVP — 7 September 2026

**Overlap correction is complete. Development can resume using training/validation only. This remains a previously inspected, retrospective evaluation cohort, not an untouched test.** No training, fresh inference, image/label edits, checkpoint changes, or historical-score replacement was performed.

The existing September audit was rerun with its output redirected to `/tmp/trustworthy-testing-original-audit`; all five generated files matched the previous audit byte-for-byte. Current inputs reproduce **51 patients: 25 shared with VS-SEG training, 4 with validation, 0 with internal test, and 22 retained**. Five selected acquisitions duplicate training series: `VS-SEG-095`, `111`, `117`, `188`, `199`.

The separate manifest is `metadata/evaluation/mcrc_t1_patient_independent.csv`. `cohort_decisions.csv` records all 51 decisions, reasons, canonical identities, and selected source series/labels. Identity evidence joins the original split, both VS-SEG masters, the source spreadsheet, directory UID mapping, and `series_overlap.csv` across **all timepoints before selection**. Every scan of a documented shared person is ineligible, even if the selected scan itself is different. The existing one-latest-scan-per-patient cohort is preserved. Counts are calculated from inputs, not hardcoded.

Current source patient/path IDs and UID-linked directory IDs have no nonidentical aliases; source DICOM PatientID agrees for all 51 selected series. The code supports these explicit identity links and refuses ambiguous latest-series choices. It does not assume equal numeric suffixes imply identity, or reuse the preprocessing script's unsafe date/filename fallback. Undocumented aliases cannot be ruled out. `provenance.json` records the input hashes and identity policy.

**Corrected Dice: reaggregation of the six existing per-case score tables.** Each model covers exactly the same 22 retained patients (132 rows total). Scores remain keyed by patient and run. Full precision, medians and sample standard deviations are in `summary_reaggregated.csv`; individual values are in `per_case_reaggregated.csv`.

| Configuration | Historical mean, n=51 | Retained mean, n=22 | Dice < 0.1, n=22 |
|---|---:|---:|---:|
| Lower dropout | 0.528989 | 0.456537 | 7 |
| DiceCE | 0.491984 | 0.432281 | 10 |
| Spatial + elastic augmentation | 0.522920 | 0.414276 | 10 |
| Baseline | 0.447505 | 0.403223 | 9 |
| Lower learning rate | 0.430691 | 0.378534 | 10 |
| Larger patch | 0.455967 | 0.327979 | 11 |

**Evaluation safeguards.** Both existing MC-RC evaluators now require an explicit audited manifest for T1, retain patient IDs in scores/scanner joins, and protect existing output files. The loader rejects duplicate patient/canonical/series IDs, known overlap, unknown or altered provenance, incomplete cohorts, and missing image/label pairs. `train.py` and `run_experiment.py` default to training/validation only, without constructing test loaders, reloading a checkpoint for testing, or logging test metrics. Final internal-test and MC-RC evaluation remain separately available through `--evaluate-internal-test` and `--evaluate-mcrc`; the latter also requires `--evaluation-manifest` for T1. New run source snapshots include the guard modules.

Six lightweight test groups passed, including malformed manifests, alias evidence at another timepoint, score-table coverage, evaluator rejection before image loading, CLI defaults, and all four combinations of final-evaluation options. The actual training entry function was exercised with training, inference, loaders, and W&B mocked. No training job was launched; GPU inference itself was not tested.

**Geometry and visual coverage.** `source_geometry.csv` covers **51/51 selected T1 source image/label pairs, including 22/22 retained**. Image RAS coordinates were reconstructed from every selected classic DICOM slice's position, orientation and spacing, with UID, duplicate-instance and slice-consistency checks. Untouched source-label affines were compared after axis permutation/flip using the corresponding affine transform, not an overwritten processed label header. All source grids agree within the stated **0.1 mm corner tolerance**. The largest source-grid corner difference is **0.051825 mm**, `VS-MC-RC-100`; the largest processed-image/DICOM corner difference is **0.002305 mm**, `VS-MC-RC-081`. All processed pair headers also match. These findings do not establish segmentation correctness throughout every volume.

Four cases received three-plane source and saved-prediction review (24 displayed planes total), covering **3/22 retained patients plus 1 excluded patient**. In all four, processed image voxels equal the independently read source DICOM voxels, and processed masks equal source masks after axis reorientation. `visual_review.csv` records coverage, numeric checks and observations:

| Case | Selection and observation |
|---|---|
| VS-MC-RC-044 | Good retained control: source mask follows the visible lesion; prediction has small boundary differences. |
| VS-MC-RC-012 | Poor retained case: source mask follows the large lesion; prediction has almost no overlap. Anisotropic source slices limit detail. |
| VS-MC-RC-024 | Poor/suspicious retained case: source mask follows the lesion; prediction captures internal fragments. |
| VS-SEG-199 | Good prediction but duplicated training acquisition: useful for inspection, ineligible as unseen-patient evidence. |

No gross source-mask displacement was seen in these samples, and no substantial alignment repair was indicated. **Unresolved:** the saved inspection mask for `VS-MC-RC-024` gives Dice **0.124099**, versus **0.124230** in the historical score table (difference −0.0001313). The other three inspected exports agree to floating-point precision. Export/evaluator provenance must be reconciled before treating that export as the exact historical prediction; the cause is unknown. Reaggregation preserves the historical table. The preprocessing equal-shape/affine-copy shortcut remains a risk for future inputs; these checks validate current source grids, not that shortcut generally. No blanket resampling was performed. Remaining patients, other modalities and VS-SEG source alignment did not receive visual review here.

**Reproduce from the repository root** using existing environments (no installs):

```bash
python3 analysis/prepare_t1_evaluation.py --check
python3 analysis/check_evaluation_guards.py
myenv/bin/python analysis/audit_t1_source_geometry.py --output-dir /tmp/t1-geometry-recheck
myenv/bin/python analysis/inspect_t1_alignment.py --output-dir /tmp/trustworthy-testing-visuals
```

Omit `--check` from the first command to regenerate only the new cohort/reaggregation products. Geometry and visual scripts run on CPU in the existing container environment. Panels stay in `/tmp/trustworthy-testing-visuals`; no images are committed. Temporary outputs can be regenerated if removed.

For a deliberately authorized **future** T1 inference run, the existing evaluator accepts:

```bash
myenv/bin/python src/scripts/reevaluate_mcrc_modality.py \
  --models t1_baseline_20260324_115013 \
  --master-csv metadata/vs_mcrc_master.csv \
  --evaluation-manifest metadata/evaluation/mcrc_t1_patient_independent.csv \
  --scanner-csv data/splits/vsmcrc_scanner_groups.csv --image-col t1_path \
  --output outputs/final_t1_new_summary.csv \
  --per-case-output outputs/final_t1_new_per_case.csv
```

This command was **not run**. Use new output filenames. Effective old-model configuration still depends partly on current defaults/experiment overrides; freeze and review that configuration before any future inference.

**Remaining evaluation limits:** historical test metrics, rankings, notes and visual cases were already exposed, and the former training entry point automatically logged internal-test/MC-RC results. Which decisions they influenced is **unknown**. Removing overlap cannot undo that exposure; group differences also prevent interpreting score changes as the causal effect of overlap. Freeze hypotheses, selection rules, preprocessing and evaluation timing before further experiments. Restrict ordinary experiment/agent access to training/validation information; the opt-in flags are workflow controls, not an access-control boundary around old files, Git history or W&B. A genuinely untouched-test claim remains unmet. The six configurations are not repeated-seed evidence; the previous audit confirms six unique T1 checkpoint hashes across twelve paired files.
