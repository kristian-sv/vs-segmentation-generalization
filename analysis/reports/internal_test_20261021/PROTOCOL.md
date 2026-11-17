# Protocol — internal-test evaluation of the slab/thick-slice augmentation (RQ3)

Written 21 October 2026 and committed **alone, before any inference or scoring** on these data. Results are reported
whatever the outcome. Nothing in this file changes after results are seen; any departure is listed as a deviation in
REPORT.md.

## 1. Question

Does the slab + thick-slice augmentation (`model_insight_20261012/PREREGISTRATION.md`) (1) reduce segmentation failures
under simulated limited-coverage / thick-slice acquisitions and (2) leave performance on normal VS-SEG scans intact,
when tested on patients that played no part in training, checkpoint selection or augmentation design?

Existing evidence used data that also informed or selected the models: the 24 validation patients selected the
checkpoints and defined the stress test; the 13 MC-RC slab cases were inspected earlier and supplied the augmentation
ranges.

## 2. Data

**Internal test: the 24 reserved VS-SEG patients** (`experiments/data_lock.json`, split `test`):
004, 005, 015, 019, 027, 036, 072, 090, 093, 099, 100, 102, 108, 119, 123, 138, 148, 149, 161, 181, 182, 190, 215, 244
(all `VS-SEG-`). Images and references: `data/processed/VS_SEG_T1/<id>/{T1,label}.nii.gz`, the same processed source
and label version as the development data.

Pre-protocol checks (`code/preflight.py`, Slurm job 2591718; image headers and file bytes only, reference masks hashed
but **not opened**):
- the 24 IDs equal the data-lock test split; `check_lock()` passed (split file and master hashes unchanged);
- all 24 images are stored LPS with millimetre units (23 × 512×512×120 at 0.41×0.41×1.5 mm; 1 × 512×512×160 at 1.0 mm);
- the data lock deliberately stores **no** hashes for test bytes. Image hashes match the independent record of
  22 Sep 2026 (`outputs/t1/review/vsseg242_nnunet_cv/review_manifest.csv`, commit c536fae). No earlier record of the
  reference-mask hashes exists; they are first recorded here (`preflight.json`, `test_manifest.csv`
  sha256 4c55817c…). File modification times: labels 2026-03-11, images 2026-03-23 (before the data lock).

**Exposure history (disclosed, not hidden).** None of the four evaluated models, nor the seven original runs, used any
test patient (`splits_final.json` of every Dataset901 run: 193/24, 0 test IDs; checked 21 Oct). However, the test images
are not untouched: on 22 Sep 2026 the five-fold CV fold-0 model (not evaluated here) produced mirrored review masks for
all 24 (`outputs/t1/review/vsseg242_nnunet_cv/`; no Dice computed there), and the author's notes record that a viewer
showed **VS-SEG-004 with Dice 0.000 for that CV model**. So at least one test case outcome (for a different model) has
been seen. This did not feed into the augmentation design or the selection of any model below.

**Stress conditions.** The same 11 conditions as the validation stress test, built by the unchanged
`model_insight.stress_build` (`CONDITIONS`, `perturb`, `block_average`; fixed 12 Oct 2026, commit fbfee1f) applied to
the test manifest: C0 (unchanged), S50, S40, S33, S40edge, T3, T4, S40T3, AP, COR_thin, COR_thick. As on validation,
S40edge alternates superior/inferior by sorted patient ID, and each prediction is scored against the identically
perturbed reference on its own grid. `code/internal_test.py build` only redirects the builder's output folder and input
manifest; no perturbation code is re-implemented.

## 3. Models

Primary — the four augmentation-experiment models (`followup_checkpoints.csv`; `checkpoint_best.pth`; hashes verified
21 Oct and re-verified in each GPU task before inference):

| Model | Arm | Checkpoint sha256 |
|---|---|---|
| plain_s42 (Sept nnU-Net, `outputs/t1/nnunet/…/nnUNetTrainer__nnUNetPlans__3d_fullres`) | plain | 61015f706d0b4347fc6ba9d506a2e8a6b3c2b61de083e55383c98be63d11fe5a |
| plain_s43 (`outputs/t1/nnunet_slab/plain_s43/…`) | plain | b2d2d37a63c4d32503e00ad1ea90b3b3c4f0415cf47e0811cad1a468741100c8 |
| slabthick_s42 (`outputs/t1/nnunet_slab/slabthick_s42/…/nnUNetTrainerSlabThick…`) | augmented | f4a331f01a3fb7ebb9ec8cc8e73d62153ecb758c5f2a0af4393c18274c5b2f50 |
| slabthick_s43 (`outputs/t1/nnunet_slab/slabthick_s43/…/nnUNetTrainerSlabThick…`) | augmented | 8210dd3f3ee74aff352ac3a7c6e0b1ac99f0d066fc2ce07e8004188c7b18c88c |

Secondary (RQ1 context) — the seven original runs (`t1_evaluation_20260912/runs.csv`): baseline s42/43/44
(446907…, 68f01e…, 1e1279…), dropout01 s42/43/44 (74a267…, see runs.csv), nnU-Net s42 (= plain_s42, same checkpoint
file and hash; its unchanged-scan predictions are reused, not re-run).

Never used: the five-fold CV models (`outputs/t1/nnunet_cv/`), the review exports, the smoke runs.

## 4. Inference and scoring (same protocol as steps A and B)

- nnU-Net: `nnUNetv2_predict_from_modelfolder -m <model folder> -f 0 -chk checkpoint_best.pth --disable_tta -npp 2 -nps 2`
  (identical to `model_insight.slurm followup-predict`). No mirroring/TTA, no ensembling, no postprocessing.
- Inputs: every condition as stored, plus COR_thin and COR_thick re-stacked to the training voxel order
  (`write_training_order`, permutation/flip only), as on validation. All 24 test images are stored LPS (= training
  order), so for C0 and all axial conditions the stored input *is* the re-stacked input. For COR_thin/COR_thick the
  **re-stacked** input is the protocol input (as for MC-RC); as-stored is reported alongside for comparison with the
  validation table. Re-stacked masks are mapped back with `restore_from_training_order`.
- Manual runs (secondary only): `src/evaluate.py manual-export` with the unchanged test manifest (unchanged scans only),
  identical to `experiments/evaluate.slurm`.
- Scoring: `src/evaluate.py` (`score`, `patient_metrics`: binary Dice, volumes, empty status, all grid/coverage checks
  before any score) and `model_insight.case_metrics` / `stress_unit` for empty predictions and FP components
  (26-connected predicted components with no voxel overlapping the reference). For C0, the stress-test Dice must equal
  the common-scorer Dice exactly, or scoring stops.
- **Arm value per patient** = mean of its two seeds' Dice (no mask ensembling). **Arm empty count** = mean of the two
  seeds' empty-prediction counts (as in `mcrc_followup/tables/summary.csv`).
- **Bootstrap**: `model_insight.bootstrap_ci` — paired patient percentile bootstrap of the mean augmented-minus-plain
  difference, 10,000 resamples of the 24 patients (sorted IDs), `numpy.random.default_rng(42)`.

## 5. Primary outcomes and decision rules

**(a) Source performance preserved** — unchanged internal test (C0): augmented-arm mean Dice minus plain-arm mean Dice
**≥ −0.01** → *pass*; otherwise *fail*. (Point estimate, same rule as the validation check.)

**(b) Fewer failures** — internal test, combined condition **S40T3** (40 mm head-feet slab + 3.3 mm slices):
- *beneficial (pass)*: bootstrap 95 % CI of the arm difference lies entirely above 0 **and** augmented-arm empty count
  ≤ plain-arm empty count;
- *inconclusive (fail)*: CI includes 0 — reported as estimate and CI, not as evidence of no effect;
- otherwise *fail*, stated as "CI excludes 0 below" or "CI above 0 but more empty predictions".

## 6. Secondary outcomes (descriptive; no multiplicity correction, no pass/fail)

1. (a) restated as non-inferiority: is the C0 CI lower bound ≥ −0.01?
2. Arm difference with CI, empty counts, Dice < 0.1 counts and FP components (count and cases with ≥ 1) per model and arm
   for every condition (C0, S50, S40, S33, S40edge, T3, T4, S40T3, AP, COR_thin, COR_thick; re-stacked and as stored).
3. Per-seed contrasts slabthick_sXX − plain_sXX with CI, per condition.
4. Per-model internal-test results next to the existing validation (step A) and MC-RC (step B) results.
5. Seven original runs on the unchanged test: mean, median, patient SD, Dice < 0.1, empty predictions, FP components;
   paired bootstrap of nnU-Net s42 − manual six-run mean and dropout01 − baseline per seed.

## 7. Stop rules

A failed checkpoint hash, patient-ID/manifest mismatch, missing/extra prediction, grid mismatch or non-binary mask stops
that part; it is reported, not worked around. A failing GPU task may be resubmitted unchanged (same inputs, empty
output folder); that is logged.

## 8. Other data inventoried (not scored)

- **Two further MC-RC examinations with direct T1 references** exist locally (DICOM + `seg_T1`): VS-MC-RC-016
  (1991-09-16, `t1_vibe_fs_tra`) and VS-MC-RC-057 (1989-11-05, `t1_se_tra_3mm`). Both are other timepoints of patients
  already in the inspected 22-case cohort (both in the 13 primary slab cases) and have not been converted with the
  frozen preparation. They are not independent evidence and are **not scored**.
- The 27 deferred MC-RC cases with T2-derived references: **not scored** (reference validity unverified).
- 40 MC-RC T1 examinations under `VS-SEG-` identities belong to 36 training and 4 validation patients (none to the
  test split): not independent, not scored.
- MC-RC2: not downloaded; not scored.

## 9. Known limitations, stated in advance

- The test patients come from the same source and centre as training. (b) therefore tests **simulated** acquisitions on
  unseen patients, not real slab acquisitions; it cannot replace an external cohort.
- The stress conditions resemble the augmentation by construction (S40T3 lies inside the training ranges).
- 24 patients, two seeds per arm; plain s42 is the September model trained earlier; GPU inference is not bit-exact
  across nodes (1-voxel differences were seen before).

## 10. Code and records

`code/preflight.py` (sha256 3323df32…), `code/internal_test.py` (975c8c1b…), `code/internal_test.slurm`
(7cef0470…) as of this commit; they import the unchanged `analysis/model_insight.py` and `src/evaluate.py`. Outputs:
`stress_test/` (inputs, masks, `stress_test.csv`), `original_runs/`, `scores/`, `tables/`, `bootstrap.json`, `REPORT.md`.
Large image files stay local (`.gitignore`); tables and records are committed.
