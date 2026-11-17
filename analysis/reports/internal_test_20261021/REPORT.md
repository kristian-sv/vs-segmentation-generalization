# Internal-test evaluation of the slab/thick-slice augmentation (RQ3) — report

Evaluated 23 October 2026 under `PROTOCOL.md`, which was committed alone before any inference or scoring
(`20901f9`, 2026-10-21 14:36 +0200). Data: the 24 reserved VS-SEG internal-test patients. No model was trained, tuned
or selected on these patients. Both primary rules pass. This report also covers the caveats that limit what the passes
mean (§3).

## 1. Answers

**Did failures decrease? Yes, under the simulated combined slab + thick-slice acquisition (S40T3).**
- Plain arm: mean Dice 0.579, with 6.5 empty predictions (mean of 8 for s42 and 5 for s43).
- Augmented arm: mean Dice 0.885, with 0 empty predictions (0 for both seeds).
- Difference: **+0.306, paired bootstrap 95 % CI [+0.183, +0.438]**. 22 of 24 patients improved.
- Both seeds agree: s42 +0.290 [+0.147, +0.446]; s43 +0.322 [+0.174, +0.478].
- **Rule (b): PASS ("beneficial")** — the CI lies above 0 and the augmented arm has fewer empty predictions.

**Was source performance preserved? Yes, on the unchanged internal-test scans (C0).**
- Plain arm mean Dice 0.916; augmented arm 0.935.
- Difference: **+0.018, CI [+0.003, +0.042]**, against a margin of −0.01.
- **Rule (a): PASS.** The stricter secondary reading (CI lower bound ≥ −0.01) also holds.
- No empty predictions in either arm.
- Most of the gain comes from one patient (§3), so the result is "no harm", not "a gain".

| Rule | Condition | Plain arm | Augmented arm | Difference [95 % CI] | Empty (plain / aug) | Outcome |
|---|---|---:|---:|---|---|---|
| (a) | C0, unchanged | 0.916 | 0.935 | +0.018 [+0.003, +0.042] | 0 / 0 | **pass** (≥ −0.01) |
| (b) | S40T3 | 0.579 | 0.885 | +0.306 [+0.183, +0.438] | 6.5 / 0 | **pass** (beneficial) |

Arm value = per-patient mean of the two seeds' Dice. Arm empty count = mean of the two seeds' counts. Bootstrap:
10,000 paired patient resamples, seed 42 (`model_insight.bootstrap_ci`). Source: `bootstrap.json`.

## 2. All conditions, per model (internal test; secondary, descriptive)

Mean Dice per model; the arm difference with CI; empty predictions and FP components per model (order: plain s42,
plain s43, slabthick s42, slabthick s43). An FP component is a 26-connected predicted component that does not touch the
reference, the same definition as `case_metrics`. Full table: `tables/summary_by_condition.csv`.

| Condition | p42 | p43 | a42 | a43 | Arm Δ [95 % CI] | Empty | FP components |
|---|---:|---:|---:|---:|---|---|---|
| C0 | 0.915 | 0.918 | 0.944 | 0.925 | +0.018 [+0.003, +0.042] | 0/0/0/0 | 1/1/0/**8** |
| S50 | 0.900 | 0.933 | 0.943 | 0.940 | +0.025 [−0.002, +0.071] | 1/0/0/0 | 0/0/0/0 |
| S40 | 0.778 | 0.869 | 0.937 | 0.941 | +0.115 [+0.036, +0.218] | 2/0/0/0 | 0/0/1/0 |
| S33 | 0.739 | 0.775 | 0.916 | 0.941 | +0.171 [+0.069, +0.293] | 4/1/0/0 | 0/1/4/0 |
| S40edge | 0.836 | 0.848 | 0.943 | 0.939 | +0.099 [+0.023, +0.201] | 1/0/0/0 | 1/0/3/0 |
| T3 | 0.823 | 0.857 | 0.864 | 0.843 | +0.013 [−0.014, +0.044] | 1/0/0/0 | 1/1/2/6 |
| T4 | 0.828 | 0.845 | 0.878 | 0.834 | +0.020 [−0.013, +0.063] | 1/1/0/1 | 0/0/1/3 |
| **S40T3** | 0.580 | 0.578 | 0.869 | 0.900 | **+0.306 [+0.183, +0.438]** | **8/5/0/0** | 0/0/2/0 |
| AP | 0.865 | 0.887 | 0.944 | 0.929 | +0.061 [+0.018, +0.116] | 0/0/0/0 | 4/5/0/0 |
| COR_thin, re-stacked | 0.811 | 0.788 | 0.938 | 0.920 | +0.130 [+0.060, +0.219] | 1/0/0/0 | 3/16/2/6 |
| COR_thick, re-stacked | 0.823 | 0.839 | 0.917 | 0.881 | +0.068 [+0.026, +0.127] | 1/0/0/1 | 0/7/2/0 |
| COR_thin, as stored | 0.193 | 0.244 | 0.799 | 0.856 | +0.609 [+0.479, +0.737] | 15/12/0/1 | 0/1/2/0 |
| COR_thick, as stored | 0.602 | 0.702 | 0.779 | 0.839 | +0.156 [+0.072, +0.256] | 7/4/1/2 | 0/0/1/0 |

- **Slab conditions:** the augmented arm is better on every slab condition. The CI excludes 0 for S40, S33, S40edge and
  S40T3; it touches 0 only for the mildest slab, S50.
- **Thick slices alone (T3, T4):** no clear effect. The CIs include 0, and seed 43 points slightly negative (T3 −0.014,
  T4 −0.011). This matches validation (T3 +0.010, T4 +0.001).
- **Coronal, re-stacked (the protocol input):** both conditions improve, and the CIs exclude 0.

## 3. Caveats to read with the passes

1. **The C0 gain rests mainly on VS-SEG-004.** Its arm difference is +0.253 (plain 0.46 / 0.33; slabthick s42 0.86, s43
   0.44). The other 23 patients average +0.008, and the median per-patient difference is +0.002 (range −0.007 to
   +0.064). Rule (a) still passes without this patient, but the honest reading is **no detectable harm**, not
   improvement. VS-SEG-004 is a hard case for every model: the six manual runs score 0.00–0.22. It is also the test
   case whose CV-model Dice of 0.000 had been seen in a viewer before this study (PROTOCOL.md §2).
2. **slabthick s43 makes more small false-positive components on unchanged scans:** 8 components in 7/24 patients,
   against 1 in 1/24 for each plain seed and 0 for slabthick s42. All are tiny (0.001–0.013 ml) and none emptied or
   broke a segmentation; the s43 Dice in these cases is 0.895–0.979. Validation showed the same direction (s43: 4 in
   4/24). It is seed-specific, so two seeds cannot tell whether it is a property of the augmentation. No rule covers
   it, and it is reported, not explained away.
3. **(b) is evidence on simulated degradations of same-source patients, not on real slab acquisitions.** The S40T3
   perturbation lies inside the augmentation's training ranges by construction (PROTOCOL.md §9), so a gain there is
   expected if training worked at all. The new information is that it holds on patients that played no part in
   development, with numbers almost identical to validation (+0.306 vs +0.315). It does not replace an external cohort.
4. The plain arm is somewhat lower on the internal test than on validation (C0 0.916 vs 0.929). Validation selected
   these checkpoints, so a small optimistic bias there is expected. The augmented arm is unchanged (0.935 vs 0.934).

## 4. Next to the existing results

| Unit | Validation C0 (24) | **Test C0 (24)** | Validation S40T3 | **Test S40T3** | MC-RC 13 slab | MC-RC all 22 |
|---|---:|---:|---:|---:|---:|---:|
| plain s42 | 0.932 | **0.915** | 0.546 (9 empty) | **0.580 (8)** | 0.361 (6) | 0.543 |
| plain s43 | 0.925 | **0.918** | 0.586 (5) | **0.578 (5)** | 0.367 (4) | 0.550 |
| slabthick s42 | 0.928 | **0.944** | 0.866 (0) | **0.869 (0)** | 0.752 (0) | 0.775 |
| slabthick s43 | 0.940 | **0.925** | 0.895 (0) | **0.900 (0)** | 0.704 (1) | 0.747 |
| Plain arm | 0.929 | **0.916** | 0.566 | **0.579** | 0.364 (5) | 0.546 |
| Augmented arm | 0.934 | **0.935** | 0.880 | **0.885** | 0.728 (0.5) | 0.761 |
| Arm Δ [95 % CI] | +0.006 [−0.003, +0.017] | **+0.018 [+0.003, +0.042]** | +0.315 [+0.184, +0.447] | **+0.306 [+0.183, +0.438]** | +0.364 [+0.214, +0.513] | +0.215 |

Validation: `model_insight_20261012/stress_test/followup_stress_summary.csv`. MC-RC: `mcrc_followup/tables/summary.csv`
(re-stacked inputs; empty counts in brackets). Full side-by-side for every condition:
`tables/comparison_with_validation_and_mcrc.csv`.

**Seven original runs on the unchanged internal test (RQ1 context)** — `tables/unchanged_all_runs_summary.csv`:

| Run | Mean | Median | SD | Dice < 0.1 | Empty | FP comp. (cases) |
|---|---:|---:|---:|---:|---:|---|
| baseline s42 / s43 / s44 | 0.886 / 0.884 / 0.893 | 0.936 / 0.929 / 0.933 | 0.19 / 0.19 / 0.15 | 1 / 1 / 0 | 1 / 1 / 0 | 5 (4) / 8 (7) / 7 (5) |
| dropout01 s42 / s43 / s44 | 0.887 / 0.885 / 0.887 | 0.932 / 0.931 / 0.934 | 0.19 / 0.19 / 0.18 | 1 / 1 / 1 | 1 / 0 / 0 | 5 (5) / 6 (4) / 2 (2) |
| nnU-Net s42 (= plain s42) | 0.915 | 0.949 | 0.11 | 0 | 0 | 1 (1) |

- nnU-Net s42 − manual six-run mean: **+0.028 [+0.003, +0.067]**; 20 patients higher, 4 lower. On validation the same
  contrast was +0.013 [−0.008, +0.027].
- dropout01 − baseline per seed: +0.001 [−0.001, +0.004], +0.000 [−0.002, +0.003], −0.006 [−0.021, +0.004]. No
  consistent difference, as on validation and MC-RC.
- The manual runs' only empty or < 0.1 predictions are on VS-SEG-004.

## 5. Provenance

| Item | Value |
|---|---|
| Protocol commit | `20901f941ce257d2b5394b30d8a2ed9279c806cc` — 2026-10-21 14:36:00 +0200 (PROTOCOL.md only) |
| Code + interim stop report | `200d7b2` — 2026-10-22 00:02:43 +0200 |
| Code used by inference/scoring | `code/` committed in `200d7b2` (recorded HEAD of every inference task, `provenance/code_commit_task*.txt`); sha256 of all three files equal to those written in PROTOCOL.md §10 before the protocol commit, so the code is unchanged since the protocol |
| Preflight (headers/hashes only) | Slurm 2591718, 2026-10-21 |
| Build stress inputs | Slurm 2591723 (4 min): 264 inputs, 48 re-stacked, no truncation |
| First inference attempt | Slurm array 2591724, 2026-10-22 00:00 — all 10 failed at container start (image missing) |
| Container verification | Slurm 2602850, 2026-10-23: pip freeze inside the restored image identical to the training records |
| Inference | Slurm array 2602906, 2026-10-23 11:25–11:42, all 10 COMPLETED (nnU-Net 3.5 min/task, manual 1.2–1.5 min) |
| Scoring | Slurm 2602993 (20.6 min): 1,248 stress rows + 240 unchanged-scan rows; all coverage/grid/consistency checks passed |
| Summary | Slurm 2602994 |
| Container | `/cluster/work/support/container/pytorch_nvidia_25.06_arm64.sif`, sha256 `3c081bd7…cc620` (first recorded 2026-10-23; no earlier hash exists) |
| Checkpoints (sha256, verified in each task) | plain_s42 `61015f70…`, plain_s43 `b2d2d37a…`, slabthick_s42 `f4a331f0…`, slabthick_s43 `8210dd3f…`; manual runs as `t1_evaluation_20260912/runs.csv` |
| Test manifest | `test_manifest.csv` sha256 `4c55817c…`; image hashes match the 22 Sep 2026 review record |

### Deviations from the protocol, stated plainly

1. **Container outage.** The first inference array failed because the shared image had been removed. Following the
   stop rule, nothing was substituted. When the image reappeared at the same path, I first checked it before any
   inference (job 2602850): `pip freeze` in the nnU-Net environment is identical to the `environment.txt` of all four
   training runs, and the manual environment is identical to baseline s42's. The image's own hash cannot be compared,
   because none was recorded before. The verification job was not in the protocol; it adds a check and changes nothing.
2. Empty output folders left by the failed tasks were deleted before the unchanged resubmission, as protocol §7 allows.
3. The validation stress predictions for plain s42 were made with `nnUNetv2_predict -d 901`. Here all four models used
   `nnUNetv2_predict_from_modelfolder` on the same model folder, as protocol §4 specifies and as MC-RC step B did.
   This is the same checkpoint and the same predictor settings.

No other departures. Every check in the code passed: hashes, patient coverage, grids, binary masks, and C0 stress Dice
equal to the common-scorer Dice.

## 6. What I found in the repository that changed the approach

- **The test split was not untouched.** On 22 Sep the five-fold CV fold-0 model exported mirrored masks for all 24 test
  patients, and the author's notes record seeing VS-SEG-004 at Dice 0.000 for that model. None of the evaluated models
  used test data (every `splits_final.json` checked). This is disclosed in the protocol, and it matters for caveat 1.
- **`data_lock.json` deliberately holds no hashes for test files.** Image hashes were checked against the 22 Sep review
  manifest instead. Reference hashes were first recorded at preflight, before the protocol commit.
- **`src/evaluate.py manifest` refuses the test split**, so `code/preflight.py` wrote the manifest in the identical
  format and verified it with the common `read_manifest`.
- **On validation, inputs were used as stored except for the coronal conditions.** VS-SEG is stored in the training
  orientation, so stored and re-stacked inputs are the same file. The test run follows this, using re-stacked input for
  COR_thin/COR_thick.
- **No new perturbation, scoring or FP code was written.** `code/internal_test.py` points the existing
  `model_insight` functions (`stress_build`, `stress_unit`, `case_metrics`, `restore_from_training_order`,
  `bootstrap_ci`) and `src/evaluate.py score` at the new folder. Only the loops that call them are new.
- **Other data were inventoried and not scored.** Two further direct-T1 MC-RC examinations exist (VS-MC-RC-016
  1991-09-16 and VS-MC-RC-057 1989-11-05); both are earlier timepoints of patients in the already-inspected 13-case
  slab cohort. 40 MC-RC T1 examinations belong to VS-SEG training/validation patients, and none to the test split.
  The 27 T2-derived MC-RC cases remain deferred. Verifying them would need a documented T1/T2 alignment check, which
  looks feasible for the 25 pairs that share a DICOM frame of reference. MC-RC2 has not been downloaded. Within the
  data at hand, the internal test was the only clean source left.

## 7. Proposals for further evidence (not started)

- **More seeds:** plain and augmented seeds 44–46, giving five per arm. Six runs at about 16.5 GPU-hours each
  (measured: 16.8 h, 16.1 h, 16.4 h) comes to **about 100 GPU-hours**, plus about 25 min of inference to repeat this
  evaluation. This would replace the reused September plain s42 with a matched set. It would also show whether
  slabthick s43's extra small FPs (caveat 2) and the T3/T4 seed split are seed noise.
- **Real slab acquisitions on unseen patients:** MC-RC2, under the protocol in
  `independent_test_feasibility_mvp/REPORT.md`, or a verified subset of the 27 T2-referenced MC-RC cases. Either would
  give real rather than simulated slabs and need its own preregistration.

## Files

`tables/summary_by_condition.csv`, `tables/per_case_primary.csv` (C0 and S40T3 per patient),
`tables/per_case_unchanged_all_runs.csv`, `tables/unchanged_all_runs_summary.csv`,
`tables/comparison_with_validation_and_mcrc.csv`, `stress_test/stress_test.csv` (every scored mask; Dice, volumes, empty,
FP components, HD95), `scores/*_patients.csv` / `*_summary.json` (common scorer), `bootstrap.json`, `preflight.json`,
`provenance/`. Masks and images stay local (`.gitignore`).
