# Original-label T1 evaluation — 12 September 2026

## Results

All seven models completed common validation and retrospective external evaluation.
These are **original-training-label results**, providing the reference for a later
controlled training-label correction study.

| Run | Validation Dice (24) | MC-RC Dice (22) | MC-RC Dice < 0.1 | MC-RC empty predictions |
|---|---:|---:|---:|---:|
| Baseline, seed 42 | 0.918269 | 0.412764 | 8 | 2 |
| Baseline, seed 43 | 0.918960 | 0.438854 | 6 | 2 |
| Baseline, seed 44 | 0.917319 | 0.441903 | 7 | 0 |
| Lower dropout, seed 42 | 0.920488 | 0.437004 | 8 | 3 |
| Lower dropout, seed 43 | 0.919889 | 0.367154 | 12 | 4 |
| Lower dropout, seed 44 | 0.920254 | 0.450731 | 6 | 0 |
| nnU-Net, seed 42 | 0.932344 | 0.482683 | 9 | 7 |

Across the three manual seeds, baseline means are **0.918183 validation / 0.431174
MC-RC**, and lower-dropout means are **0.920211 / 0.418296**. External mean Dice has
sample SD 0.016016 across baseline seeds and 0.044819 across lower-dropout seeds.
All validation runs have zero cases below Dice 0.1.

Lower dropout exceeds baseline in all three paired validation runs by a mean
0.002028 (0.203 percentage points). On MC-RC, paired differences for seeds 42/43/44
are +0.024240, -0.071700 and +0.008828; the average difference is -0.012877.
Thus the small validation advantage does not establish a consistent external
advantage. The predeclared validation-based selection still favours lower dropout;
external scores are not used to switch the selected configuration or choose a seed.

nnU-Net has the highest observed mean on both cohorts, but 9/22 external cases have
Dice below 0.1, including seven completely empty predictions. Its higher mean does
not imply fewer failures than the baseline (6–8/22 across seeds). All setups show
substantially poorer transfer than validation performance. This small, previously
inspected cohort and one nnU-Net seed do not establish a universally superior
pipeline or statistical significance. The comparison does not identify the cause
of failures or establish that reference-label errors caused them.

`results.csv` contains each run's mean, median, patient SD and failure count.
`patient_scores.csv` contains the individual results. Patient SD describes
variation between patients; it is not the variability between training seeds.
All **322 predictions** (168 validation + 154 MC-RC) passed the shared scorer's
coverage, binary-mask and original-grid checks. Source manifests and checkpoint hashes match the frozen
records, and means were independently recalculated from the per-patient tables.
Three-plane review of seed-42 predictions from all three setups covered VS-SEG-002
and VS-SEG-122 (18 displayed planes). The first shows close agreement; the latter,
nnU-Net's lowest-scoring validation case, shows extra foreground relative to the
reference. No gross export displacement was seen in these samples. This is a
limited export check, not clinical verification of the reference annotations.
External review covered VS-MC-RC-009 (manual undersegmentation and empty nnU-Net
prediction) and VS-MC-RC-028 (successful control), with all three seed-42 setups
in three planes each. These selected examples illustrate differing outcomes;
they are not an exhaustive visual audit or estimates of failure frequency.

## Frozen comparison

The seven selected checkpoints in `runs.csv` were evaluated on the same
24 VS-SEG validation patients and 22 previously inspected MC-RC patients.
The two manifests record exact image/reference hashes and repository-relative
paths (resolved from the manifest directory). No tumor outlines are edited.
Selection uses the existing best-validation checkpoints, never MC-RC scores.

All predictions return to each original processed T1 grid. Both pipelines use
the existing `src/evaluate.py` patient-level Dice scorer. No test-time augmentation,
ensembling or component filtering is used. Each pipeline retains its prescribed
preprocessing and sliding-window blending; this compares complete pipelines.
Native training-validation scores remain separate from these common scores.
The mean of the patient Dice scores is the primary result; report all three manual
seeds. One nnU-Net run cannot establish its training variability.

## Why the external cohort remains 22

Independent reconstruction from the original source spreadsheet and VS-SEG split
confirms 51 locally available patients with a directly mapped T1 reference mask.
Twenty-five are training patients and four are validation patients, leaving 22.
The previous identity/source audit and its input hashes were checked before freezing.
This cohort remains retrospective; correcting overlap does not undo prior inspection.

The wider local T1 inventory contains 27 additional candidate patients outside
VS-SEG. `additional_candidates.csv` records the latest local T1 scan and same-study
T2 reference for every candidate. All 27 source labels match their T2 grids within
0.1 mm (maximum 0.012985 mm), and their physical foreground coordinates fall inside
the corresponding T1 field of view. Twenty-five pairs share a DICOM frame of
reference; two do not (VS-MC-RC-040 and VS-MC-RC-086).

These checks establish identity and source geometry, not accurate transfer of a
tumor boundary between acquisitions. Three-plane diagnostic projections were
viewed for VS-MC-RC-002, -020 and -040: approximate correspondence is visible, but
the coarse T1 slices and differing T2 boundary appearance do not establish a
validated T1 reference for all 27 cases. No registration or manual correction was
performed. All 27 are deferred from this **direct-T1-reference comparison** under
the same rule, without inspecting model predictions or scores. This does not mean
they are permanently unusable. Expansion needs a separate, documented validation
of transferred references; 49 is a candidate count, not a verified test size.
Restricting evaluation to directly T1-labelled cases also limits representativeness.

The [TCIA release page](https://www.cancerimagingarchive.net/collection/vestibular-schwannoma-mc-rc/)
lists 160 patients in version 2, despite its 124-subject header. The local download
matches version 1: 124 patients and 354 series. The source spreadsheet already lists
all 487 series; the 133 absent locally comprise 126 T2-labelled series and seven
T1 series without a direct segmentation path. A larger download alone would not
increase this directly T1-labelled cohort. The
[dataset paper](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2024.1365727/full)
explains that paired-modality annotations use the higher-resolution image grid.

## Run and inspect

`experiments/prepare_evaluation.py` creates these frozen records on a fresh
workspace using the existing source audit and patient manifest. It refuses to
overwrite a frozen evaluation. MC-RC's original processed files omit spatial units;
the source DICOM audit establishes millimetres. Preparation therefore makes copies
under `data/processed/VS_MCRC_T1_evaluation_mm/` with explicit millimetre units.
Every copied image and mask is checked for exactly unchanged voxel values and
affine coordinates. `header_copies.csv` records original/copy hashes. Original
files remain unchanged; these are header corrections, not annotation corrections.

From the repository root:

```bash
sbatch --array=1-7 experiments/evaluate.slurm validation
# Inspect validation exports and common scores before submitting MC-RC:
sbatch --array=1-7 experiments/evaluate.slurm mcrc
```

Each task reuses the existing environments, exporter and scorer, verifies the
checkpoint hash, and refuses an existing output directory. Logs are
`logs/t1-eval-<array job>_<task>.log`. Masks, per-patient scores, manifest copies
and execution records are under `outputs/t1/evaluation_20260912/<cohort>/<run>/`.
Validation must finish and pass review before MC-RC submission.

Validation array: `2237083` (all seven tasks completed with exit code 0; 2:06–2:24
per task, excluding queue time). Tasks 1–7 follow `runs.csv`. Four existing CPU checks
passed for Dice/volumes, geometry, checkpoint settings and patient pairing.
All 46 frozen image/reference pairs passed hash and geometry checks, and all
44 MC-RC header copies preserved exact voxel values and affine coordinates.
MC-RC array: `2237116` (all seven tasks completed with exit code 0; 0:37–1:07 per
task, excluding queue time). Submitted after validation passed. No additional
training, test-time augmentation or annotation editing was performed.

## Next experiment

Review **training** labels, preserve the originals, and freeze justified corrections
as a separate version. Repeat the same configurations and seeds using that version.
Keep the validation/MC-RC manifests and reference voxel values fixed for a paired
original-versus-corrected-training-label comparison. Any evaluation-reference
revision or cohort expansion is a separately reported study, not an unnoticed
change to the benchmark above.
