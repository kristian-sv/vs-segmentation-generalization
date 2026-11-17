# Training readiness — 8 September 2026

**Ready to start the planned manual T1 and nnU-Net training.** Both pipelines
passed real-data GPU checks, all 217 development masks match the selected source
annotations, nnU-Net preparation is complete, and common scoring/export checks pass.
As of this report, no full experiment or held-out model evaluation has been run.

## What this means

The next study is deliberately small: baseline versus lower dropout, each trained
with seeds 42, 43 and 44, plus one nnU-Net pipeline. A seed changes the starting
randomness; three repetitions help distinguish an improvement from a lucky run.
Both pipelines use the same 193 training and 24 validation patients. Validation
is the feedback used during development. Test results must stay outside that loop.

The existing 22-patient MC-RC cohort remains useful retrospective evidence. Its
results have already been inspected, so removing overlapping patients does not
make it an untouched test. Fresh MC-RC2 access is a separate, optional route.

## Completed computing checks

| Check | Evidence | Outcome |
|---|---|---|
| Existing software and GPU | Job 2178655 | Manual forward/backward and compiled nnU-Net CUDA checks passed |
| Manual real-data path | Job 2185696 | Three updates on training cases 001/003; full validation volume 002; finite outputs |
| nnU-Net preparation | Job 2185696 | 193 training-only planning patients; 217 preprocessed image/mask/property triples; fixed 193/24 fold |
| nnU-Net real trainer | Job 2185700 | Three updates and one validation batch; compiled network; correct fixed split |

The selected node reported an NVIDIA GH200 120GB device name with 95 GiB
accessible GPU memory. Measured peak tensor allocations were 2.10 GiB for the
manual diagnostic and 5.01 GiB for nnU-Net. These are short tests, not worst-case
memory guarantees or full-run benchmarks. The nnU-Net import metadata omitted
NumPy's version; the successful trainer run directly reported NumPy 1.26.4.

nnU-Net 2.8.1 planned a 48×224×192 patch, batch size 2, and spacing
1.5×0.41015625×0.41015625 mm in its own axis order. Only `3d_fullres` was
preprocessed. The default planner also describes other configurations; they are
not additional experiments. Fingerprint, plans and split hashes were verified
after preprocessing. Copies of these small provenance files accompany this report.

The diagnostic checkpoints were never saved as trained study models. nnU-Net
diagnostic logs use a separate directory from future experiment outputs.

## Data and scoring verification

- VS-SEG source audit: **passed, 217/217 exact native T1 RTSTRUCT mask matches**.
  Seven cases also passed source-image equality and visual alignment review.
  Initial exceptions concerned an additional JSON representation and ROI selection;
  no label or data-lock changes were needed. See `../vsseg_t1_readiness_mvp/REPORT.md`.
- Common original-grid scoring: **passed** in job 2186086: five synthetic test
  groups, all 217 development geometries and hashes, and the real manual-export
  command on validation case VS-SEG-002 with freshly initialized diagnostic weights.
  No performance score was computed for that case. An initial test exposed missing
  MONAI transform history; the corrected MetaTensor input restored orientation,
  spacing, crop and padding before assigning classes. This is why export was tested
  before interpreting future model scores. The scorer also refuses mismatched
  checkpoint configurations, patient coverage, input hashes and image coordinates.
- MC-RC2: public package authorization/listing works, but the image transfer client
  and actual download remain pending. The separate metadata/pilot-QC adapter passed
  four tests; its candidate list is **not a verified fresh test cohort**. This
  optional route does not block development training. See
  `../mcrc2_preparation_mvp/REPORT.md` for download and quality-control requirements.

## Next action: start the small experiment

Use the commands in [the experiment guide](../../../experiments/README.md). Start the seed-42
baseline and lower-dropout runs, plus the single nnU-Net run. Confirm that the
first full runs write usable checkpoints and validation logs, then run the manual
pair with seeds 43 and 44. No new tuning or extra model families are required.
The batch script requests one GPU per run and saves records locally.

Do not judge improvement from the three-step diagnostics. They verify that the
machinery works. The actual experiments will measure performance. Select using
validation only, report all seeds, and keep the original 22-case MC-RC evaluation
explicitly retrospective. nnU-Net changes the whole pipeline, so a better score
cannot be attributed to architecture alone. Freeze final evaluation details before
opening any new held-out model results.

This is the five-week core. Extra submission time can improve external testing,
analysis and writing without invalidating these runs. Optional fresh-data access
or more model families must not delay the core study.

## Study status

The six manual runs and one nnU-Net run remain future work as of this report.
