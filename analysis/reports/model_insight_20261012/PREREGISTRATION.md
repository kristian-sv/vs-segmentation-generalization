# Preregistration — slab + thick-slice augmentation for nnU-Net (approved 12 Oct 2026 with four edits; committed before any training code ran)

**Question.** Does training-time simulation of limited slab coverage and thick slices reduce nnU-Net failures on
MC-RC slab acquisitions without harming whole-head performance?

**Basis.** VS-SEG stress test (`FINDINGS.md`, validation data only): nnU-Net loses little from a 40 mm slab or 3–4 mm
slices alone, but the combination (S40T3) costs −0.39 Dice with 9/24 empty predictions. Slab position had no measurable
effect. The stress test therefore supports **one variant: slab and thick slices, applied both separately and together.**
The optional second variant is not used. MC-RC Dice was not used to choose the augmentation or its parameters.

**Augmentation (ranges from MC-RC acquisition metadata in `cases.csv`; 13 slab cases).**
- Slab: with p = 0.5, keep a contiguous slab of U(30, 55) mm (MC-RC slab coverage 32.8–52.8 mm) along the head-feet axis.
  With probability 0.15 of slab events, use the front-back axis instead (2 of 13 MC-RC slabs are coronal). Place the slab
  uniformly at random among positions that keep the whole tumour, so position is not a design factor. Voxels outside the
  slab are set to **0 in z-score-normalised intensity**, exactly the value nnU-Net's predictor pads with when the input is
  smaller than the patch: `pad_nd_image(input_image, patch_size, 'constant', {'value': 0}, True, None)` in
  `nnunetv2/inference/predict_from_raw_data.py:690–692` (nnU-Net 2.8.1, `predict_sliding_window_return_logits`).
  Dataset901 uses `ZScoreNormalization` with `use_mask_for_norm: [False]`, so 0 is the case's mean intensity, not air.
  The training dataloader pads with the same value (`crop_and_pad_nd(data, bbox, 0)`, `training/dataloading/data_loader.py`).
  The transform runs after the intensity augmentations and immediately before mirroring, so the fill stays exactly 0.
- Thick slices: independently with p = 0.5, area-average along the slab axis (head-feet when there is no slab) to
  U(1.0, 3.5) mm spacing (MC-RC slab slice spacing 0.91–3.48 mm), then interpolate linearly back to the training grid.
  This degrades the **image only**; the label is unchanged, as in nnU-Net's `SimulateLowResolutionTransform`. Draws
  at or below the axis's current spacing (1.5 mm head-feet) leave the image unchanged.
- Result: ~25 % none, ~25 % slab only, ~25 % thick only, ~25 % both. Implemented as an nnU-Net trainer subclass. Every
  other setting stays as in Dataset901: plans, 193/24 split, 1000 epochs, `checkpoint_best.pth`.

**Runs.** Plain nnU-Net seed 43 (new) plus the existing plain seed 42; augmented nnU-Net seeds 42 and 43. No manual-U-Net runs.

**Inference.** Every model, old and new, receives all MC-RC inputs re-stacked to the training orientation (permutation/flip
only, `model_insight.py reorient-stage`). Masks are mapped back to the original grid and scored with `src/evaluate.py`.
No TTA. **The frozen September nnU-Net results remain reported as the original pre-specified result; the comparator
for the new runs is plain nnU-Net with re-stacked inputs.** MC-RC is predicted and scored **once**, after all four runs are trained and validation has been reviewed.
Nothing changes after MC-RC is seen.

**Primary endpoint.** On the 13 MC-RC slab cases (11 axial slabs plus coronal 012 and 100): mean Dice and number of empty
predictions, augmented vs plain (mean of seeds 42 and 43 per arm). Reported as a paired patient bootstrap 95 % CI of the
Dice difference (10,000 replicates, seed 42). Augmentation is called beneficial only if the CI excludes 0 and the arm has
no more empty predictions than plain. **If the primary CI includes 0, the result is reported as inconclusive (estimate
and CI), not as evidence of no effect.**

**Secondary endpoints.** The 7 axial whole-head MC-RC cases; VS-SEG validation (pre-specified check: mean Dice not more than
0.01 below plain); the 11 stress-test conditions. Caveat: the stress test resembles the augmentation by construction.

**Reporting rule for rare acquisition types** (strata defined by acquisition properties only, never by scores):
- The overall result is always reported on **all 22** MC-RC cases, with no exclusions.
- Per stratum: axial whole-head (021, 028, 044, 063, 082, 085, 099), axial slab (009, 016, 024, 026, 030, 045, 057, 076,
  081, 097, 098), non-axial (coronal 012, 100; sagittal 043, 060).
- Pre-specified sensitivity analysis: axial acquisitions only (18 cases, the intended use), reported next to the all-cases result.
- Every table shows mean, median and the per-case values.

**Not claimed.** Generalisation beyond this retrospective cohort; site, sequence-contrast or tumour-size effects; the
cause of VS-MC-RC-012's drop. Two seeds per arm cannot establish training variability.
