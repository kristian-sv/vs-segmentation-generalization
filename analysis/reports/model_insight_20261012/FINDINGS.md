# Why MC-RC cases fail — 12 October 2026

Inputs: frozen seven-run evaluation (`t1_evaluation_20260912`), 24 validation + 22 MC-RC patients.
Tables: `cases.csv` (322 rows; Dice/volumes equal `patient_scores.csv` exactly), `orientation.csv`,
`acquisition_by_outcome.csv`, `nnunet_reoriented_scores.csv`, `paired_bootstrap.json`.
Success = Dice > 0.7 (manual: mean of six runs). "Coverage" = `n_slices × slice_spacing_mm`.

## 1. Orientation check (observations)
- All 217 VS-SEG development images are stored LPS, axial, 0° tilt, 180 mm slice coverage, 3D.
- Manual pipeline: `Orientationd('RAS')` gives every case RAS axes (verified on the transformed tensors);
  tilt is kept (`manual_network_max_tilt_deg` = stored grid tilt, up to 32° for 057).
- nnU-Net (installed `SimpleITKIO.read_images`, printed in job 2501551): no reorientation; plans transpose is identity.
  **012, 100 (coronal) and 043, 060 (sagittal) reached nnU-Net with permuted axes.**
- Axis-order-only correction (no resampling), same checkpoint, no TTA: the 18 axial cases gave voxel-identical masks.
  043 0.00→0.77, 100 0.00→0.89, 060 0.01→0.23, **012 0.60→0.04**. 22-case nnU-Net Dice:
  mean **0.483 → 0.543**, median **0.679 → 0.788** (without → with correction).
- VS-MC-RC-044 is a Siemens Avanto 1.5 T axial 3D MPRAGE (`t1_mpr_tra_p2_iso_postGd`, `*tfl3d1`, TI 1100 ms),
  1 × 1 × 1 mm, 192 × 256 in-plane, 176 axial partitions, ~4° tilt. It has no thick axis; "axis 0 thickest"
  was an `argmax` tie in my first header check. All seven models succeed on it (0.84–0.87).

## 2. Success vs failure, MC-RC (median, range)
| | Manual ok (9) | Manual fail (13) | nnU-Net ok (11) | nnU-Net fail (11) |
|---|---|---|---|---|
| Coverage, mm | 180 (49.5–200) | 44 (32.8–176) | 180 (36.3–200) | 44 (32.8–176) |
| Slice thickness, mm | 2 (1–3) | 1 (0.9–3) | 2 (1–3) | 1 (0.9–3) |
| In-plane spacing, mm | 0.83 (0.37–1) | 0.44 (0.31–0.98) | 0.63 (0.37–1) | 0.44 (0.31–0.98) |
| Obliquity, ° | 0 (0–6.4) | 5.1 (0–32) | 0 (0–32) | 3.9 (0–14) |
| Tumour–FOV edge, mm | 46 (19–61) | 9.6 (4.1–64) | 40 (4.7–61) | 14 (4.1–64) |
| Reference volume, ml | 2.2 (0.20–7.8) | 1.1 (0.03–27) | 2.2 (0.39–27) | 0.86 (0.03–15) |
| 2D / non-axial / 3 T | 2 / 0 / 0 | 6 / 4 / 4 | 3 / 0 / 2 | 5 / 4 / 2 |

- Coverage splits the cohort: 13 slabs (32.8–52.8 mm) and 9 whole-head (158–200 mm). Mean Dice slab/whole:
  manual 0.20/0.76, nnU-Net 0.34/0.70 (0.36/0.81 reoriented). Successes on slabs: manual 2/13, nnU-Net 4/13.
- Failure type differs: nnU-Net fails empty (all 7 empties are slabs with reference ≤ 1.5 ml); manual fails
  non-empty (`under` / `wrong_location`). MC-RC median HD95: manual ≈ 20 mm, nnU-Net 4.8 mm.
- Not discriminating: field strength, contrast (18 DICOM tag, 1 series description, 3 none found but all succeed),
  `saturated_frac` (≤ 1.2e-6 everywhere).
- Paired bootstrap, MC-RC: nnU-Net − manual mean +0.058 [−0.095, 0.215]; dropout − baseline s42/43/44
  +0.024 [−0.009, 0.064], −0.072 [−0.161, 0.005], +0.009 [−0.067, 0.083]. All intervals include 0.

## 3. Notable cases
- **057** (32° oblique, 3.3 mm slab): nnU-Net 0.86; 5/6 manual runs predict blobs 42–102 mm away.
- **026**: all six manual runs `wrong_location` (centroid 57–123 mm away); nnU-Net empty.
- **060** (sagittal 3D, 12.6 ml): all seven models capture 1–9 % of the reference at precision ≈ 1.0.
- **012** (coronal 2D SE, 3.3 mm): the only case made worse by axis correction.
- **016**: reference 0.03 ml, below the validation minimum (0.14 ml); every model fails.

## 4. Hypotheses (not tested)
- Slab coverage, not scanner or field strength, drives most failures: training never had < 180 mm coverage, and
  slabs also put the tumour near the FOV edge and change the image context.
- 012 got worse because the 3.3 mm axis moved from nnU-Net's coarse first axis to an in-plane axis, where it is
  upsampled 8× by nearest neighbour (anisotropy > 3).
- 060's consistent sub-segmentation may reflect the reference (e.g. a cystic part), not the models.

## 5. Open questions
- Is coverage independent of 2D acquisition, site and tumour size? n = 22 cannot separate them.
- Should the reoriented nnU-Net result replace the frozen one in the thesis? That changes the protocol after inspection.
- Visual review of 060 and 026 against the reference.

# Stress test on VS-SEG validation — 12 October 2026

24 validation patients, 11 acquisition perturbations fixed before inference (`stress_test/perturbations.csv`,
commit fbfee1f), 7 frozen models, September inference settings, scored on each perturbed grid
(`stress_test/stress_test.csv`, 1,896 rows; `stress_summary.csv`, `stress_bootstrap.json`, `stress_test.pdf`).
Δ = mean Dice minus C0 [paired patient bootstrap 95 % CI, 10,000 replicates, seed 42]; manual = per-patient mean of six runs.
**C0 sanity check: not exact.** 166/168 masks are bit-identical to the frozen exports. Two masks differ by one voxel each
(dropout01_s43 VS-SEG-169, nnU-Net VS-SEG-122; ΔDice ≤ 1.4e-4). The code, checkpoints and inputs are unchanged; only the
GPU nodes and CPU count differ. All Δ use the new C0 from the same run.

| Condition | Manual Δ | Manual empty | nnU-Net Δ | nnU-Net empty |
|---|---|---|---|---|
| S50 (49.5 mm) | −0.03 [−0.07, −0.01] | 2/144 | −0.02 [−0.05, +0.01] | 0/24 |
| S40 (40.5 mm) | −0.21 [−0.30, −0.13] | 7/144 | −0.07 [−0.17, +0.00] | 1/24 |
| S33 (33 mm) | −0.42 [−0.54, −0.30] | 10/144 | −0.17 [−0.32, −0.04] | 2/24 |
| S40edge (5.25 mm) | −0.20 [−0.29, −0.13] | 6/144 | −0.08 [−0.19, +0.01] | 1/24 |
| T3 / T4 | −0.05 / −0.05 | 0/144 | −0.08 / −0.07 | 0/24 |
| S40T3 | −0.38 [−0.50, −0.26] | 21/144 | −0.39 [−0.55, −0.22] | 9/24 |
| AP (52.9 mm front-back) | −0.74 [−0.79, −0.67] | 8/144 | −0.00 [−0.01, +0.00] | 0/24 |
| COR_thin as stored / re-stacked | −0.72 | 8/144 | −0.79 / −0.09 | 17/24 / 0/24 |
| COR_thick as stored / re-stacked | −0.84 | 13/144 | −0.36 / −0.08 | 7/24 / 0/24 |

1. **Slab coverage alone:** yes. The manual models drop slightly at 49.5 mm and clearly from 40.5 mm. nnU-Net drops only
   clearly at 33 mm (−0.17, 2 empty); at 40.5 mm its CI touches 0.
2. **Position in the slab:** no measurable effect. S40edge − S40: manual +0.007 [−0.031, +0.048], nnU-Net −0.010 [−0.037, +0.012].
3. **Thickness alone:** modest drop (−0.05 to −0.08, CIs exclude 0), no empty predictions. Part of the drop may be due to the
   reference being regridded too (T3 reference volume +23 % median at the 0.5 threshold).
4. **Combination:** worse than either alone. For nnU-Net, S40T3 − S40 is −0.32 [−0.48, −0.17] and S40T3 − T3 is −0.31
   [−0.47, −0.15]: larger than the sum of the single effects (−0.07 and −0.08), and 9/24 predictions are empty.
5. **Different failure modes:** yes, matching MC-RC. nnU-Net fails by predicting nothing: never `wrong_location`, and its
   S40T3 / COR_thick empties are 8/9 and 7/7 below-median tumours. The manual models under-segment on head-feet slabs and
   scatter far-away foreground on front-back-limited images (median HD95 ≈ 112–115 mm).
6. **Coronal:** limited front-back coverage alone (AP) leaves nnU-Net unchanged but destroys the manual models (0.18).
   Re-stacking helps COR_thin strongly (+0.70 [+0.58, +0.82], 24/24 patients), reproducing VS-MC-RC-100. It **also helps
   COR_thick** (+0.28 [+0.12, +0.44], 15 better / 9 worse), so 012's drop is **not** reproduced. My 8× nearest-neighbour
   hypothesis is not supported by this simulation.

**Hypotheses (not tested).** The manual pipeline pads every image to its 192 × 192 × 64 patch with `mode='edge'`
(`src/dataset.py`). A 53 mm front-back crop is therefore filled with ~140 mm of repeated edge slices, and head-feet slabs
with 14–31 slices (S50 to S33). That could explain the AP collapse and the manual models' greater slab sensitivity. I have not checked
nnU-Net's padding mode. 012's drop may depend on properties this simulation lacks.

**Limitations.** Only geometry is simulated. Sequence contrast (2D spin echo vs 3D gradient echo), noise, slice gaps and
slice profile are not. Coronal versions built from VS-SEG keep 1.5 mm head-feet spacing, coarser than a real coronal
in-plane resolution (0.44–0.56 mm). Twenty-four patients, one nnU-Net seed.
