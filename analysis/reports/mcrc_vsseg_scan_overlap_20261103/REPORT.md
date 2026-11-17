# Scan-level overlap between VS-MC-RC and VS-SEG — 3 November 2026

**The patient count stands: 29 of the 51 MC-RC patients with direct T1 references are shared with VS-SEG (25 training, 4 validation), leaving 22. The earlier count of duplicated scans does not: 11 selected acquisitions are the same scans as VS-SEG series, not 5.** The 22-patient cohort and all scores are unaffected, because all 29 patients were already excluded.

## Question and result

For each shared patient: is a local MC-RC series the same MRI acquisition as a VS-SEG series, or a different examination of the same person?

| Group | Patients (`VS-SEG-`) | n | Train / val |
|---|---|---:|---:|
| Selected MC-RC T1 is the same acquisition as VS-SEG | 045, 060, 095, 111, 112, 117, 126, 188, 197, 199, 221 | 11 | 10 / 1 |
| Selected T1 differs; another local MC-RC T1 is the same acquisition | 092, 135, 154, 178 | 4 | 4 / 0 |
| No local same-acquisition series | 003, 010, 013, 021, 028, 058, 061, 070, 077, 098, 104, 107, 116, 195 | 14 | 11 / 3 |

Per-patient decisions are in `patient_classification.csv`, per-series evidence in `same_acquisition_pairs.csv`, and counts with input hashes in `summary.json`.

## Evidence

Same-acquisition pairs share FrameOfReferenceUID and in-plane matrix. All 17 pairs found (15 T1 pairs in the cohort, plus T2 pairs for `VS-SEG-133` and `-172`, outside it) also have equal AcquisitionTime to the second. All have the same slice count except `VS-SEG-221` (118 in MC-RC, 120 in VS-SEG), and all of the MC-RC slices sit at matching positions.

- **Not byte-identical.** None of the 354 local MC-RC series shares a single slice's stored pixel bytes with any of the 484 VS-SEG T1/T2 series.
- **Same voxels in the central volume.** In the 15 T1 pairs, 98.6–100% of voxels in the central half of the volume are equal.
- **One-directional difference.** 18–27% of foreground voxels are zero in MC-RC and nonzero in VS-SEG; the reverse never occurs. This suggests masking of the MC-RC copy, possibly defacing. The method was not established. A small share of voxels nonzero in both also differ, and that was not explained.
- **Published list is incomplete.** All eight rows of `series_overlap.csv` are confirmed, but it omits nine pairs: `VS-SEG-045`, `-060`, `-092`, `-112`, `-126`, `-172`, `-178`, `-197` and `-221`. That file was the source of the earlier count of five in `trustworthy_testing_20260907/REPORT.md`.

## Limits

- **Dates cannot be compared across collections.** The same acquisition has dates 0–341 days apart. For the 14 patients without a match, the scans are different examinations, but their order and interval cannot be determined.
- **Identity of the 14 rests on documentation.** The 15 are confirmed by image data. The 14 rely on the shared patient ID and the published statement of 58 shared patients (Kujawa et al., 2024).
- **The 22 retained patients.** None shares a FrameOfReferenceUID with VS-SEG. This rules out a shared session among local series, not an undocumented shared person.
- **Series not available locally.** 18 published series for the 29 patients are absent from the local TCIA version 1 download, 11 of them for the 14 patients. Their descriptions do not resemble the VS-SEG protocols, but their pixels were not checked.
- **Superseded statement.** `trustworthy_testing_20260907/REPORT.md` is left unchanged as a historical record; its count of five is superseded by this report.

## Reproduce

From the repository root, on CPU, in about ten minutes:

```bash
myenv/bin/python analysis/audit_mcrc_vsseg_scan_overlap.py
```

The audit only reads the data. No training, inference, image or label change was performed.
