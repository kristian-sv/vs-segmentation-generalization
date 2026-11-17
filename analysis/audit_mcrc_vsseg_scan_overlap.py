#!/usr/bin/env python3
"""Scan-level overlap audit between the local VS-MC-RC and VS-SEG DICOM data.

Patient overlap is taken from the existing cohort decisions. This script asks the
next question: for each shared patient, is a local MC-RC series the same MRI
acquisition as a VS-SEG series, or a different examination of the same person?

Two series are called the same acquisition when they share FrameOfReferenceUID and
in-plane matrix. Acquisition time and voxel equality are reported as evidence.
Study dates are reported but not used: they differ between the collections even for
the same acquisition.

Read-only with respect to the data. Run from the repository root:
    myenv/bin/python analysis/audit_mcrc_vsseg_scan_overlap.py
"""

import argparse
import collections
import csv
import datetime
import glob
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pydicom

RAW = "data/raw/datasets"
SHEET = f"{RAW}/VS-MC-RC-Segmentations_with_pathname_spreadsheet-Release2023/Vestibular-Schwannoma-MC-RC_NIfTI-segmentation-info3.csv"
MAPPING = f"{RAW}/DirectoryNamesMappingModality.csv"
SERIES_OVERLAP = f"{RAW}/series_overlap.csv"
DECISIONS = "analysis/reports/trustworthy_testing_20260907/cohort_decisions.csv"
SPLIT = "data/splits/vsseg_baseline.csv"
MASTER = "metadata/vs_mcrc_master.csv"
DEFAULT_OUTPUT = "analysis/reports/mcrc_vsseg_scan_overlap_20261103"


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def scan_series(job):
    """Tags of one series and the MD5 of each slice's stored pixel bytes."""
    collection, series = job
    out = {"collection": collection, "series": series, "slices": []}
    for path in sorted(glob.glob(f"{RAW}/{collection}/{series}/*.dcm")):
        ds = pydicom.dcmread(path)
        if "PixelData" not in ds:
            continue
        out["slices"].append(hashlib.md5(ds.PixelData).hexdigest())
        if "patient" not in out:
            out.update(
                patient=str(ds.get("PatientID", "")),
                study_date=str(ds.get("StudyDate", "")),
                acquisition_time=str(ds.get("AcquisitionTime", "")),
                frame_of_reference=str(ds.get("FrameOfReferenceUID", "")),
                description=str(ds.get("SeriesDescription", "")),
                model=str(ds.get("ManufacturerModelName", "")),
                rows=int(ds.get("Rows", 0)),
                columns=int(ds.get("Columns", 0)),
            )
    return out


def load_slices(collection, series):
    slices = {}
    for path in glob.glob(f"{RAW}/{collection}/{series}/*.dcm"):
        ds = pydicom.dcmread(path)
        if "PixelData" in ds:
            position = tuple(round(float(x), 2) for x in ds.ImagePositionPatient)
            slices[position] = ds.pixel_array.astype(np.int32)
    return slices


def compare_voxels(job):
    """Voxel agreement at shared slice positions, overall and in the central half box."""
    mcrc_series, vsseg_series = job
    a = load_slices("VS-MC-RC", mcrc_series)
    b = load_slices("VS-SEG", vsseg_series)
    common = sorted(set(a) & set(b), key=lambda p: p[2])
    if not common:
        return {"common_slices": 0}
    x = np.stack([a[p] for p in common])
    y = np.stack([b[p] for p in common])
    nz, ny, nx = x.shape
    box = (slice(nz // 4, 3 * nz // 4), slice(ny // 4, 3 * ny // 4), slice(nx // 4, 3 * nx // 4))
    foreground = (x > 0) | (y > 0)
    both = (x > 0) & (y > 0)
    return {
        "common_slices": len(common),
        "central_voxels_equal": float((x[box] == y[box]).mean()),
        "central_correlation": float(np.corrcoef(x[box].ravel(), y[box].ravel())[0, 1]),
        "both_nonzero_equal": float((x[both] == y[both]).mean()),
        "zero_only_in_mcrc": float(((x == 0) & (y > 0)).sum() / foreground.sum()),
        "zero_only_in_vsseg": float(((y == 0) & (x > 0)).sum() / foreground.sum()),
    }


def parse_date(value):
    return datetime.datetime.strptime(value, "%Y%m%d").date()


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    decisions = {row["patient_id"]: row for row in read_csv(DECISIONS)}
    split = {row["case_id"]: row["split"] for row in read_csv(SPLIT)}
    master = {row["patient_id"] for row in read_csv(MASTER)}
    sheet = read_csv(SHEET)
    mapping = read_csv(MAPPING)
    listed = {row["VS-MC-RC Series Instance UID"]: row["VS-SEG Series Instance UID"] for row in read_csv(SERIES_OVERLAP)}
    selected = {row["selected_mcrc_series_uid"] for row in decisions.values()}
    shared_cohort = sorted(p for p in master if p in split)

    vsseg_modality = {
        row["Classic Directory Name"].split("/")[2]: row["Modality"]
        for row in mapping
        if row["Modality"] in ("T1 image", "T2 image")
    }
    jobs = [("VS-SEG", series) for series in vsseg_modality]
    jobs += [("VS-MC-RC", d) for d in sorted(os.listdir(f"{RAW}/VS-MC-RC")) if d.startswith("1.")]
    with ProcessPoolExecutor(args.workers) as pool:
        series = [s for s in pool.map(scan_series, jobs, chunksize=4) if s["slices"]]
    vsseg = [s for s in series if s["collection"] == "VS-SEG"]
    mcrc = [s for s in series if s["collection"] == "VS-MC-RC"]

    vsseg_hashes = {h for s in vsseg for h in s["slices"]}
    shared_slices = sum(h in vsseg_hashes for s in mcrc for h in s["slices"])

    by_frame = collections.defaultdict(list)
    for s in vsseg:
        by_frame[s["frame_of_reference"]].append(s)
    frame_pairs = [(m, v) for m in mcrc for v in by_frame.get(m["frame_of_reference"], [])]
    id_mismatches = sorted({(m["patient"], v["patient"]) for m, v in frame_pairs if m["patient"] != v["patient"]})
    pairs = [(m, v) for m, v in frame_pairs if (m["rows"], m["columns"]) == (v["rows"], v["columns"])]

    with ProcessPoolExecutor(args.workers) as pool:
        voxels = list(pool.map(compare_voxels, [(m["series"], v["series"]) for m, v in pairs]))

    pair_rows = []
    for (m, v), stats in sorted(zip(pairs, voxels), key=lambda t: t[0][0]["patient"]):
        pair_rows.append(
            {
                "patient_id": m["patient"],
                "vsseg_split": split.get(m["patient"], ""),
                "in_51_t1_cohort": m["patient"] in master,
                "selected_for_t1_cohort": m["series"] in selected,
                "in_series_overlap_csv": listed.get(m["series"]) == v["series"],
                "sequence": vsseg_modality[v["series"]].split()[0],
                "mcrc_series_uid": m["series"],
                "vsseg_series_uid": v["series"],
                "mcrc_description": m["description"],
                "vsseg_description": v["description"],
                "mcrc_study_date": m["study_date"],
                "vsseg_study_date": v["study_date"],
                "date_offset_days": (parse_date(v["study_date"]) - parse_date(m["study_date"])).days,
                "acquisition_time_equal": m["acquisition_time"] == v["acquisition_time"],
                "mcrc_slices": len(m["slices"]),
                "vsseg_slices": len(v["slices"]),
                **stats,
            }
        )
    write_csv(f"{args.output_dir}/same_acquisition_pairs.csv", pair_rows)

    t1_pairs = [r for r in pair_rows if r["sequence"] == "T1" and r["in_51_t1_cohort"]]
    selected_same = {r["patient_id"] for r in t1_pairs if r["selected_for_t1_cohort"]}
    other_same = {r["patient_id"] for r in t1_pairs} - selected_same
    frames_by_patient = {m["patient"] for m, _ in frame_pairs}
    patient_rows = []
    for patient in shared_cohort:
        if patient in selected_same:
            group = "selected_scan_same_acquisition"
        elif patient in other_same:
            group = "other_local_scan_same_acquisition"
        else:
            group = "no_local_same_acquisition"
        missing = [
            row["series_description"]
            for row in sheet
            if row["patient_id"] == patient and row["series_instance_uid"] not in {s["series"] for s in mcrc}
        ]
        patient_rows.append(
            {
                "patient_id": patient,
                "vsseg_split": split[patient],
                "group": group,
                "selected_mcrc_series_uid": decisions[patient]["selected_mcrc_series_uid"],
                "selected_examination": decisions[patient]["source_segmentation_path"].split("/")[1],
                "local_mcrc_series": sum(s["patient"] == patient for s in mcrc),
                "published_series_not_local": len(missing),
            }
        )
    write_csv(f"{args.output_dir}/patient_classification.csv", patient_rows)

    groups = collections.defaultdict(collections.Counter)
    for row in patient_rows:
        groups[row["group"]][row["vsseg_split"]] += 1
    summary = {
        "cohort_patients": len(master),
        "shared_by_split": dict(collections.Counter(split[p] for p in shared_cohort)),
        "retained_patients": len(master) - len(shared_cohort),
        "published_vsseg_prefixed_patients": len({r["patient_id"] for r in sheet if r["patient_id"].startswith("VS-SEG")}),
        "local_mcrc_series": len(mcrc),
        "vsseg_image_series": len(vsseg),
        "byte_identical_mcrc_slices_in_vsseg": shared_slices,
        "same_acquisition_pairs": len(pair_rows),
        "same_acquisition_t1_pairs_in_cohort": len(t1_pairs),
        "patient_groups": {k: {"total": sum(v.values()), **v} for k, v in groups.items()},
        "selected_same_acquisition": sorted(selected_same),
        "other_scan_same_acquisition": sorted(other_same),
        "series_overlap_csv_rows": len(listed),
        "series_overlap_csv_rows_confirmed": sum(r["in_series_overlap_csv"] for r in pair_rows),
        "frame_of_reference_id_mismatches": id_mismatches,
        "retained_patients_sharing_frame_of_reference": sorted(p for p in frames_by_patient if p in master and p not in split),
        "patients_outside_cohort_with_same_acquisition": sorted({r["patient_id"] for r in pair_rows if not r["in_51_t1_cohort"]}),
        "date_offset_days_range": [min(r["date_offset_days"] for r in pair_rows), max(r["date_offset_days"] for r in pair_rows)],
        "central_voxels_equal_range_t1": [min(r["central_voxels_equal"] for r in t1_pairs), max(r["central_voxels_equal"] for r in t1_pairs)],
        "zero_only_in_mcrc_range_t1": [min(r["zero_only_in_mcrc"] for r in t1_pairs), max(r["zero_only_in_mcrc"] for r in t1_pairs)],
        "zero_only_in_vsseg_max": max(r["zero_only_in_vsseg"] for r in pair_rows),
        "input_sha256": {p: sha256(p) for p in (SHEET, MAPPING, SERIES_OVERLAP, DECISIONS, SPLIT, MASTER)},
    }
    with open(f"{args.output_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps({k: v for k, v in summary.items() if k != "input_sha256"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
