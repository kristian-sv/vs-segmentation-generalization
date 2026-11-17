#!/usr/bin/env python3
"""Pre-protocol checks of the 24 reserved internal-test patients. Reads image HEADERS and file bytes only;
reference masks are hashed as bytes and never opened. Writes test_manifest.csv and preflight.json (never overwrites).

    sbatch analysis/reports/internal_test_20261021/code/internal_test.slurm preflight
"""
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import nibabel as nib

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from data import check_lock, digest  # noqa: E402
from evaluate import FIELDS, read_manifest  # noqa: E402

REVIEW = ROOT / 'outputs/t1/review/vsseg242_nnunet_cv/review_manifest.csv'  # 22 Sep 2026 image hashes (c536fae)


def main():
    lock = check_lock()  # split file, master and frozen patient list unchanged (test bytes are not in the lock)
    test = [p for p in lock['patients'] if p['split'] == 'test']
    with REVIEW.open(newline='') as f:
        review = {r['case_id']: r for r in csv.DictReader(f)}
    rows, records = [], []
    for p in sorted(test, key=lambda r: r['case_id']):
        image, label = ROOT / 'data/processed' / p['image'], ROOT / 'data/processed' / p['label']
        header = nib.load(image)
        image_sha, label_sha = digest(image), digest(label)
        records.append(dict(
            case_id=p['case_id'], axcodes=''.join(nib.aff2axcodes(header.affine)), shape=list(header.shape),
            zooms_mm=[round(float(z), 4) for z in header.header.get_zooms()[:3]],
            spatial_unit=header.header.get_xyzt_units()[0], image_sha256=image_sha, label_sha256=label_sha,
            image_mtime=datetime.fromtimestamp(image.stat().st_mtime).isoformat(timespec='seconds'),
            label_mtime=datetime.fromtimestamp(label.stat().st_mtime).isoformat(timespec='seconds'),
            review_original_split=review[p['case_id']]['original_split'],
            image_sha256_matches_22sep_review_record=review[p['case_id']]['image_sha256'] == image_sha))
        rows.append(dict(case_id=p['case_id'], image=str(Path('../../../data/processed') / p['image']),
                         reference=str(Path('../../../data/processed') / p['label']),
                         image_sha256=image_sha, reference_sha256=label_sha))
    with (HERE / 'test_manifest.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    read_manifest(HERE / 'test_manifest.csv')  # re-verify paths and hashes with the common reader
    summary = dict(n=len(records), all_lps=all(r['axcodes'] == 'LPS' for r in records),
                   all_mm=all(r['spatial_unit'] == 'mm' for r in records),
                   all_image_hashes_match_22sep_review=all(r['image_sha256_matches_22sep_review_record'] for r in records),
                   all_review_split_test=all(r['review_original_split'] == 'test' for r in records),
                   reference_masks_opened=False, patients=records)
    with (HERE / 'preflight.json').open('x') as f:
        json.dump(summary, f, indent=2)
        f.write('\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'patients'}, indent=2))


if __name__ == '__main__':
    main()
