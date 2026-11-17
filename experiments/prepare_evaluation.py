#!/usr/bin/env python3
"""Freeze this study's validation/MC-RC inputs and seven original-label checkpoints."""
import csv
import json
import os
from pathlib import Path
import sys

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from config import load_config
from data import check_lock, digest
from evaluate import FIELDS, pair_geometry, read_manifest, validate_checkpoint_config, write_csv

REPORT = ROOT / 'analysis/reports/t1_evaluation_20260912'
DATA = ROOT / 'data/processed'


def read_csv(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def main():
    targets = [REPORT / name for name in ('validation_manifest.csv', 'mcrc_manifest.csv', 'runs.csv')]
    if any(p.exists() for p in targets):
        raise FileExistsError('Frozen evaluation manifests already exist; do not replace them')
    lock = check_lock(full=True)
    audit = ROOT / 'analysis/reports/trustworthy_testing_20260907'
    for source, expected in json.loads((audit / 'provenance.json').read_text())['input_sha256'].items():
        if digest(ROOT / source) != expected:
            raise ValueError('Historical cohort input changed: ' + source)
    mcrc = read_csv(ROOT / 'metadata/evaluation/mcrc_t1_patient_independent.csv')
    blocked = {r['case_id'] for r in lock['patients']}
    if len(mcrc) != 22 or blocked.intersection(r['canonical_patient_id'] for r in mcrc):
        raise ValueError('MC-RC cohort differs from the reviewed patient-independent cohort')
    geometry = {r['patient_id']: r for r in read_csv(audit / 'source_geometry.csv')}
    pairs = {'validation': [(r['case_id'], DATA / r['image'], DATA / r['label'])
                            for r in lock['patients'] if r['split'] == 'val'], 'mcrc': []}
    headers = []
    for row in mcrc:
        pid = row['patient_id']
        evidence = geometry[pid]
        if evidence['geometry_status'] != 'source_grid_matches' or float(evidence['processed_image_to_dicom_corner_error_mm']) > 0.1:
            raise ValueError('Source millimetre geometry is unverified: ' + pid)
        copies = []
        for field in ('t1_path', 'seg_path'):
            source = DATA / row[field]
            image = nib.load(source)
            units, time_units = image.header.get_xyzt_units()
            if units not in ('unknown', 'mm'):
                raise ValueError('Unexpected spatial units: ' + str(source))
            destination = DATA / 'VS_MCRC_T1_evaluation_mm' / pid / source.name
            if destination.exists():
                raise FileExistsError(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            values = np.asanyarray(image.dataobj)
            header = image.header.copy()
            header.set_xyzt_units('mm', time_units)
            nib.save(nib.Nifti1Image(values, image.affine, header), destination)
            saved = nib.load(destination)
            if not np.array_equal(values, np.asanyarray(saved.dataobj)) or not np.array_equal(image.affine, saved.affine):
                raise ValueError('Header copy changed values or coordinates: ' + pid)
            headers.append(dict(patient_id=pid, source=str(source.relative_to(ROOT)),
                                source_sha256=digest(source), copy=str(destination.relative_to(ROOT)),
                                copy_sha256=digest(destination), original_units=units,
                                voxels_and_affine_unchanged=True))
            copies.append(destination)
        pairs['mcrc'].append((pid, *copies))
    REPORT.mkdir(parents=True, exist_ok=True)
    write_csv(REPORT / 'header_copies.csv', headers)
    for cohort, cohort_pairs in pairs.items():
        rows = [dict(case_id=pid, image=os.path.relpath(image, REPORT),
                     reference=os.path.relpath(label, REPORT), image_sha256=digest(image),
                     reference_sha256=digest(label)) for pid, image, label in cohort_pairs]
        path = REPORT / (cohort + '_manifest.csv')
        write_csv(path, rows, FIELDS)
        for row in read_manifest(path):
            pair_geometry(row)
        print(f'Frozen {cohort}: {len(rows)} verified pairs', flush=True)
    runs = []
    for family in ('baseline', 'dropout01'):
        for seed in (42, 43, 44):
            paths = list((ROOT / 'outputs/t1/manual').glob(f't1_mvp_{family}_s{seed}_20260908_*/best_model.pt'))
            if len(paths) != 1:
                raise ValueError(f'Missing or ambiguous checkpoint: {family} {seed}')
            checkpoint = paths[0]
            validate_checkpoint_config(checkpoint, load_config(family, seed))
            runs.append(dict(run=f'{family}_s{seed}', family=family, seed=seed,
                             checkpoint=str(checkpoint.relative_to(ROOT)), checkpoint_sha256=digest(checkpoint)))
    checkpoint = ROOT / 'outputs/t1/nnunet/Dataset901_T1VSComparison/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth'
    runs.append(dict(run='nnunet_s42', family='nnunet', seed=42,
                     checkpoint=str(checkpoint.relative_to(ROOT)), checkpoint_sha256=digest(checkpoint)))
    write_csv(REPORT / 'runs.csv', runs)
    print('Frozen seven best-validation checkpoints')


if __name__ == '__main__':
    main()
