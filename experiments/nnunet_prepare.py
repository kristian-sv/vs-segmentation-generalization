#!/usr/bin/env python3
"""T1-only nnU-Net staging/planning. Prints steps unless --execute is supplied."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from config import EXPERIMENTS as HERE, ROOT, DATA, WORK
from data import check_lock, digest

NAME = 'Dataset901_T1VSComparison'


def dataset_name(cv_fold=None):
    if cv_fold is None:
        return NAME
    if cv_fold not in range(5):
        raise ValueError('Expected a cross-validation fold from 0 to 4')
    # Separate datasets let each outer fold plan on its training patients only.
    return f'Dataset{902 + cv_fold}_T1VSCVFold{cv_fold}'


def configure_paths(cv_fold=None):
    for key in ('nnUNet_raw', 'nnUNet_preprocessed'):
        os.environ[key] = str(WORK / key)
    os.environ['nnUNet_results'] = str(ROOT / 'outputs/t1' / (
        'nnunet' if cv_fold is None else 'nnunet_cv'))


def expected_split(lock, cv_fold=None):
    if cv_fold is not None:
        dataset_name(cv_fold)
        ids = [r['case_id'] for r in lock['patients'] if r['split'] in ('train', 'val')]
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate development patient')
        # Stable seed-42 ordering, independent of input order and Python RNG version.
        ids.sort(key=lambda pid: hashlib.sha256(f'42:{pid}'.encode()).hexdigest())
        held_out = set(ids[cv_fold::5])
        return [{'train': sorted(set(ids) - held_out), 'val': sorted(held_out)}]
    return [{s: [r['case_id'] for r in lock['patients'] if r['split'] == s]
             for s in ('train', 'val')}]


def stage(rows, raw):
    import nibabel as nib
    import numpy as np
    for row in rows:
        image, label = DATA / row['image'], DATA / row['label']
        # Fail before admitting any new label version or geometry mismatch.
        img, lab = nib.load(image), nib.load(label)
        if img.shape != lab.shape or not np.allclose(img.affine, lab.affine, atol=1e-4, rtol=0):
            raise ValueError(f'Image/label grid mismatch: {row["case_id"]}')
        if not set(np.unique(np.asanyarray(lab.dataobj))).issubset({0, 1}):
            raise ValueError(f'Expected existing binary VS mask: {row["case_id"]}')
        for src, folder, suffix in [(image, 'imagesTr', '_0000'), (label, 'labelsTr', '')]:
            shutil.copyfile(src, raw / folder / (row['case_id'] + suffix + '.nii.gz'))


def check_preparation(lock, cv_fold=None):
    configure_paths(cv_fold)
    pre = Path(os.environ['nnUNet_preprocessed']) / dataset_name(cv_fold)
    completed = json.loads((pre / 'preparation_complete.json').read_text())
    if completed['data_lock_sha256'] != digest(HERE / 'data_lock.json'):
        raise ValueError('Preparation used different data')
    if any(digest(pre / n) != h for n, h in completed['frozen'].items()):
        raise ValueError('Prepared plans/fingerprint changed')
    split = expected_split(lock, cv_fold)
    if json.loads((pre / 'splits_final.json').read_text()) != split:
        raise ValueError('Fixed training/validation split changed')
    provenance = json.loads((pre / 'planning_provenance.json').read_text())
    if provenance['planning_patients'] != split[0]['train']:
        raise ValueError('Planning patients differ from the training partition')
    return pre


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cv-fold', type=int, choices=range(5), help='Separate T1 five-fold experiment')
    p.add_argument('--execute', action='store_true', help='Stage and preprocess only; never train')
    args = p.parse_args()
    lock = check_lock(full=args.execute)
    name = dataset_name(args.cv_fold)
    split = expected_split(lock, args.cv_fold)
    lookup = {r['case_id']: r for r in lock['patients']}
    train, val = ([lookup[pid] for pid in split[0][key]] for key in ('train', 'val'))
    print(json.dumps({'dataset': name, 'cv_fold': args.cv_fold, 'split': split, 'steps': [
        f'copy {len(train)} training T1/VS pairs, verify binary masks and grids',
        'fingerprint and default ExperimentPlanner on training ONLY (8 GB target)',
        f'freeze fingerprint/plans hashes; append {len(val)} held-out pairs',
        'preprocess 3d_fullres using frozen plans; write fixed fold 0 split',
        'no test/external staging, training, inference or postprocessing search']}, indent=2))
    if not args.execute:
        return
    configure_paths(args.cv_fold)
    raw = Path(os.environ['nnUNet_raw']) / name
    pre = Path(os.environ['nnUNet_preprocessed']) / name
    results = Path(os.environ['nnUNet_results'])
    if args.cv_fold is not None:
        results = results / name
    if raw.exists() or pre.exists() or results.exists():
        raise ValueError('Use a fresh work directory; refusing to overwrite/replan staged data')
    from nnunetv2.experiment_planning.plan_and_preprocess_api import (
        extract_fingerprint_dataset, plan_experiment_dataset, preprocess_dataset)
    (raw / 'imagesTr').mkdir(parents=True)
    (raw / 'labelsTr').mkdir()
    dataset = {'channel_names': {'0': 'T1'}, 'labels': {'background': 0, 'VS': 1},
               'numTraining': len(train), 'file_ending': '.nii.gz'}
    stage(train, raw)
    (raw / 'dataset.json').write_text(json.dumps(dataset, indent=2))
    dataset_id = int(name[7:10])
    extract_fingerprint_dataset(dataset_id, num_processes=4, check_dataset_integrity=True)
    plan_experiment_dataset(dataset_id, gpu_memory_target_in_gb=8)
    frozen = {name: digest(pre / name) for name in ('dataset_fingerprint.json', 'nnUNetPlans.json')}
    (pre / 'planning_provenance.json').write_text(json.dumps({
        'data_lock_sha256': digest(HERE / 'data_lock.json'),
        'planning_patients': [r['case_id'] for r in train], 'frozen': frozen}, indent=2))
    stage(val, raw)
    dataset['numTraining'] += len(val)
    (raw / 'dataset.json').write_text(json.dumps(dataset, indent=2))
    # DefaultPreprocessor reads raw dataset.json; refresh the trainer's copy as well.
    (pre / 'dataset.json').write_text(json.dumps(dataset, indent=2))
    preprocess_dataset(dataset_id, configurations=['3d_fullres'], num_processes=[4])
    if any(digest(pre / name) != value for name, value in frozen.items()):
        raise ValueError('Frozen plans/fingerprint changed during preprocessing')
    (pre / 'splits_final.json').write_text(json.dumps(split, indent=2))
    (pre / 'preparation_complete.json').write_text(json.dumps({
        'cv_fold': args.cv_fold,
        'data_lock_sha256': digest(HERE / 'data_lock.json'), 'frozen': frozen,
        'split_sha256': digest(pre / 'splits_final.json')}, indent=2))


if __name__ == '__main__':
    main()
