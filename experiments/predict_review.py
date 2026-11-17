"""Export all local VS-SEG T1 cases using models that excluded each patient."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from config import ROOT, DATA, WORK, EXPERIMENTS
from data import check_lock, digest
from nnunet_prepare import dataset_name, expected_split, check_preparation

OUTPUT = ROOT / 'outputs/t1/review/vsseg242_nnunet_cv'
STAGING = WORK / 'vsseg242_nnunet_cv_review'
MODEL_DIR = 'nnUNetTrainer__nnUNetPlans__3d_fullres'


def assignments(lock, splits):
    owners = {}
    for fold, split in enumerate(splits):
        if set(split['train']) & set(split['val']):
            raise ValueError('Patient appears in both training and held-out data')
        for pid in split['val']:
            if pid in owners:
                raise ValueError('Patient held out more than once')
            owners[pid] = fold
    excluded = {r['case_id'] for r in lock['excluded']}
    rows = []
    for patient in sorted(lock['patients'] + lock['excluded'], key=lambda r: r['case_id']):
        pid = patient['case_id']
        if pid in owners:
            fold, source = owners[pid], 'existing_out_of_fold'
        elif patient['split'] == 'test' or pid in excluded:
            # Fixed before inference; all five models excluded these patients.
            if any(pid in s['train'] or pid in s['val'] for s in splits):
                raise ValueError(f'Reserved/excluded patient used during development: {pid}')
            fold, source = 0, 'new_inference'
        else:
            raise ValueError(f'No held-out model for {pid}')
        if pid in splits[fold]['train']:
            raise ValueError(f'Prediction model trained on {pid}')
        rows.append(dict(case_id=pid, original_split=patient['split'], cv_fold=fold,
                         prediction_source=source))
    return rows


def plan():
    lock = check_lock()
    models, splits = [], []
    for fold in range(5):
        check_preparation(lock, fold)
        directory = ROOT / 'outputs/t1/nnunet_cv' / dataset_name(fold)
        snapshot = directory / 'setup_snapshot'
        split = json.loads((snapshot / 'splits_final.json').read_text())
        if split != expected_split(lock, fold):
            raise ValueError(f'Trained fold {fold} has unexpected patients')
        provenance = json.loads((snapshot / 'planning_provenance.json').read_text())
        if provenance['planning_patients'] != split[0]['train']:
            raise ValueError(f'Trained fold {fold} has unexpected planning patients')
        config = json.loads((snapshot / 'experiment_config.json').read_text())
        if config['cv_fold'] != fold or config['checkpoint'] != 'checkpoint_final.pth':
            raise ValueError(f'Unexpected checkpoint selection in fold {fold}')
        checkpoint = directory / MODEL_DIR / 'fold_0/checkpoint_final.pth'
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        splits.append(split[0])
        models.append(dict(cv_fold=fold, checkpoint=str(checkpoint),
                           split_sha256=digest(snapshot / 'splits_final.json')))
    rows = assignments(lock, splits)
    images = {p.parent.name: p for p in (DATA / 'VS_SEG_T1').glob('*/T1.nii.gz')}
    if set(images) != {r['case_id'] for r in rows} or len(rows) != len(images):
        raise ValueError('VS-SEG T1 image inventory differs from patient assignments')
    for row in rows:
        row['image'] = str(images[row['case_id']])
        model = Path(models[row['cv_fold']]['checkpoint']).parent
        row['source_prediction'] = str(model / 'validation' / (row['case_id'] + '.nii.gz'))
        if row['prediction_source'] == 'existing_out_of_fold' and not Path(row['source_prediction']).is_file():
            raise FileNotFoundError(row['source_prediction'])
    return lock, rows, models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    lock, rows, models = plan()
    print(json.dumps({'patients': len(rows), 'sources': dict(Counter(r['prediction_source'] for r in rows)),
                      'assigned_folds': dict(Counter(r['cv_fold'] for r in rows)),
                      'output': str(OUTPUT), 'checkpoint': 'checkpoint_final.pth', 'mirroring': True}, indent=2), flush=True)
    if not args.execute:
        return
    if OUTPUT.exists() or STAGING.exists():
        raise FileExistsError('Review output/staging already exists; refusing to overwrite')
    import importlib.metadata
    import nibabel as nib
    import torch
    from evaluate import binary_mask, check_grid, prediction_paths, write_csv, write_json
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    if importlib.metadata.version('nnunetv2') != '2.8.1' or not torch.cuda.is_available():
        raise RuntimeError('Use the existing nnU-Net 2.8.1 GPU environment')
    torch.set_num_threads(4)
    OUTPUT.mkdir(parents=True)
    inputs = STAGING / 'input'
    inputs.mkdir(parents=True)
    frozen_images = {r['case_id']: r.get('image_sha256') for r in lock['patients']}
    for model in models:
        model['checkpoint_sha256'] = digest(model['checkpoint'])
    for row in rows:
        pid = row['case_id']
        row['image_sha256'] = digest(row['image'])
        if frozen_images.get(pid) and row['image_sha256'] != frozen_images[pid]:
            raise ValueError(f'Development input changed since training: {pid}')
        if row['prediction_source'] == 'existing_out_of_fold':
            image, mask = nib.load(row['image']), nib.load(row['source_prediction'])
            check_grid(image, mask, pid)
            binary_mask(mask)
            shutil.copyfile(row['source_prediction'], OUTPUT / (pid + '.nii.gz'))
        else:
            (inputs / (pid + '_0000.nii.gz')).symlink_to(row['image'])
    predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                               perform_everything_on_device=True, device=torch.device('cuda'),
                               verbose=False, verbose_preprocessing=False, allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(str(Path(models[0]['checkpoint']).parents[1]),
                                                  use_folds=(0,), checkpoint_name='checkpoint_final.pth')
    fresh = STAGING / 'predictions'
    predictor.predict_from_files(str(inputs), str(fresh), save_probabilities=False, overwrite=False,
                                 num_processes_preprocessing=2, num_processes_segmentation_export=2)
    new_rows = [r for r in rows if r['prediction_source'] == 'new_inference']
    prediction_paths(new_rows, fresh)
    for row in new_rows:
        row['source_prediction'] = str(fresh / (row['case_id'] + '.nii.gz'))
        shutil.copyfile(row['source_prediction'], OUTPUT / (row['case_id'] + '.nii.gz'))
    prediction_paths(rows, OUTPUT)
    for row in rows:
        prediction = OUTPUT / (row['case_id'] + '.nii.gz')
        mask = nib.load(prediction)
        check_grid(nib.load(row['image']), mask, row['case_id'])
        row['foreground_voxels'] = int(binary_mask(mask).sum())
        row['prediction_sha256'] = digest(prediction)
        if row['prediction_sha256'] != digest(row['source_prediction']):
            raise ValueError(f'Copied prediction differs from source: {row["case_id"]}')
        model = models[row['cv_fold']]
        row['checkpoint'] = model['checkpoint']
        row['checkpoint_sha256'] = model['checkpoint_sha256']
        row['prediction'] = prediction.name
        row['image'] = os.path.relpath(row['image'], OUTPUT)
        row['source_prediction'] = os.path.relpath(row['source_prediction'], OUTPUT)
        row['checkpoint'] = os.path.relpath(row['checkpoint'], OUTPUT)
    write_csv(OUTPUT / 'review_manifest.csv', rows)
    write_json(OUTPUT / 'verification.json', {
        'status': 'passed', 'patients': len(rows), 'models': models,
        'all_patients_excluded_from_predicting_model_training': True,
        'coverage_binary_masks_original_T1_grids_and_copy_hashes': 'passed',
        'mirroring': True, 'new_inference_patients': len(new_rows),
        'reference_masks_loaded': False, 'scores_computed': False,
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'script_sha256': digest(__file__), 'data_lock_sha256': digest(EXPERIMENTS / 'data_lock.json')})
    (OUTPUT / 'README.md').write_text(
        '# VS-SEG T1 predictions for review\n\n'
        '242 binary model prediction masks, named <patient_id>.nii.gz, on the original processed T1 grids.\n'
        '217 unchanged out-of-fold masks use the final checkpoint that excluded that patient.\n'
        'The 24 reserved test patients and unlabelled VS-SEG-159 use outer fold 0, selected in advance;\n'
        'none of these 25 patients entered any fold training or planning. All predictions use mirroring.\n\n'
        'review_manifest.csv maps each mask to its image, predicting fold, checkpoint and SHA-256 hashes.\n'
        'Relative paths resolve from this folder. verification.json records completed checks.\n'
        'Reference masks were not loaded and no new Dice scores were calculated.\n')
    print(f'COMPLETE: {len(rows)} verified review masks in {OUTPUT}', flush=True)


if __name__ == '__main__':
    main()
