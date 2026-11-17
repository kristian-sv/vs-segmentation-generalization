#!/usr/bin/env python3
"""Export a manual prediction onto its original MRI grid, then score either pipeline."""
import argparse
import csv
import json
from pathlib import Path
import re
import sys

import nibabel as nib
import numpy as np

from config import ROOT, EXPERIMENTS, DATA, load_config
from data import check_lock, digest
AFFINE_ATOL = 1e-4  # mm, elementwise; relative tolerance is deliberately zero
FIELDS = ['case_id', 'image', 'reference', 'image_sha256', 'reference_sha256']


def write_json(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False, default=str)
        f.write('\n')


def write_csv(path, rows, fields=None):
    with Path(path).open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def read_manifest(path):
    """Explicit IDs and hashed image/reference pairs; relative paths use CSV directory."""
    path = Path(path).resolve()
    with path.open(newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != FIELDS:
            raise ValueError(f'Manifest columns must be exactly {FIELDS}')
        rows = list(reader)
    ids, paths = set(), set()
    if not rows:
        raise ValueError('Empty manifest')
    for row in rows:
        pid = row['case_id']
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', pid or '') or pid in ids:
            raise ValueError(f'Duplicate/invalid patient ID: {pid}')
        ids.add(pid)
        for key in ('image', 'reference'):
            p = (path.parent / row[key]).resolve()
            if not row[key] or p in paths or not p.is_file():
                raise ValueError(f'Missing/reused {key} path: {pid}')
            paths.add(p)
            row[key] = str(p)
            expected = row[key + '_sha256']
            if not re.fullmatch('[0-9a-f]{64}', expected or '') or digest(p) != expected:
                raise ValueError(f'Input hash mismatch: {pid} {key}')
    return rows


def check_grid(image, other, context='image/reference'):
    for volume in (image, other):
        if len(volume.shape) != 3 or not np.isfinite(volume.affine).all():
            raise ValueError(f'Expected finite 3D grid: {context}')
        if abs(np.linalg.det(volume.affine[:3, :3])) < 1e-8:
            raise ValueError(f'Singular affine: {context}')
        if volume.header.get_xyzt_units()[0] != 'mm':
            raise ValueError(f'Spatial units must explicitly be mm: {context}')
    if image.shape != other.shape or not np.allclose(
            image.affine, other.affine, atol=AFFINE_ATOL, rtol=0):
        raise ValueError(f'Grid mismatch: {context}')


def binary_mask(volume):
    data = np.asanyarray(volume.dataobj)
    if not np.isfinite(data).all() or not np.isin(data, (0, 1)).all():
        raise ValueError('Expected finite binary mask with labels 0 and 1 only')
    return data.astype(bool)


def pair_geometry(row):
    image, reference = nib.load(row['image']), nib.load(row['reference'])
    check_grid(image, reference, row['case_id'])
    binary_mask(reference)
    return image, reference


def prediction_paths(rows, directory):
    """nnU-Net and manual export both use <exact-case-id>.nii.gz; no positional pairing."""
    paths = {p.name[:-7]: p for p in Path(directory).glob('*.nii.gz')}
    expected = {r['case_id'] for r in rows}
    if set(paths) != expected or list(Path(directory).glob('*.nii')):
        raise ValueError(f'Prediction coverage mismatch: missing={sorted(expected-set(paths))}, '
                         f'extra={sorted(set(paths)-expected)}; only .nii.gz is accepted')
    return paths


def patient_metrics(pred, ref, affine):
    """Dice overlap and tumor volumes; boundary metrics remain in the archive."""
    n_pred, n_ref = int(pred.sum()), int(ref.sum())
    status = ('both_empty' if not n_pred and not n_ref else
              'prediction_empty' if not n_pred else 'reference_empty' if not n_ref else 'nonempty')
    dice = 2.0 * np.count_nonzero(pred & ref) / (n_pred + n_ref) if n_pred + n_ref else 1.0
    voxel_ml = abs(float(np.linalg.det(affine[:3, :3]))) / 1000
    return dict(dice=float(dice), prediction_voxels=n_pred, reference_voxels=n_ref,
                prediction_ml=n_pred * voxel_ml, reference_ml=n_ref * voxel_ml, empty_status=status)


def summary(values):
    values = np.asarray(values, dtype=float)
    return dict(n=len(values), mean=float(values.mean()) if len(values) else None,
                median=float(np.median(values)) if len(values) else None,
                sample_sd=float(values.std(ddof=1)) if len(values) > 1 else None)


def manual_pipeline(family, seed):
    from dataset import base_transforms
    settings = load_config(family, seed)
    return settings, base_transforms(settings)


def validate_checkpoint_config(checkpoint, values):
    """Require the selected checkpoint's sibling snapshot to match EVERY frozen key."""
    path = Path(checkpoint).parent / 'experiment_config.json'
    saved = json.loads(path.read_text())
    expected = json.loads(json.dumps(values, default=str))
    mismatches = [key for key, value in expected.items() if saved.get(key) != value]
    if mismatches:
        raise ValueError('Checkpoint experiment_config.json differs from frozen run: ' + ', '.join(mismatches))
    return dict(checkpoint_sha256=digest(checkpoint), experiment_config=str(path.resolve()),
                experiment_config_sha256=digest(path))


def preprocess_image(path, transform):
    from monai.transforms.utils import allow_missing_keys_mode
    # The image branch is identical to training; labels never enter inference.
    with allow_missing_keys_mode(transform):
        return transform({'image': str(path)})['image']


def restore_probabilities(probabilities, transformed_image, transform, original):
    """Invert every spatial transform on two continuous channels BEFORE argmax."""
    import torch
    from monai.data import MetaTensor
    from monai.transforms import Invertd
    p = torch.as_tensor(probabilities).detach().float().cpu()
    if tuple(p.shape) != (2, *transformed_image.shape[1:]):
        raise ValueError('Expected two probability channels on the transformed image grid')
    if not torch.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('Expected finite probabilities in [0, 1]')
    if not torch.allclose(p.sum(0), torch.ones_like(p[0]), atol=1e-4, rtol=0):
        raise ValueError('Probability channels must sum to one')
    inverse = Invertd(keys='probabilities', orig_keys='image', transform=transform,
                      nearest_interp=False, to_tensor=True, device='cpu')
    # Invertd uses the image MetaTensor trace only when its prediction input is
    # also a MetaTensor; a plain Tensor selects the legacy *_transforms dict API.
    restored = inverse({'probabilities': MetaTensor(p), 'image': transformed_image})['probabilities']
    if tuple(restored.shape) != (2, *original.shape) or not np.allclose(
            restored.affine.cpu().numpy(), original.affine, atol=AFFINE_ATOL, rtol=0):
        raise ValueError(f'Inverse grid mismatch: shape {tuple(restored.shape)} vs {(2, *original.shape)}; '
                         f'affine {restored.affine.cpu().numpy().tolist()} vs {original.affine.tolist()}')
    restored = restored.as_tensor().numpy()
    if not np.isfinite(restored).all():
        raise ValueError('Nonfinite restored probabilities')
    # Crop inversion introduces zeros in both channels outside the observed crop.
    # Explicit background fill prevents arbitrary outside-crop foreground labels.
    total = restored.sum(axis=0)
    restored[0, total <= 0] = 1.0
    restored /= restored.sum(axis=0, keepdims=True)
    return restored


def save_mask(mask, original, path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    header = original.header.copy()
    header.set_data_dtype(np.uint8)
    header.set_slope_inter(1, 0)
    output = nib.Nifti1Image(mask.astype(np.uint8), original.affine, header)
    nib.save(output, path)
    check_grid(original, nib.load(path), str(path))


def score(rows, pred_dir, compare_dir=None):
    paths = prediction_paths(rows, pred_dir)
    other = prediction_paths(rows, compare_dir) if compare_dir else None
    # Validate ALL coverage, grids and label values before computing ANY patient score.
    for row in rows:
        image, _ = pair_geometry(row)
        for directory in (paths, other) if other else (paths,):
            pred = nib.load(directory[row['case_id']])
            check_grid(image, pred, row['case_id'] + ' prediction')
            binary_mask(pred)
    records = []
    for row in rows:
        pid = row['case_id']
        image, reference = nib.load(row['image']), nib.load(row['reference'])
        ref = binary_mask(reference)
        record = dict(row, prediction=str(paths[pid].resolve()),
                      prediction_sha256=digest(paths[pid]))
        record.update(patient_metrics(binary_mask(nib.load(paths[pid])), ref, image.affine))
        if other:
            comparator = patient_metrics(binary_mask(nib.load(other[pid])), ref, image.affine)
            record.update(compare_prediction=str(other[pid].resolve()),
                          compare_prediction_sha256=digest(other[pid]),
                          compare_dice=comparator['dice'],
                          dice_difference=record['dice'] - comparator['dice'])
        records.append(record)
    report = dict(patient_ids=[r['case_id'] for r in records], dice=summary([r['dice'] for r in records]),
                  empty_counts={s: sum(r['empty_status'] == s for r in records)
                                for s in ('nonempty', 'both_empty', 'prediction_empty', 'reference_empty')})
    if other:
        report['dice_difference_prediction_minus_compare'] = summary([r['dice_difference'] for r in records])
    return records, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    manifest = commands.add_parser('manifest', help='Write frozen development pairs only; no image loading')
    manifest.add_argument('--split', choices=['train', 'val'], required=True)
    manifest.add_argument('--output', type=Path, required=True)
    export = commands.add_parser('manual-export', help='One checkpoint; no TTA, ensemble or postprocessing')
    export.add_argument('--manual', choices=['baseline', 'dropout01'], required=True)
    export.add_argument('--seed', type=int, choices=[42, 43, 44], required=True)
    export.add_argument('--checkpoint', type=Path, required=True)
    export.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    scoring = commands.add_parser('score', help='Score original-grid manual OR nnU-Net binary masks')
    scoring.add_argument('--predictions', type=Path, required=True)
    scoring.add_argument('--compare', type=Path, help='Optional second prediction set; pair by patient ID')
    for command in (export, scoring):
        command.add_argument('--manifest', type=Path, required=True)
        command.add_argument('--output', type=Path, required=True, help='New output directory, never overwritten')
    args = parser.parse_args()
    if args.command == 'manifest':
        rows = [dict(case_id=r['case_id'], image=str(DATA / r['image']), reference=str(DATA / r['label']),
                     image_sha256=r['image_sha256'], reference_sha256=r['label_sha256'])
                for r in check_lock()['patients'] if r['split'] == args.split]
        write_csv(args.output, rows, FIELDS)
        print(f'Wrote {len(rows)} {args.split} pairs to {args.output}')
        return
    rows = read_manifest(args.manifest)
    if args.output.exists():
        raise FileExistsError(args.output)
    provenance = dict(argv=sys.argv, manifest_sha256=digest(args.manifest),
                      scorer_sha256=digest(__file__), protocol_sha256=digest(EXPERIMENTS / 'README.md'))
    if args.command == 'score':
        records, report = score(rows, args.predictions, args.compare)
        args.output.mkdir(parents=True)
        write_csv(args.output / 'patients.csv', records)
        write_json(args.output / 'summary.json', dict(provenance=provenance, **report))
    else:
        import torch
        from monai.inferers import sliding_window_inference
        from train import build_model
        values, transform = manual_pipeline(args.manual, args.seed)
        checkpoint_provenance = validate_checkpoint_config(args.checkpoint, values)
        for row in rows:
            pair_geometry(row)
        model = build_model(values)
        model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True))
        model.to(args.device).eval()
        args.output.mkdir(parents=True)
        provenance.update(**checkpoint_provenance, frozen_config=values,
                          tta=False, ensemble=False, postprocessing=None,
                          continuous_export='softmax of blended logits, bilinear inverse, then argmax')
        write_json(args.output / 'provenance.json', provenance)
        with torch.inference_mode():
            for row in rows:
                original = nib.load(row['image'])
                image = preprocess_image(row['image'], transform)
                logits = sliding_window_inference(
                    image.as_tensor()[None].to(args.device), roi_size=values['patch_size'],
                    sw_batch_size=values['sw_batch_size'], predictor=model,
                    overlap=values['sw_overlap'], mode='constant', device='cpu')
                probs = restore_probabilities(logits[0].softmax(dim=0), image, transform, original)
                save_mask(probs.argmax(axis=0), original, args.output / (row['case_id'] + '.nii.gz'))
                print(f'Exported {row["case_id"]}', flush=True)
        write_json(args.output / 'complete.json', dict(patient_ids=[r['case_id'] for r in rows]))
    print(f'{args.command} complete: {len(rows)} patients; {args.output}')


if __name__ == '__main__':
    main()
