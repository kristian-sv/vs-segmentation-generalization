#!/usr/bin/env python3
"""Per-case failure analysis of the seven frozen T1 models (validation + MC-RC).

Reads existing images, references and predictions only; reuses the common scorer in
src/evaluate.py. The single bounded exception is nnU-Net on axis-reordered MC-RC
copies (reorient-stage -> GPU job in model_insight.slurm -> reorient-restore).

    python analysis/model_insight.py orientation | reorient-stage | reorient-restore | cases | bundle

Stress test on the 24 validation patients (perturbed copies, existing models, no training):

    stress-build -> GPU array 'stress-predict' -> stress-score -> stress-summary | stress-bundle
"""
import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
try:
    import nibabel as nib
    from evaluate import (binary_mask, check_grid, manual_pipeline, patient_metrics,  # noqa: E402
                          prediction_paths, preprocess_image, read_manifest, save_mask, score, write_csv,
                          write_json)
    from data import digest
except ModuleNotFoundError:  # stress-summary runs under the matplotlib module (Python 3.13, no nibabel)
    nib = None

RECORD = ROOT / 'analysis/reports/t1_evaluation_20260912'
OUT = ROOT / 'analysis/reports/model_insight_20261012'
REORIENTED = OUT / 'nnunet_reoriented'
STRESS = OUT / 'stress_test'
# Levels fixed on 12 Oct 2026 before any stress-test inference. COV_012/COV_100 and 100's slice spacing
# are MC-RC acquisition metadata from cases.csv (012: 16 x 3.3 mm; 100: 36 x 0.91 mm), not scores.
COV_012, COV_100, SPACING_100 = 52.8, 32.76, 0.91
EDGE_MM = 5.0
CONDITIONS = {
    'C0': [],
    'S50': [('slab', 'SI', 50.0)], 'S40': [('slab', 'SI', 40.0)], 'S33': [('slab', 'SI', 33.0)],
    'S40edge': [('edge_slab', 'SI', 40.0)],
    'T3': [('thick', 'SI', 3.0)], 'T4': [('thick', 'SI', 4.0)],
    'S40T3': [('slab', 'SI', 40.0), ('thick', 'SI', 3.3)],
    'AP': [('slab', 'AP', COV_012)],
    'COR_thin': [('slab', 'AP', COV_100), ('thick', 'AP', SPACING_100), ('store', 'LIP', None)],
    'COR_thick': [('slab', 'AP', COV_012), ('thick', 'AP', 3.3), ('store', 'LIP', None)],
}
RESTACKED = ('COR_thin', 'COR_thick')  # nnU-Net also runs on these after re-stacking to training order
PREDICTIONS = ROOT / 'outputs/t1/evaluation_20260912'
RAW = ROOT / 'data/raw/datasets'
COHORTS = {'validation': RECORD / 'validation_manifest.csv', 'mcrc': RECORD / 'mcrc_manifest.csv'}
# All 217 VS-SEG development images are stored with voxel axes L, P, S (axial, 0 deg tilt).
TRAINING_AXCODES = ('L', 'P', 'S')
WORLD = {'L': 'LR', 'R': 'LR', 'P': 'AP', 'A': 'AP', 'S': 'SI', 'I': 'SI'}
PLANES = ('sagittal', 'coronal', 'axial')  # slice normal along patient x, y, z
TAGS = dict(orientation='0020|0037', series_description='0008|103e', acquisition_type='0018|0023',
            slice_thickness='0018|0050', contrast_agent='0018|0010', manufacturer='0008|0070',
            model='0008|1090', field_strength='0018|0087')
CONTRAST_WORDS = re.compile(r'gad|\bgd\b|\+ ?c\b|\bc\+|post|contrast|dotarem|magnevist|gadovist', re.I)


def read_rows(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def runs():
    return read_rows(RECORD / 'runs.csv')


def series_files(cohort, pid):
    """The DICOM series behind each processed T1, from the existing identity records."""
    if cohort == 'mcrc':
        uid = {r['patient_id']: r['selected_mcrc_series_uid']
               for r in read_rows(ROOT / 'metadata/evaluation/mcrc_t1_patient_independent.csv')}[pid]
        directory = RAW / 'VS-MC-RC' / uid
    else:
        [classic] = [r['Classic Directory Name'] for r in read_rows(RAW / 'DirectoryNamesMappingModality.csv')
                     if r['Modality'] == 'T1 image' and r['Descriptive Directory Name'].startswith(pid + '/')]
        directory = RAW / 'VS-SEG' / classic.split('/')[2]
    return sorted(directory.glob('*.dcm'))


def dicom_header(cohort, pid):
    """Header-only DICOM read with SimpleITK; slice-normal direction from ImageOrientationPatient."""
    import SimpleITK as sitk
    files = series_files(cohort, pid)
    values = []
    for path in files:
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(path))
        reader.ReadImageInformation()
        values.append({k: reader.GetMetaData(t).strip() if reader.HasMetaDataKey(t) else ''
                       for k, t in TAGS.items()})
    cosines_all = np.array([v['orientation'].split('\\') for v in values], float)
    # Text values can differ in the last printed digit (VS-MC-RC-081: 1e-7); compare numerically.
    if not files or np.abs(cosines_all - cosines_all[0]).max() > 1e-4:
        raise ValueError(f'{pid}: expected one DICOM series orientation in {len(files)} files')
    header = values[len(values) // 2]
    cosines = cosines_all[len(values) // 2]
    normal = np.cross(cosines[:3], cosines[3:]) * [-1, -1, 1]  # DICOM LPS -> NIfTI RAS
    header.update(dicom_files=len(files), slice_normal=normal / np.linalg.norm(normal))
    return header


def axis_angle(vector):
    """Angle (deg) between a direction and its nearest patient axis."""
    vector = np.asarray(vector, float) / np.linalg.norm(vector)
    return math.degrees(math.acos(min(1.0, float(np.abs(vector).max()))))


def orientation_record(cohort, row, dicom):
    image = nib.load(row['image'])
    codes = nib.aff2axcodes(image.affine)
    zooms = np.array(image.header.get_zooms()[:3], float)
    columns = image.affine[:3, :3] / zooms
    # Voxel axis whose world (RAS) direction is closest to the DICOM slice normal
    slice_axis = int(np.argmax(np.abs(columns.T @ dicom['slice_normal'])))
    anisotropy = zooms.max() / zooms.min()
    thick = int(np.argmax(zooms)) if anisotropy > 1.05 else None
    world = [WORLD[c] for c in codes]
    return dict(
        cohort=cohort, case_id=row['case_id'], stored_axcodes=''.join(codes), shape='x'.join(map(str, image.shape)),
        spacing_mm='x'.join(f'{z:.3f}' for z in zooms),
        acq_plane=PLANES[int(np.argmax(np.abs(dicom['slice_normal'])))], acq_type=dicom['acquisition_type'],
        obliquity_deg=round(axis_angle(dicom['slice_normal']), 2),
        grid_max_tilt_deg=round(max(axis_angle(c) for c in columns.T), 2),
        slice_voxel_axis=slice_axis, slice_world_axis=world[slice_axis],
        thick_voxel_axis='' if thick is None else thick, anisotropy=round(float(anisotropy), 3),
        nnunet_array_axes='/'.join(world), nnunet_axes_permuted=world != ['LR', 'AP', 'SI'],
        nnunet_input_matches_training=codes == TRAINING_AXCODES)


def orientation(_args):
    """Step 1: stored orientation, acquisition plane, and what each network receives."""
    _, transform = manual_pipeline('baseline', 42)  # image branch is identical for all six manual runs
    records = []
    for cohort, manifest in COHORTS.items():
        for row in read_manifest(manifest):
            record = orientation_record(cohort, row, dicom_header(cohort, row['case_id']))
            seen = preprocess_image(row['image'], transform)
            affine = seen.affine.cpu().numpy()
            record.update(manual_network_axcodes=''.join(nib.aff2axcodes(affine)),
                          manual_network_shape='x'.join(map(str, seen.shape[1:])),
                          manual_network_max_tilt_deg=round(max(axis_angle(c) for c in affine[:3, :3].T), 2))
            records.append(record)
            print(record, flush=True)
    write_csv(OUT / 'orientation.csv', records)


def training_order(image):
    """Axis permutation/flip ONLY (no resampling, no rotation) to the training voxel order."""
    transform = nib.orientations.ornt_transform(nib.io_orientation(image.affine),
                                                nib.orientations.axcodes2ornt(TRAINING_AXCODES))
    affine = image.affine @ nib.orientations.inv_ornt_aff(transform, image.shape)
    return transform, affine


def save_like(data, affine, template, path):
    """Write data on a new grid, keeping the template's header fields (units, form codes)."""
    header = template.header.copy()
    header.set_data_dtype(data.dtype)
    output = nib.Nifti1Image(data, affine, header)
    output.set_qform(affine, code=int(header['qform_code']) or 1)
    output.set_sform(affine, code=int(header['sform_code']) or 1)
    nib.save(output, path)


def write_training_order(original, path):
    transform, affine = training_order(original)
    data = nib.orientations.apply_orientation(np.asanyarray(original.dataobj), transform)
    save_like(data, affine, original, path)
    return transform, affine, data


def restore_from_training_order(lps, original):
    """Undo training_order() on a predicted mask; returns the mask on the original grid and the grid error."""
    back = nib.orientations.ornt_transform(nib.orientations.axcodes2ornt(TRAINING_AXCODES),
                                           nib.io_orientation(original.affine))
    error = float(np.abs(lps.affine @ nib.orientations.inv_ornt_aff(back, lps.shape) - original.affine).max())
    if error > 1e-3:
        raise ValueError(f'Restored grid differs from original by {error} mm')
    return nib.orientations.apply_orientation(binary_mask(lps), back), error


def reorient_stage(_args):
    import SimpleITK as sitk
    stage = REORIENTED / 'input'
    stage.mkdir(parents=True)
    records = []
    for row in read_manifest(COHORTS['mcrc']):
        original = nib.load(row['image'])
        path = stage / f"{row['case_id']}_0000.nii.gz"
        transform, affine, data = write_training_order(original, path)
        written = nib.load(path)
        identity = bool((transform == [[0, 1], [1, 1], [2, 1]]).all())
        a, b = sitk.ReadImage(row['image']), sitk.ReadImage(str(path))
        records.append(dict(
            case_id=row['case_id'], stored_axcodes=''.join(nib.aff2axcodes(original.affine)),
            staged_axcodes=''.join(nib.aff2axcodes(written.affine)), identity=identity,
            transform=';'.join(f'{int(ax)}:{int(flip)}' for ax, flip in transform),
            staged_voxels_equal=bool(np.array_equal(np.asanyarray(written.dataobj), data)),
            staged_affine_error_mm=float(np.abs(written.affine - affine).max()),
            sitk_spacing_original='x'.join(f'{s:.4f}' for s in a.GetSpacing()),
            sitk_spacing_staged='x'.join(f'{s:.4f}' for s in b.GetSpacing()),
            sitk_identical_to_original=identity and a.GetSize() == b.GetSize() and np.allclose(
                a.GetSpacing(), b.GetSpacing(), atol=1e-6) and np.allclose(a.GetDirection(), b.GetDirection(), atol=1e-6)
            and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=1e-4) and np.array_equal(
                sitk.GetArrayViewFromImage(a), sitk.GetArrayViewFromImage(b))))
        print(records[-1], flush=True)
    if not all(r['staged_voxels_equal'] and r['staged_axcodes'] == ''.join(TRAINING_AXCODES) for r in records):
        raise ValueError('Staged copy differs from the permuted original')
    write_csv(REORIENTED / 'staging.csv', records)


def reorient_restore(_args):
    """Undo the axis reordering on nnU-Net's masks, compare with frozen masks, score with the common scorer."""
    rows = read_manifest(COHORTS['mcrc'])
    restored = REORIENTED / 'predictions'
    restored.mkdir()
    frozen = prediction_paths(rows, PREDICTIONS / 'mcrc/nnunet_s42/predictions')
    staged = {r['case_id']: r for r in read_rows(REORIENTED / 'staging.csv')}
    for row in rows:
        pid = row['case_id']
        original = nib.load(row['image'])
        mask, error = restore_from_training_order(nib.load(REORIENTED / f'predictions_lps/{pid}.nii.gz'), original)
        save_mask(mask, original, restored / f'{pid}.nii.gz')
        staged[pid]['restore_affine_error_mm'] = error
        staged[pid]['voxels_differing_from_frozen'] = int(
            (binary_mask(nib.load(restored / f'{pid}.nii.gz')) != binary_mask(nib.load(frozen[pid]))).sum())
    records, report = score(rows, restored)
    (REORIENTED / 'scores').mkdir()
    write_csv(REORIENTED / 'scores/patients.csv', records)
    write_json(REORIENTED / 'scores/summary.json', report)
    old = {r['case_id']: r for r in read_rows(RECORD / 'patient_scores.csv')
           if r['cohort'] == 'mcrc' and r['run'] == 'nnunet_s42'}
    table = []
    for r in records:
        pid, s, o = r['case_id'], staged[r['case_id']], old[r['case_id']]
        table.append(dict(case_id=pid, stored_axcodes=s['stored_axcodes'], axes_reordered=s['identity'] == 'False',
                          reference_ml=r['reference_ml'], dice_frozen=float(o['dice']), dice_reoriented=r['dice'],
                          dice_change=r['dice'] - float(o['dice']), prediction_ml_frozen=float(o['prediction_ml']),
                          prediction_ml_reoriented=r['prediction_ml'], empty_frozen=o['empty_status'],
                          empty_reoriented=r['empty_status'], voxels_differing_from_frozen=s['voxels_differing_from_frozen'],
                          restore_affine_error_mm=s['restore_affine_error_mm']))
    write_csv(OUT / 'nnunet_reoriented_scores.csv', table)
    for name in ('dice_frozen', 'dice_reoriented'):
        values = [t[name] for t in table]
        print(f'{name}: mean {np.mean(values):.6f} median {np.median(values):.6f}')
    print('Cases differing from frozen masks:', [t['case_id'] for t in table if t['voxels_differing_from_frozen']])


def surface_metrics(pred, ref, zooms):
    import torch
    from monai.metrics import compute_average_surface_distance, compute_hausdorff_distance
    p, r = (torch.from_numpy(x[None, None].astype(np.float32)) for x in (pred, ref))
    spacing = [float(z) for z in zooms]
    hd95 = compute_hausdorff_distance(p, r, include_background=True, percentile=95, spacing=spacing)
    assd = compute_average_surface_distance(p, r, include_background=True, symmetric=True, spacing=spacing)
    return float(hd95.item()), float(assd.item())


def category(dice, pred_empty, overlap, recall, precision):
    if dice > 0.7:
        return 'success'
    if pred_empty:
        return 'empty'
    if not overlap:
        return 'wrong_location'
    if precision >= 0.7 > recall:
        return 'under'
    if recall >= 0.7 > precision:
        return 'over'
    return 'partial'


def edge_distance(ref, zooms):
    """Smallest distance (mm) from a reference voxel centre to the outer face of the image grid."""
    index = np.argwhere(ref)
    if not len(index):
        return float('nan')
    return float(min(min((index[:, i].min() + 0.5) * zooms[i], (ref.shape[i] - index[:, i].max() - 0.5) * zooms[i])
                     for i in range(3)))


def acquisition(cohort, row, image, ref, orient, dicom):
    zooms = np.array(image.header.get_zooms()[:3], float)
    data = np.asanyarray(image.dataobj)
    nonzero = data[data != 0]
    slice_axis = orient['slice_voxel_axis']
    ras = nib.io_orientation(image.affine)[:, 0].astype(int)  # voxel axis i -> RAS axis
    extent = {int(ras[i]): float(image.shape[i] * zooms[i]) for i in range(3)}
    edge = edge_distance(ref, zooms)
    agent, source = dicom['contrast_agent'], 'dicom_tag'
    if not agent:
        match = CONTRAST_WORDS.search(dicom['series_description'])
        agent, source = (match.group(0), 'series_description') if match else ('', 'none_found')
    return dict(
        stored_axcodes=orient['stored_axcodes'], acq_plane=orient['acq_plane'], acq_type=orient['acq_type'],
        obliquity_deg=orient['obliquity_deg'], nnunet_axes_permuted=orient['nnunet_axes_permuted'],
        slice_thickness=float(dicom['slice_thickness']) if dicom['slice_thickness'] else '',
        spacing_mm=orient['spacing_mm'], slice_spacing_mm=float(zooms[slice_axis]),
        inplane_spacing_mm=float(np.delete(zooms, slice_axis).mean()), n_slices=int(image.shape[slice_axis]),
        extent_x_mm=extent[0], extent_y_mm=extent[1], extent_z_mm=extent[2], tumor_to_fov_edge_mm=float(edge),
        contrast_agent=agent, contrast_agent_source=source, series_description=dicom['series_description'],
        manufacturer=dicom['manufacturer'], model=dicom['model'],
        field_strength=float(dicom['field_strength']) if dicom['field_strength'] else '',
        saturated_frac=float((nonzero == nonzero.max()).mean()))


def case_metrics(pred, ref, image):
    from scipy import ndimage
    zooms = image.header.get_zooms()[:3]
    m = patient_metrics(pred, ref, image.affine)
    overlap = int(np.count_nonzero(pred & ref))
    labels, n = ndimage.label(pred, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(labels.ravel(), minlength=n + 1)
    touching = np.unique(labels[ref & pred])
    fp = np.setdiff1d(np.arange(1, n + 1), touching)
    voxel_ml = abs(float(np.linalg.det(image.affine[:3, :3]))) / 1000
    recall = overlap / m['reference_voxels'] if m['reference_voxels'] else float('nan')
    precision = overlap / m['prediction_voxels'] if m['prediction_voxels'] else float('nan')
    out = dict(dice=m['dice'], ref_ml=m['reference_ml'], pred_ml=m['prediction_ml'],
               pred_empty=m['prediction_voxels'] == 0, empty_status=m['empty_status'],
               signed_vol_err_ml=m['prediction_ml'] - m['reference_ml'], recall=recall, precision=precision,
               hd95_mm='', assd_mm='', detected=overlap > 0, n_pred_components=int(n),
               n_fp_components=int(len(fp)), fp_ml=float(sizes[fp].sum() * voxel_ml), centroid_dist_mm='')
    if m['prediction_voxels'] and m['reference_voxels']:
        out['hd95_mm'], out['assd_mm'] = surface_metrics(pred, ref, zooms)
        centre = [nib.affines.apply_affine(image.affine, np.argwhere(x).mean(axis=0)) for x in (pred, ref)]
        out['centroid_dist_mm'] = float(np.linalg.norm(centre[0] - centre[1]))
    out['auto_category'] = category(m['dice'], out['pred_empty'], overlap > 0, recall, precision)
    return out


def cases(_args):
    """One row per (patient, model); Dice/volumes must equal the frozen patient_scores.csv exactly."""
    frozen = {(r['cohort'], r['run'], r['case_id']): r for r in read_rows(RECORD / 'patient_scores.csv')}
    orient = {(r['cohort'], r['case_id']): r for r in read_rows(OUT / 'orientation.csv')}
    records, mismatches = [], []
    for cohort, manifest in COHORTS.items():
        rows = read_manifest(manifest)
        paths = {run['run']: prediction_paths(rows, PREDICTIONS / cohort / run['run'] / 'predictions') for run in runs()}
        for row in rows:
            pid = row['case_id']
            image, reference = nib.load(row['image']), nib.load(row['reference'])
            check_grid(image, reference, pid)
            ref = binary_mask(reference)
            o = orient[cohort, pid]
            o['slice_voxel_axis'] = int(o['slice_voxel_axis'])
            acq = acquisition(cohort, row, image, ref, o, dicom_header(cohort, pid))
            for run in runs():
                prediction = nib.load(paths[run['run']][pid])
                check_grid(image, prediction, f'{pid} {run["run"]}')
                metrics = case_metrics(binary_mask(prediction), ref, image)
                old = frozen[cohort, run['run'], pid]
                if (metrics['dice'] != float(old['dice']) or metrics['ref_ml'] != float(old['reference_ml'])
                        or metrics['pred_ml'] != float(old['prediction_ml'])
                        or metrics['empty_status'] != old['empty_status']):
                    mismatches.append((cohort, run['run'], pid, metrics['dice'], old['dice']))
                del metrics['empty_status']
                records.append(dict(cohort=cohort, case_id=pid, run=run['run'], family=run['family'],
                                    seed=run['seed'], **metrics, **acq))
            print(pid, flush=True)
    if mismatches or len(records) != len(frozen):
        raise SystemExit(f'Dice/volume mismatch with patient_scores.csv ({len(records)} rows): {mismatches[:10]}')
    write_csv(OUT / 'cases.csv', records)
    outcome_table(records)


def outcome_table(records):
    """Medians/ranges of acquisition columns by MC-RC outcome, per pipeline (patient level)."""
    numeric = ['obliquity_deg', 'slice_thickness', 'slice_spacing_mm', 'inplane_spacing_mm', 'n_slices',
               'extent_x_mm', 'extent_y_mm', 'extent_z_mm', 'tumor_to_fov_edge_mm', 'field_strength',
               'saturated_frac', 'ref_ml']
    mcrc = [r for r in records if r['cohort'] == 'mcrc']
    patients = {r['case_id']: r for r in mcrc}
    dice = {}
    for r in mcrc:
        pipeline = 'nnunet' if r['family'] == 'nnunet' else 'manual'
        dice.setdefault((pipeline, r['case_id']), []).append(r['dice'])
    table = []
    for pipeline in ('manual', 'nnunet'):
        for outcome, rule in (('success', lambda d: d > 0.7), ('failure', lambda d: d <= 0.7)):
            ids = sorted(p for (q, p), d in dice.items() if q == pipeline and rule(np.mean(d)))
            for name in numeric:
                values = [float(patients[p][name]) for p in ids if patients[p][name] != '']
                table.append(dict(pipeline=pipeline, outcome=outcome, variable=name, n=len(values),
                                  median=float(np.median(values)) if values else '',
                                  min=min(values) if values else '', max=max(values) if values else '',
                                  cases=' '.join(p.replace('VS-MC-RC-', '') for p in ids)))
    write_csv(OUT / 'acquisition_by_outcome.csv', table)


def bundle(_args):
    """viewer_bundle/data/<cohort>/<patient>/{image,gt,pred_<run>}.nii.gz as absolute symlinks."""
    target = OUT / 'viewer_bundle'
    target.mkdir()
    (target / 'cases.csv').symlink_to(OUT / 'cases.csv')
    count = 1
    for cohort, manifest in COHORTS.items():
        rows = read_manifest(manifest)
        sources = {run['run']: prediction_paths(rows, PREDICTIONS / cohort / run['run'] / 'predictions') for run in runs()}
        if cohort == 'mcrc':
            sources['nnunet_s42_reoriented'] = prediction_paths(rows, REORIENTED / 'predictions')
        for row in rows:
            directory = target / 'data' / cohort / row['case_id']
            directory.mkdir(parents=True)
            links = {'image': Path(row['image']), 'gt': Path(row['reference']),
                     **{f'pred_{run}': paths[row['case_id']].resolve() for run, paths in sources.items()}}
            for name, source in links.items():
                (directory / f'{name}.nii.gz').symlink_to(source)
                count += 1
    print(f'{count} symlinks under {target}')


def block_average(data, axis, spacing, target):
    """Area-weighted averaging of `spacing`-mm slices into `target`-mm slices (any ratio); drops a partial last slice."""
    n_in = data.shape[axis]
    n_out = int(math.floor(n_in * spacing / target + 1e-6))
    weights = np.zeros((n_out, n_in))
    for j in range(n_out):
        for i in range(n_in):
            weights[j, i] = max(0.0, min((j + 1) * target, (i + 1) * spacing) - max(j * target, i * spacing)) / target
    out = np.moveaxis(np.tensordot(weights, np.moveaxis(data, axis, 0).astype(np.float64), axes=(1, 0)), 0, axis)
    grid = np.eye(4)  # output slice j is centred at input index -0.5 + (j + 0.5) * target / spacing
    grid[axis, axis] = target / spacing
    grid[axis, 3] = -0.5 + target / (2 * spacing)
    return out, grid


def perturb(image, ref, steps, side):
    """Apply one condition's steps to image and reference together; returns a new real-looking grid."""
    data, affine, truncated = np.asanyarray(image.dataobj), image.affine.copy(), False
    for kind, axis_name, value in steps:
        if kind == 'store':
            transform = nib.orientations.ornt_transform(nib.io_orientation(affine), nib.orientations.axcodes2ornt(axis_name))
            affine = affine @ nib.orientations.inv_ornt_aff(transform, data.shape)
            data, ref = (nib.orientations.apply_orientation(x, transform) for x in (data, ref))
            continue
        codes = nib.aff2axcodes(affine)
        axis = [WORLD[c] for c in codes].index(axis_name)
        spacing = float(np.linalg.norm(affine[:3, axis]))
        if kind == 'thick':
            data, grid = block_average(data, axis, spacing, value)
            ref = block_average(ref.astype(np.float64), axis, spacing, value)[0] >= 0.5  # partial-volume threshold
            affine = affine @ grid
            continue
        n, index = int(round(value / spacing)), np.argwhere(ref)[:, axis]
        if kind == 'slab':
            start = int(round(index.mean() - (n - 1) / 2))
        else:  # edge_slab: nearest tumour voxel centre (gap + 0.5) voxels from the chosen slab face
            gap = int(round(EDGE_MM / spacing - 0.5))
            high = (side == 'superior') == (codes[axis] == 'S')
            start = int(index.max()) + gap + 1 - n if high else int(index.min()) - gap
        start = min(max(start, 0), data.shape[axis] - n)
        crop = [slice(None)] * 3
        crop[axis] = slice(start, start + n)
        before = int(ref.sum())
        data, ref = data[tuple(crop)], ref[tuple(crop)]
        truncated |= int(ref.sum()) < before
        shift = np.eye(4)
        shift[axis, 3] = start
        affine = affine @ shift
    if data.dtype != image.get_data_dtype():  # averaged intensities are stored as integers, like a scanner file
        info = np.iinfo(image.get_data_dtype())
        data = np.clip(np.rint(data), info.min, info.max).astype(image.get_data_dtype())
    return data, ref.astype(np.uint8), affine, truncated


def stress_build(_args):
    """Perturbed copies of the 24 validation patients, one manifest for all conditions, nnU-Net inputs."""
    rows = read_manifest(COHORTS['validation'])
    STRESS.mkdir()
    manifest, records = [], []
    stored, restacked = STRESS / 'nnunet_input/stored', STRESS / 'nnunet_input/restacked'
    stored.mkdir(parents=True)
    restacked.mkdir()
    for number, row in enumerate(sorted(rows, key=lambda r: r['case_id'])):
        pid, side = row['case_id'], ('superior', 'inferior')[number % 2]
        image, reference = nib.load(row['image']), nib.load(row['reference'])
        ref = binary_mask(reference)
        ref_ml = float(ref.sum()) * abs(np.linalg.det(image.affine[:3, :3])) / 1000
        for condition, steps in CONDITIONS.items():
            case_id = f'{condition}__{pid}'
            if steps:
                data, mask, affine, truncated = perturb(image, ref, steps, side)
                directory = STRESS / 'data' / condition / pid
                directory.mkdir(parents=True)
                paths = directory / 'image.nii.gz', directory / 'label.nii.gz'
                save_like(data, affine, image, paths[0])
                save_like(mask, affine, reference, paths[1])
            else:
                paths, truncated = (Path(row['image']), Path(row['reference'])), False
            new_image, new_ref = nib.load(paths[0]), nib.load(paths[1])
            check_grid(new_image, new_ref, case_id)
            mask = binary_mask(new_ref)
            zooms = new_image.header.get_zooms()[:3]
            world = [WORLD[c] for c in nib.aff2axcodes(new_image.affine)]
            manifest.append(dict(case_id=case_id, image=str(paths[0]), reference=str(paths[1]),
                                 image_sha256=digest(paths[0]), reference_sha256=digest(paths[1])))
            records.append(dict(
                condition=condition, case_id=pid, slab_side=side if condition == 'S40edge' else '',
                axcodes=''.join(nib.aff2axcodes(new_image.affine)), shape='x'.join(map(str, new_image.shape)),
                spacing_mm='x'.join(f'{z:.3f}' for z in zooms),
                coverage_si_mm=float(new_image.shape[world.index('SI')] * zooms[world.index('SI')]),
                coverage_ap_mm=float(new_image.shape[world.index('AP')] * zooms[world.index('AP')]),
                ref_ml_original=ref_ml, ref_ml=float(mask.sum()) * float(np.prod(zooms)) / 1000,
                tumour_truncated=int(truncated), tumour_to_edge_mm=edge_distance(mask, zooms)))
            (stored / f'{case_id}_0000.nii.gz').symlink_to(paths[0].resolve())
            if condition in RESTACKED:
                write_training_order(new_image, restacked / f'{case_id}_0000.nii.gz')
            print(records[-1], flush=True)
    write_csv(STRESS / 'manifest.csv', manifest)
    write_csv(STRESS / 'perturbations.csv', records)
    read_manifest(STRESS / 'manifest.csv')  # re-verify every hash and path


def stress_runs():
    """(label, family, seed, run directory, case-id prefixes) for every scored prediction set."""
    sets = [(r['run'], r['family'], r['seed'], r['run'], None) for r in runs()]
    return sets + [('nnunet_s42_restacked', 'nnunet', '42', 'nnunet_s42_restacked', RESTACKED)]


def stress_unit(row, info, sets):
    pid = row['case_id']
    image, reference = nib.load(row['image']), nib.load(row['reference'])
    check_grid(image, reference, pid)
    ref, out = binary_mask(reference), []
    for label, family, seed, directory, only in sets:
        if only and info['condition'] not in only:
            continue
        prediction = nib.load(STRESS / 'predictions' / directory / f'{pid}.nii.gz')
        check_grid(image, prediction, f'{pid} {label}')
        metrics = case_metrics(binary_mask(prediction), ref, image)
        del metrics['empty_status']
        out.append(dict(condition=info['condition'], case_id=info['case_id'], run=label, family=family, seed=seed,
                        **metrics, ref_ml_original=float(info['ref_ml_original']),
                        tumour_to_edge_mm=float(info['tumour_to_edge_mm']),
                        tumour_truncated=int(info['tumour_truncated']),
                        coverage_si_mm=float(info['coverage_si_mm']), coverage_ap_mm=float(info['coverage_ap_mm'])))
    return out


def stress_score(_args):
    """Restore re-stacked nnU-Net masks, then score every prediction on its perturbed grid (common scorer)."""
    import torch
    from concurrent.futures import ProcessPoolExecutor
    torch.set_num_threads(1)
    rows = read_manifest(STRESS / 'manifest.csv')
    restored = STRESS / 'predictions/nnunet_s42_restacked'
    restored.mkdir()
    for row in rows:
        if row['case_id'].split('__')[0] in RESTACKED:
            original = nib.load(row['image'])
            mask, _ = restore_from_training_order(
                nib.load(STRESS / f"predictions_lps/nnunet_s42_restacked/{row['case_id']}.nii.gz"), original)
            save_mask(mask, original, restored / f"{row['case_id']}.nii.gz")
    for label, _, _, directory, only in stress_runs():  # coverage check, as in the scorer
        expected = [r for r in rows if not only or r['case_id'].split('__')[0] in only]
        prediction_paths(expected, STRESS / 'predictions' / directory)
    info = {f"{r['condition']}__{r['case_id']}": r for r in read_rows(STRESS / 'perturbations.csv')}
    with ProcessPoolExecutor(max_workers=8) as pool:
        jobs = [pool.submit(stress_unit, row, info[row['case_id']], stress_runs()) for row in rows]
        records = [r for job in jobs for r in job.result()]
    frozen = {(r['run'], r['case_id']): float(r['dice']) for r in read_rows(RECORD / 'patient_scores.csv')
              if r['cohort'] == 'validation'}
    mismatches = [(r['run'], r['case_id'], r['dice'], frozen[r['run'], r['case_id']]) for r in records
                  if r['condition'] == 'C0' and r['dice'] != frozen[r['run'], r['case_id']]]
    write_csv(STRESS / 'stress_test.csv', records)
    print(f'{len(records)} rows; C0 rows differing from frozen validation Dice: {len(mismatches)}')
    if mismatches:
        raise SystemExit(f'C0 sanity check failed: {mismatches[:10]}')


FAMILIES = {'Slab size': ['C0', 'S50', 'S40', 'S33'], 'Slab position': ['C0', 'S40', 'S40edge'],
            'Slice thickness': ['C0', 'T3', 'T4'], 'Combined': ['C0', 'S40', 'T3', 'S40T3'],
            'Coronal slabs': ['C0', 'AP', 'COR_thin', 'COR_thick']}


def bootstrap_ci(differences, replicates=10000, seed=42):
    """Paired patient percentile bootstrap of a mean difference (same convention as thesis_analysis.py)."""
    differences = np.asarray(differences, float)
    indices = np.random.default_rng(seed).integers(0, len(differences), size=(replicates, len(differences)))
    return np.quantile(differences[indices].mean(axis=1), [.025, .975]).tolist()


def stress_summary(_args):
    """Per condition and pipeline: means, empties, paired bootstrap vs C0; figure stress_test.pdf/.png."""
    rows = read_rows(STRESS / 'stress_test.csv')
    patients = sorted({r['case_id'] for r in rows})
    dice = {(r['condition'], r['run'], r['case_id']): float(r['dice']) for r in rows}
    empty = {(r['condition'], r['run'], r['case_id']): r['pred_empty'] == 'True' for r in rows}
    manual = [r['run'] for r in runs() if r['family'] != 'nnunet']
    pipelines = {'manual': manual, 'nnunet': ['nnunet_s42'], 'nnunet_restacked': ['nnunet_s42_restacked']}

    def per_patient(condition, pipeline):
        return np.array([np.mean([dice[condition, run, p] for run in pipelines[pipeline]]) for p in patients])

    table, comparisons = [], {}
    for condition in CONDITIONS:
        for pipeline, members in pipelines.items():
            if (condition, members[0], patients[0]) not in dice:
                continue
            values, base = per_patient(condition, pipeline), per_patient('C0', 'manual' if pipeline == 'manual' else 'nnunet')
            run_means = [np.mean([dice[condition, run, p] for p in patients]) for run in members]
            n_empty = sum(empty[condition, run, p] for run in members for p in patients)
            entry = dict(condition=condition, pipeline=pipeline, n_patients=len(patients), mean_dice=float(values.mean()),
                         median_dice=float(np.median(values)), min_dice=float(values.min()), max_dice=float(values.max()),
                         run_mean_min=float(min(run_means)), run_mean_max=float(max(run_means)),
                         n_empty=int(n_empty), n_predictions=len(members) * len(patients),
                         n_below_0_1=int(sum(dice[condition, run, p] < .1 for run in members for p in patients)),
                         diff_vs_c0=float((values - base).mean()))
            entry['diff_vs_c0_ci95_low'], entry['diff_vs_c0_ci95_high'] = bootstrap_ci(values - base)
            table.append(entry)
    for a, b in (('S40edge', 'S40'), ('S40T3', 'S40'), ('S40T3', 'T3')):  # questions 2 and 4
        for pipeline in ('manual', 'nnunet'):
            diff = per_patient(a, pipeline) - per_patient(b, pipeline)
            comparisons[f'{pipeline}: {a} - {b}'] = dict(mean=float(diff.mean()), ci95=bootstrap_ci(diff),
                                                       patients_better=int((diff > 0).sum()),
                                                       patients_worse=int((diff < 0).sum()))
    for condition in RESTACKED:  # the direct test: re-stacked minus stored nnU-Net on the same images
        diff = per_patient(condition, 'nnunet_restacked') - per_patient(condition, 'nnunet')
        comparisons[f'{condition}: restacked - stored nnU-Net'] = dict(mean=float(diff.mean()), ci95=bootstrap_ci(diff),
                                                                       patients_better=int((diff > 0).sum()),
                                                                       patients_worse=int((diff < 0).sum()))
    with (STRESS / 'stress_summary.csv').open('x', newline='') as f:  # evaluate.write_csv needs nibabel
        writer = csv.DictWriter(f, fieldnames=list(table[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(table)
    (STRESS / 'stress_bootstrap.json').write_text(json.dumps(dict(
        method='paired patient percentile bootstrap of condition minus C0 (manual: per-patient mean of six runs)',
        replicates=10000, seed=42, paired_comparisons=comparisons), indent=2) + '\n')
    figure(table)


def figure(table):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ink, muted = '#0b0b0b', '#52514e'
    style = {'manual': ('#2a78d6', 'o', 'Manual U-Net (mean, range of 6 runs)'),
             'nnunet': ('#eb6834', 's', 'nnU-Net s42 (as stored)'),
             'nnunet_restacked': ('#1baf7a', 'D', 'nnU-Net s42 (re-stacked)')}
    by = {(t['condition'], t['pipeline']): t for t in table}
    fig, axes = plt.subplots(1, len(FAMILIES), figsize=(15, 3.9), sharey=True,
                             gridspec_kw=dict(width_ratios=[len(c) for c in FAMILIES.values()]))
    for ax, (title, conditions) in zip(axes, FAMILIES.items()):
        x = np.arange(len(conditions))
        for offset, pipeline in zip((-0.12, 0.12, 0.24), style):
            color, marker, label = style[pipeline]
            points = [(i, by[c, pipeline]) for i, c in enumerate(conditions) if (c, pipeline) in by]
            if not points:
                continue
            xs, ts = np.array([i for i, _ in points]) + offset, [t for _, t in points]
            ys = [t['mean_dice'] for t in ts]
            if pipeline == 'manual':
                ax.errorbar(xs, ys, yerr=[[y - t['run_mean_min'] for y, t in zip(ys, ts)],
                                          [t['run_mean_max'] - y for y, t in zip(ys, ts)]],
                            color=color, marker=marker, ms=6, lw=1.5, capsize=3, label=label)
            elif pipeline == 'nnunet':
                ax.plot(xs, ys, color=color, marker=marker, ms=6, lw=1.5, label=label)
            else:
                ax.plot(xs, ys, color=color, marker=marker, ms=6, lw=0, label=label)
            for xi, y, t in zip(xs, ys, ts):
                if t['n_empty']:
                    below = pipeline == 'manual'  # manual labels below the marker, nnU-Net labels above
                    ax.annotate(f"{t['n_empty']}/{t['n_predictions']} empty", (xi, y), xytext=(-5, -6) if below else (5, 6),
                                textcoords='offset points', ha='right' if below else 'left', va='top' if below else 'bottom',
                                fontsize=7, color=muted)
        ax.set_xticks(x, conditions)
        ax.set_title(title, fontsize=10, color=ink, loc='left')
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', color='#e4e3df', lw=0.8)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=muted, labelsize=8)
    axes[0].set_ylabel('Mean Dice (24 validation patients)', color=ink)
    handles, labels = axes[-1].get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: not labels[i].startswith('Manual'))
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc='lower center', ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    for suffix in ('pdf', 'png'):
        fig.savefig(STRESS / f'stress_test.{suffix}', dpi=200)


def stress_bundle(_args):
    """3 validation patients (smallest/median/largest reference) x 11 conditions: image, gt, nnU-Net masks."""
    info = [r for r in read_rows(STRESS / 'perturbations.csv') if r['condition'] == 'C0']
    ordered = sorted(info, key=lambda r: float(r['ref_ml_original']))
    chosen = [ordered[0]['case_id'], ordered[len(ordered) // 2 - 1]['case_id'], ordered[-1]['case_id']]
    rows = {r['case_id']: r for r in read_manifest(STRESS / 'manifest.csv')}
    target = OUT / 'viewer_bundle/data/stress_test'
    target.mkdir()
    for pid in chosen:
        for condition in CONDITIONS:
            row = rows[f'{condition}__{pid}']
            directory = target / f'{pid}_{condition}'
            directory.mkdir()
            links = {'image': row['image'], 'gt': row['reference'],
                     'pred_nnunet_s42': STRESS / f"predictions/nnunet_s42/{row['case_id']}.nii.gz"}
            if condition in RESTACKED:
                links['pred_nnunet_s42_restacked'] = STRESS / f"predictions/nnunet_s42_restacked/{row['case_id']}.nii.gz"
            for name, source in links.items():
                (directory / f'{name}.nii.gz').symlink_to(Path(source).resolve())
    print('Stress-test viewer patients:', chosen)


FOLLOWUP = OUT / 'followup_training'
FOLD = 'Dataset901_T1VSComparison/{}__nnUNetPlans__3d_fullres/fold_0'
FOLLOWUP_RUNS = {  # existing outputs only; no inference
    'plain_s42': ROOT / 'outputs/t1/nnunet' / FOLD.format('nnUNetTrainer'),
    'plain_s43': ROOT / 'outputs/t1/nnunet_slab/plain_s43' / FOLD.format('nnUNetTrainer'),
    'slabthick_s42': ROOT / 'outputs/t1/nnunet_slab/slabthick_s42' / FOLD.format('nnUNetTrainerSlabThick'),
    'slabthick_s43': ROOT / 'outputs/t1/nnunet_slab/slabthick_s43' / FOLD.format('nnUNetTrainerSlabThick'),
}


def parse_training_log(path):
    """Per-epoch values from an nnU-Net 2.8.1 training log; EMA recomputed as nnU-Net does (0.9 old + 0.1 new)."""
    epochs, current, ema = [], None, None
    for line in Path(path).read_text().splitlines():
        text = line.split(': ', 1)[-1].strip()
        if re.fullmatch(r'Epoch \d+', text):
            current = dict(epoch=int(text.split()[1]), new_best=False)
            epochs.append(current)
        elif current is None:
            continue
        elif text.startswith('Current learning rate:'):
            current['lr'] = float(text.split()[-1])
        elif text.startswith(('train_loss', 'val_loss')):
            current[text.split()[0]] = float(text.split()[1])
        elif text.startswith('Pseudo dice'):
            current['pseudo_dice'] = float(text.split('[')[1].split(']')[0])
            ema = current['pseudo_dice'] if ema is None else 0.9 * ema + 0.1 * current['pseudo_dice']
            current['ema_pseudo_dice'] = ema
        elif text.startswith('Epoch time:'):
            current['epoch_time_s'] = float(text.split()[2])
        elif text.startswith('Yayy! New best EMA pseudo Dice:'):
            current['new_best'] = True
            current['logged_best_ema'] = float(text.split()[-1])
    return epochs


def followup_training(_args):
    """Training curves, checkpoint selection and nnU-Net's native validation for the four Dataset901 runs."""
    FOLLOWUP.mkdir()
    curves, summary, validation = [], [], {}
    for run, folder in FOLLOWUP_RUNS.items():
        [log] = sorted(folder.glob('training_log_*.txt'))
        epochs = parse_training_log(log)
        best = [e for e in epochs if e['new_best']]
        mismatch = max(abs(e['ema_pseudo_dice'] - e['logged_best_ema']) for e in best)
        if mismatch > 1.5e-4:  # per-epoch values and the best are logged to 4 decimals
            raise ValueError(f'{run}: recomputed EMA differs from logged best by {mismatch}')
        curves += [dict(run=run, **e) for e in epochs]
        tail = epochs[-50:]
        cases = json.loads((folder / 'validation/summary.json').read_text())['metric_per_case']
        validation[run] = {Path(c['reference_file']).name.removesuffix('.nii.gz'): c['metrics']['1']['Dice'] for c in cases}
        values = np.array(list(validation[run].values()))
        summary.append(dict(run=run, epochs=len(epochs), best_epoch=best[-1]['epoch'],
                            best_ema_pseudo_dice=best[-1]['logged_best_ema'], n_new_best=len(best),
                            last50_train_loss=float(np.mean([e['train_loss'] for e in tail])),
                            last50_val_loss=float(np.mean([e['val_loss'] for e in tail])),
                            last50_pseudo_dice=float(np.mean([e['pseudo_dice'] for e in tail])),
                            median_epoch_time_s=float(np.median([e['epoch_time_s'] for e in epochs])),
                            native_val_mean=float(values.mean()), native_val_median=float(np.median(values)),
                            native_val_min=float(values.min()), native_val_n=len(values)))
    patients = sorted(validation['plain_s42'])
    if any(sorted(v) != patients for v in validation.values()) or len(patients) != 24:
        raise ValueError('Validation patients differ between runs')
    table = [dict(case_id=p, **{run: validation[run][p] for run in FOLLOWUP_RUNS}) for p in patients]
    d = {run: np.array([validation[run][p] for p in patients]) for run in FOLLOWUP_RUNS}
    contrasts = {'augmented arm - plain arm (seed means)': (d['slabthick_s42'] + d['slabthick_s43'] - d['plain_s42'] - d['plain_s43']) / 2,
                 'slabthick_s42 - plain_s42': d['slabthick_s42'] - d['plain_s42'],
                 'slabthick_s43 - plain_s43': d['slabthick_s43'] - d['plain_s43'],
                 'plain_s43 - plain_s42 (seed only)': d['plain_s43'] - d['plain_s42'],
                 'slabthick_s43 - slabthick_s42 (seed only)': d['slabthick_s43'] - d['slabthick_s42']}
    comparisons = {name: dict(mean=float(x.mean()), median=float(np.median(x)), ci95=bootstrap_ci(x),
                              patients_higher=int((x > 0).sum()), patients_lower=int((x < 0).sum()),
                              mean_without_VS_SEG_122=float(np.delete(x, patients.index('VS-SEG-122')).mean()))
                   for name, x in contrasts.items()}
    for name, rows in (('training_curves.csv', curves), ('run_summary.csv', summary), ('native_validation.csv', table)):
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with (FOLLOWUP / name).open('x', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
    (FOLLOWUP / 'native_validation_comparisons.json').write_text(json.dumps(dict(
        note='nnU-Net native validation (checkpoint_best, default test-time mirroring); not the preregistered common-scorer check',
        method='paired patient percentile bootstrap, 10,000 replicates, seed 42', comparisons=comparisons), indent=2) + '\n')
    followup_figure(curves, d, patients)


def followup_figure(curves, d, patients):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ink, muted = '#0b0b0b', '#52514e'
    style = {'plain_s42': ('#2a78d6', '-'), 'plain_s43': ('#2a78d6', '--'),
             'slabthick_s42': ('#eb6834', '-'), 'slabthick_s43': ('#eb6834', '--')}
    label = {'plain_s42': 'Plain s42 (Sept)', 'plain_s43': 'Plain s43',
             'slabthick_s42': 'Slab+thick s42', 'slabthick_s43': 'Slab+thick s43'}
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.9))
    for run, (color, dash) in style.items():
        e = [c for c in curves if c['run'] == run]
        x = [c['epoch'] for c in e]
        for ax, key in zip(axes[:3], ('train_loss', 'val_loss', 'ema_pseudo_dice')):
            ax.plot(x, [c[key] for c in e], color=color, ls=dash, lw=1.2, label=label[run])
        best = [c for c in e if c['new_best']][-1]
        axes[2].plot(best['epoch'], best['ema_pseudo_dice'], marker='o', ms=6, color=color, mec='#fcfcfb', mew=1.5)
    titles = ('Training loss', 'Validation loss (patches)', 'EMA pseudo-Dice (dot = selected checkpoint)')
    for ax, title in zip(axes[:3], titles):
        ax.set_xlabel('Epoch', color=muted)
        ax.set_title(title, fontsize=10, color=ink, loc='left')
    axes[0].set_ylim(-0.97, -0.75)  # first ~20 epochs are off-scale; full values in training_curves.csv
    axes[1].set_ylim(-0.97, -0.75)
    axes[2].set_ylim(0.85, 0.97)
    plain, aug = (d['plain_s42'] + d['plain_s43']) / 2, (d['slabthick_s42'] + d['slabthick_s43']) / 2
    ax = axes[3]
    ax.plot([0.4, 1], [0.4, 1], color='#c3c2b7', lw=1)
    ax.scatter(plain, aug, s=28, color='#eb6834', edgecolor='#fcfcfb', linewidth=1, zorder=3)
    k = patients.index('VS-SEG-122')
    ax.annotate('VS-SEG-122', (plain[k], aug[k]), xytext=(6, -2), textcoords='offset points', fontsize=7, color=muted)
    ax.set(xlim=(0.4, 1), ylim=(0.4, 1))
    ax.set_xlabel('Plain arm (mean of s42, s43)', color=muted)
    ax.set_ylabel('Slab+thick arm', color=muted)
    ax.set_title('Native validation Dice per patient', fontsize=10, color=ink, loc='left')
    for ax in axes:
        ax.grid(color='#e4e3df', lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=muted, labelsize=8)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    for suffix in ('pdf', 'png'):
        fig.savefig(FOLLOWUP / f'followup_training.{suffix}', dpi=200)


MODELS = ('plain_s42', 'plain_s43', 'slabthick_s42', 'slabthick_s43')
ARMS = {'plain_arm': ('plain_s42', 'plain_s43'), 'augmented_arm': ('slabthick_s42', 'slabthick_s43')}
STRESS_LABEL = {'plain_s42': 'nnunet_s42', 'plain_s43': 'nnunet_plain_s43',
                'slabthick_s42': 'nnunet_slabthick_s42', 'slabthick_s43': 'nnunet_slabthick_s43'}
MCRC_FOLLOWUP = OUT / 'mcrc_followup'
# PREREGISTRATION.md strata (acquisition properties only); checked against cases.csv below.
STRATA = {'axial_whole_head': ('021', '028', '044', '063', '082', '085', '099'),
          'axial_slab': ('009', '016', '024', '026', '030', '045', '057', '076', '081', '097', '098'),
          'non_axial': ('012', '100', '043', '060')}
PRIMARY = STRATA['axial_slab'] + ('012', '100')


def write_table(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def arm_values(dice, patients):
    """Per-patient arm Dice = mean of the arm's two seeds."""
    return {arm: np.array([np.mean([dice[m][p] for m in members]) for p in patients]) for arm, members in ARMS.items()}


def followup_stress_score(_args):
    """Step A: stress test (11 conditions) for the three new models; validation check from C0 (= 24 original images)."""
    import torch
    from concurrent.futures import ProcessPoolExecutor
    torch.set_num_threads(1)
    rows = read_manifest(STRESS / 'manifest.csv')
    for model in MODELS[1:]:
        restored = STRESS / f'predictions/{STRESS_LABEL[model]}_restacked'
        restored.mkdir()
        for row in rows:
            if row['case_id'].split('__')[0] in RESTACKED:
                original = nib.load(row['image'])
                mask, _ = restore_from_training_order(
                    nib.load(STRESS / f"predictions_lps/{STRESS_LABEL[model]}_restacked/{row['case_id']}.nii.gz"), original)
                save_mask(mask, original, restored / f"{row['case_id']}.nii.gz")
    sets = []
    for model in MODELS:
        sets += [(model, 'nnunet', model[-2:], STRESS_LABEL[model], None),
                 (model + '_restacked', 'nnunet', model[-2:], STRESS_LABEL[model] + '_restacked', RESTACKED)]
    for _, _, _, directory, only in sets:
        prediction_paths([r for r in rows if not only or r['case_id'].split('__')[0] in only], STRESS / 'predictions' / directory)
    info = {f"{r['condition']}__{r['case_id']}": r for r in read_rows(STRESS / 'perturbations.csv')}
    with ProcessPoolExecutor(max_workers=8) as pool:
        jobs = [pool.submit(stress_unit, row, info[row['case_id']], sets) for row in rows]
        records = [r for job in jobs for r in job.result()]
    write_table(STRESS / 'followup_stress_test.csv', records)
    dice = {}
    for r in records:
        dice.setdefault((r['condition'], r['run']), {})[r['case_id']] = r['dice']
    empty = {}
    for r in records:
        empty[(r['condition'], r['run'])] = empty.get((r['condition'], r['run']), 0) + int(r['pred_empty'])
    patients = sorted({r['case_id'] for r in records})
    summary = []
    for condition in CONDITIONS:
        for variant in ('', '_restacked'):
            if (condition, MODELS[0] + variant) not in dice:
                continue
            per_model = {m: dice[condition, m + variant] for m in MODELS}
            arms = arm_values(per_model, patients)
            row = dict(condition=condition, input='restacked' if variant else 'as_stored')
            for m in MODELS:
                values = np.array([per_model[m][p] for p in patients])
                row.update({f'{m}_mean': float(values.mean()), f'{m}_median': float(np.median(values)),
                            f'{m}_empty': empty[condition, m + variant]})
            for arm, values in arms.items():
                row.update({f'{arm}_mean': float(values.mean()), f'{arm}_median': float(np.median(values))})
            difference = arms['augmented_arm'] - arms['plain_arm']
            row['arm_difference_mean'] = float(difference.mean())
            row['arm_difference_ci95_low'], row['arm_difference_ci95_high'] = bootstrap_ci(difference)
            summary.append(row)
    write_table(STRESS / 'followup_stress_summary.csv', summary)
    # Preregistered validation check (common scorer, no TTA): augmented arm not more than 0.01 below plain arm.
    validation = {m: dice['C0', m] for m in MODELS}
    arms = arm_values(validation, patients)
    frozen = {r['case_id']: float(r['dice']) for r in read_rows(RECORD / 'patient_scores.csv')
              if r['cohort'] == 'validation' and r['run'] == 'nnunet_s42'}
    difference = float(arms['augmented_arm'].mean() - arms['plain_arm'].mean())
    check = dict(rule='augmented arm mean Dice not more than 0.01 below plain arm (PREREGISTRATION.md, secondary)',
                 source='stress-test condition C0 = the 24 original validation images; common scorer; no TTA',
                 n=len(patients), model_means={m: float(np.mean(list(v.values()))) for m, v in validation.items()},
                 plain_arm_mean=float(arms['plain_arm'].mean()), augmented_arm_mean=float(arms['augmented_arm'].mean()),
                 difference=difference, passes=difference >= -0.01,
                 frozen_september_plain_s42_mean=float(np.mean(list(frozen.values()))),
                 note='plain_s42 C0 differs from the frozen September export by one voxel in VS-SEG-122 (see FINDINGS.md)')
    (OUT / 'followup_validation_check.json').write_text(json.dumps(check, indent=2) + '\n')
    print(json.dumps(check, indent=2))


def followup_mcrc(_args):
    """Step B: four models on re-stacked MC-RC inputs, restored to the original grid, common scorer; once."""
    rows = read_manifest(COHORTS['mcrc'])
    acq = {r['case_id']: r for r in read_rows(OUT / 'cases.csv') if r['cohort'] == 'mcrc'}
    for name, ids in STRATA.items():  # strata must match acquisition properties (never scores)
        for short in ids:
            a = acq[f'VS-MC-RC-{short}']
            slab = int(a['n_slices']) * float(a['slice_spacing_mm']) < 60
            expected = 'non_axial' if a['acq_plane'] != 'axial' else ('axial_slab' if slab else 'axial_whole_head')
            if expected != name:
                raise ValueError(f'{short}: stratum {name} does not match acquisition ({expected})')
    if sorted(sum(STRATA.values(), ())) != sorted(r['case_id'][-3:] for r in rows):
        raise ValueError('Strata do not cover the 22 MC-RC cases exactly once')
    scored, earlier = {}, {}
    for model in MODELS:
        restored = MCRC_FOLLOWUP / 'predictions' / model
        restored.mkdir(parents=True)
        for row in rows:
            original = nib.load(row['image'])
            mask, _ = restore_from_training_order(
                nib.load(MCRC_FOLLOWUP / f"predictions_lps/{model}/{row['case_id']}.nii.gz"), original)
            save_mask(mask, original, restored / f"{row['case_id']}.nii.gz")
        records, _ = score(rows, restored)
        scored[model] = {r['case_id']: r for r in records}
    for row in rows:  # sanity: plain s42 re-run vs the 12 Oct re-stacked masks
        pid = row['case_id']
        earlier[pid] = int((binary_mask(nib.load(MCRC_FOLLOWUP / f'predictions/plain_s42/{pid}.nii.gz'))
                            != binary_mask(nib.load(REORIENTED / f'predictions/{pid}.nii.gz'))).sum())
    frozen = {r['case_id']: r for r in read_rows(RECORD / 'patient_scores.csv')
              if r['cohort'] == 'mcrc' and r['run'] == 'nnunet_s42'}
    stratum = {f'VS-MC-RC-{s}': name for name, ids in STRATA.items() for s in ids}
    patients = sorted(scored['plain_s42'])
    dice = {m: {p: scored[m][p]['dice'] for p in patients} for m in MODELS}
    dice['september_frozen'] = {p: float(frozen[p]['dice']) for p in patients}
    empty = {m: {p: scored[m][p]['empty_status'] == 'prediction_empty' for p in patients} for m in MODELS}
    empty['september_frozen'] = {p: frozen[p]['empty_status'] == 'prediction_empty' for p in patients}
    arms = dict(zip(patients, zip(*arm_values(dice, patients).values())))
    per_case = []
    for p in patients:
        row = dict(case_id=p, stratum=stratum[p], primary_slab_case=p[-3:] in PRIMARY,
                   acq_plane=acq[p]['acq_plane'], reference_ml=scored['plain_s42'][p]['reference_ml'])
        for m in ('september_frozen',) + MODELS:
            row[f'dice_{m}'] = dice[m][p]
            row[f'empty_{m}'] = empty[m][p]
        row['dice_plain_arm'], row['dice_augmented_arm'] = arms[p]
        row['arm_difference'] = arms[p][1] - arms[p][0]
        row['plain_s42_voxels_differing_from_12oct_restacked'] = earlier[p]
        per_case.append(row)
    MCRC_FOLLOWUP.joinpath('tables').mkdir()
    write_table(MCRC_FOLLOWUP / 'tables/per_case.csv', per_case)
    subsets = {'primary_13_slab': [p for p in patients if p[-3:] in PRIMARY], 'all_22': patients,
               **{name: [f'VS-MC-RC-{s}' for s in ids] for name, ids in STRATA.items()},
               'axial_only_18': [p for p in patients if stratum[p] != 'non_axial']}
    summary = []
    for subset, ids in subsets.items():
        row = dict(subset=subset, n=len(ids))
        for m in ('september_frozen',) + MODELS:
            values = np.array([dice[m][p] for p in ids])
            row.update({f'{m}_mean': float(values.mean()), f'{m}_median': float(np.median(values)),
                        f'{m}_empty': int(sum(empty[m][p] for p in ids))})
        for arm, members in ARMS.items():
            values = np.array([np.mean([dice[m][p] for m in members]) for p in ids])
            row.update({f'{arm}_mean': float(values.mean()), f'{arm}_median': float(np.median(values)),
                        f'{arm}_empty': float(np.mean([sum(empty[m][p] for p in ids) for m in members]))})
        difference = np.array([np.mean([dice[m][p] for m in ARMS['augmented_arm']]) -
                               np.mean([dice[m][p] for m in ARMS['plain_arm']]) for p in ids])
        row['arm_difference_mean'] = float(difference.mean())
        if subset == 'primary_13_slab':  # the preregistered inferential test is on the primary endpoint only
            low, high = bootstrap_ci(difference)
            row.update(arm_difference_ci95_low=low, arm_difference_ci95_high=high)
            fewer_or_equal = row['augmented_arm_empty'] <= row['plain_arm_empty']
            row['preregistered_classification'] = (
                'inconclusive' if low <= 0 <= high else
                'beneficial' if low > 0 and fewer_or_equal else
                'not covered by rule: CI excludes 0 ' + ('below' if high < 0 else 'above, but more empty predictions'))
        summary.append(row)
    write_table(MCRC_FOLLOWUP / 'tables/summary.csv', summary)
    print(json.dumps(summary[0], indent=2), '\nplain_s42 voxels differing from 12 Oct re-stacked:', earlier)


# Figure helper analysis/make_case_figure.py; followup-figures reuses its case-figure style unchanged.
CONTEXT = ROOT / 'analysis'
# Rows requested 14 Oct 2026. Slices are the author's chosen_slice in context/case_notes.md (zero-based canonical);
# 016 has none recorded, so blank = maximum GT area, as in make_case_figure.py.
FIGURE_ROWS = (('VS-MC-RC-098', '7'), ('VS-MC-RC-081', '6'), ('VS-MC-RC-026', '8'), ('VS-MC-RC-057', '8'),
               ('VS-MC-RC-016', ''))
FIGURE_MODELS = {'plain_s42': 'plain s42\n(re-stacked)', 'slabthick_s42': 'slabthick s42'}


def followup_bundle(_args):
    """Add the four follow-up models' restored MC-RC masks (re-stacked inputs) to viewer_bundle/data/mcrc."""
    directories = sorted((OUT / 'viewer_bundle/data/mcrc').iterdir())
    if len(directories) != 22:
        raise ValueError(f'Expected 22 MC-RC bundle folders, found {len(directories)}')
    for directory in directories:
        for model in MODELS:  # symlink_to refuses to replace an existing link
            source = MCRC_FOLLOWUP / f'predictions/{model}/{directory.name}.nii.gz'
            (directory / f'pred_{model}.nii.gz').symlink_to(source.resolve(strict=True))
    print(f'{len(directories) * len(MODELS)} symlinks added under {OUT / "viewer_bundle/data/mcrc"}')


def spread(values, gap, top):
    """Label positions >= gap apart, order kept, capped at top (for direct labels on the slope plot)."""
    order = np.argsort(values, kind='stable')
    pos = np.asarray(values, float)[order]
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + gap)
    pos[-1] = min(pos[-1], top)
    for i in range(len(pos) - 2, -1, -1):
        pos[i] = min(pos[i], pos[i + 1] - gap)
    result = np.empty_like(pos)
    result[order] = pos
    return result


def followup_figures(_args):
    """Case figure (make_case_figure.py style; Dice from per_case.csv) and slope plot of the 13 slab cases."""
    import importlib.util
    import shutil
    import tempfile
    spec = importlib.util.spec_from_file_location('make_case_figure', CONTEXT / 'make_case_figure.py')
    mcf = sys.modules[spec.name] = importlib.util.module_from_spec(spec)  # its dataclass needs the module registered
    spec.loader.exec_module(mcf)
    plt = mcf.plt
    table = {r['case_id']: r for r in read_rows(MCRC_FOLLOWUP / 'tables/per_case.csv')}
    target = MCRC_FOLLOWUP / 'figures'
    target.mkdir()

    class Reports:  # make_case_figure.Reports interface, reading the step-B per-case table
        @staticmethod
        def metric(_cohort, case, model):
            return table[case][f'dice_{model}'], table[case][f'empty_{model}'] == 'True', 'per_case.csv'

    def legend(fig):
        original_legend(fig)
        fig.texts[-1].set_text('Dice source: mcrc_followup/tables/per_case.csv\n'
                               'both models on re-stacked inputs; masks on the original grid')

    original_legend, mcf.legend, mcf.model_name = mcf.legend, legend, FIGURE_MODELS.get
    cases = []
    for case_id, z in FIGURE_ROWS:
        stratum = table[case_id]['stratum'].replace('_', ' ').capitalize()
        cases.append(mcf.Case(mcf.Spec(case_id, z, mcf.DEFAULT_CROP_MM, list(FIGURE_MODELS), stratum),
                              OUT / 'viewer_bundle', Reports))
    with tempfile.TemporaryDirectory(dir=MCRC_FOLLOWUP) as tmp:
        mcf.render_final(cases, Path(tmp))
        for suffix in ('pdf', 'png'):
            shutil.move(Path(tmp) / f'cases_figure.{suffix}', target / f'cases_followup.{suffix}')
    for case in cases:
        for note in sorted(case.notes):
            print(f'NOTE {case.spec.case_id}: {note}')

    ids = [p for p in sorted(table) if table[p]['primary_slab_case'] == 'True']
    plain = np.array([float(table[p]['dice_plain_arm']) for p in ids])
    augmented = np.array([float(table[p]['dice_augmented_arm']) for p in ids])
    fig, ax = plt.subplots(figsize=(8 * mcf.CM, 9.5 * mcf.CM))
    fig.subplots_adjust(left=.15, right=.97, top=.92, bottom=.14)
    for a, b in zip(plain, augmented):
        ax.plot([0, 1], [a, b], color='#8c8c8c', lw=1, marker='o', ms=3, zorder=3)
    ax.plot([0, 1], [plain.mean(), augmented.mean()], color='black', lw=2, marker='D', ms=5, zorder=2)
    for x, values, side in ((0, plain, -1), (1, augmented, 1)):
        for value, label, short in zip(values, spread(values, .045, 1.0), ids):
            ax.plot([x + side * .03, x + side * .12], [value, label], color='#bdbdbd', lw=.5, zorder=1)
            ax.text(x + side * .14, label, short[-3:], va='center', ha='left' if side > 0 else 'right', fontsize=7)
    ax.set_xticks([0, 1], [f'Plain arm\nmean {plain.mean():.3f}', f'Augmented arm\nmean {augmented.mean():.3f}'])
    ax.set_xlim(-.5, 1.5)
    ax.set_ylim(-.04, 1.04)
    ax.set_ylabel('Dice per patient (mean of seeds 42 and 43)')
    ax.set_title(f'MC-RC slab cases (primary endpoint, n = {len(ids)})')
    ax.grid(axis='y', color='#e5e5e5', lw=.5)
    ax.set_axisbelow(True)
    for side in ('top', 'right', 'bottom'):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis='x', length=0)
    fig.text(.5, .01, 'Black: arm mean. Source: mcrc_followup/tables/per_case.csv', ha='center', va='bottom', fontsize=7)
    for suffix in ('pdf', 'png'):
        fig.savefig(target / f'slab_slope.{suffix}', dpi=300)
    plt.close(fig)
    print('Saved', *sorted(p.name for p in target.iterdir()))


POSTHOC = MCRC_FOLLOWUP / 'posthoc'
CANAL_PROXY_MM = 20.0  # FP centroid this close to the mirrored reference centroid = "near contralateral canal" (proxy)
SEED_GAP, UNCHANGED = 0.3, 0.05


def followup_posthoc(_args):
    """Post hoc / exploratory, NOT preregistered: false positives and side, seed stability, robustness of the primary
    difference, gain versus acquisition, remaining failures. Existing masks only (CPU, no inference)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy import ndimage, stats
    POSTHOC.mkdir()
    structure = np.ones((3, 3, 3))  # components as in case_metrics() (26-connectivity; FP = no overlap with reference)
    per_case = {r['case_id']: r for r in read_rows(MCRC_FOLLOWUP / 'tables/per_case.csv')}
    stress = {(r['run'], r['case_id']): r for r in read_rows(STRESS / 'followup_stress_test.csv') if r['condition'] == 'C0'}
    units = [('mcrc', row, {m: MCRC_FOLLOWUP / f"predictions/{m}/{row['case_id']}.nii.gz" for m in MODELS})
             for row in read_manifest(COHORTS['mcrc'])]
    units += [('validation', row, {m: STRESS / f"predictions/{STRESS_LABEL[m]}/{row['case_id']}.nii.gz" for m in MODELS})
              for row in read_manifest(STRESS / 'manifest.csv') if row['case_id'].startswith('C0__')]

    def otsu(values):
        """Otsu threshold on a 256-bin histogram (intensities clipped at the 99.5th percentile)."""
        values = np.minimum(values, np.percentile(values, 99.5))
        hist, edges = np.histogram(values, bins=256)
        centres = (edges[:-1] + edges[1:]) / 2
        w0, s0 = np.cumsum(hist), np.cumsum(hist * centres)
        w1, s1 = w0[-1] - w0, s0[-1] - s0
        between = w0 * w1 * (s0 / np.maximum(w0, 1) - s1 / np.maximum(w1, 1)) ** 2
        return float(edges[int(np.argmax(between[:-1])) + 1])

    components, cases_out, seed_overlap = [], [], {}
    for cohort, row, paths in units:
        pid = row['case_id'].replace('C0__', '')
        image, reference = nib.load(row['image']), nib.load(row['reference'])
        check_grid(image, reference, pid)
        ref, affine = binary_mask(reference), image.affine
        data = np.asanyarray(image.dataobj).astype(np.float32)
        threshold = otsu(data.ravel())
        labels, n = ndimage.label(data > threshold, structure=structure)
        head = labels == int(np.argmax(np.bincount(labels.ravel())[1:])) + 1
        x = nib.affines.apply_affine(affine, np.argwhere(head))[:, 0]  # RAS+ world x: + = patient right
        midline = float((x.min() + x.max()) / 2)  # left-right centre of the head mask
        ref_index = np.argwhere(ref).mean(axis=0)
        ref_centre = nib.affines.apply_affine(affine, ref_index)
        mirrored = ref_centre * [-1, 1, 1] + [2 * midline, 0, 0]
        tumour_offset = float(ref_centre[0] - midline)
        stratum = next((s for s, ids in STRATA.items() if pid[-3:] in ids), None) if cohort == 'mcrc' else 'validation'
        base = dict(cohort=cohort, case_id=pid, stratum=stratum, primary_slab_case=cohort == 'mcrc' and pid[-3:] in PRIMARY)
        geometry = dict(midline_x_mm=midline, head_centroid_x_mm=float(x.mean()), head_lr_extent_mm=float(x.max() - x.min()),
                        otsu_threshold=threshold, tumour_side='right' if tumour_offset > 0 else 'left',
                        tumour_offset_from_midline_mm=tumour_offset, tumour_centroid_slice=int(round(ref_index[2])) + 1,
                        n_slices=int(ref.shape[2]))
        voxel_ml = abs(float(np.linalg.det(affine[:3, :3]))) / 1000
        masks = {}
        for model in MODELS:
            prediction = nib.load(paths[model])
            check_grid(image, prediction, f'{pid} {model}')
            pred = binary_mask(prediction)
            masks[model] = pred
            m = patient_metrics(pred, ref, affine)
            labels, n = ndimage.label(pred, structure=structure)
            sizes = np.bincount(labels.ravel(), minlength=n + 1)
            fp = np.setdiff1d(np.arange(1, n + 1), np.unique(labels[ref & pred]))
            fp = fp[np.argsort(-sizes[fp], kind='stable')]  # largest first
            boxes = ndimage.find_objects(labels)
            rows = []
            for rank, label in enumerate(fp, 1):
                index = np.array(ndimage.center_of_mass(pred, labels, int(label)))
                centre = nib.affines.apply_affine(affine, index)
                offset = float(centre[0] - midline)
                distance_mirrored = float(np.linalg.norm(centre - mirrored))
                rows.append(dict(base, model=model, arm=model.split('_')[0], seed=int(model[-2:]), component=rank,
                                 n_voxels=int(sizes[label]), volume_ml=float(sizes[label] * voxel_ml),
                                 centroid_x_mm=float(centre[0]), centroid_y_mm=float(centre[1]), centroid_z_mm=float(centre[2]),
                                 centroid_voxel_i=float(index[0]), centroid_voxel_j=float(index[1]),
                                 centroid_voxel_k=float(index[2]), viewer_slice=int(round(index[2])) + 1,
                                 viewer_slice_first=boxes[label - 1][2].start + 1, viewer_slice_last=boxes[label - 1][2].stop,
                                 dist_to_ref_centroid_mm=float(np.linalg.norm(centre - ref_centre)),
                                 midline_x_mm=midline, fp_offset_from_midline_mm=offset,
                                 tumour_offset_from_midline_mm=tumour_offset,
                                 side='ipsilateral' if offset * tumour_offset > 0 else 'contralateral',
                                 dist_to_mirrored_ref_centroid_mm=distance_mirrored,
                                 near_contralateral_canal_proxy=distance_mirrored <= CANAL_PROXY_MM))
            components += rows
            contra = [r for r in rows if r['side'] == 'contralateral']
            cases_out.append(dict(base, model=model, dice=m['dice'], pred_empty=m['prediction_voxels'] == 0,
                                  pred_ml=m['prediction_ml'], reference_ml=m['reference_ml'], n_pred_components=int(n),
                                  n_fp_components=len(rows), fp_ml=float(sum(r['volume_ml'] for r in rows)),
                                  n_contralateral_fp=len(contra), contralateral_fp_ml=float(sum(r['volume_ml'] for r in contra)),
                                  n_near_contralateral_canal_fp=sum(r['near_contralateral_canal_proxy'] for r in rows),
                                  largest_fp_ml=rows[0]['volume_ml'] if rows else '',
                                  largest_fp_side=rows[0]['side'] if rows else '',
                                  largest_fp_viewer_slice=rows[0]['viewer_slice'] if rows else '', **geometry))
            # Same masks and component definition as the scored tables: Dice and FP counts must match exactly.
            if cohort == 'mcrc':
                expected = [(float(per_case[pid][f'dice_{model}']), m['dice'])]
            else:
                s = stress[model, pid]
                expected = [(float(s['dice']), m['dice']), (int(s['n_fp_components']), len(rows)),
                             (float(s['fp_ml']), float(sizes[fp].sum() * voxel_ml))]
            if any(a != b for a, b in expected):
                raise SystemExit(f'{pid} {model}: recomputed values differ from the scored table {expected}')
        for arm, (a, b) in ARMS.items():  # Dice between the two seeds' masks; undefined when both are empty
            both_empty = not (masks[a].any() or masks[b].any())
            seed_overlap[cohort, pid, arm] = '' if both_empty else patient_metrics(masks[a], masks[b], affine)['dice']
        print(pid, f'midline {midline:.1f} mm, tumour {geometry["tumour_side"]}', flush=True)
    write_table(POSTHOC / 'fp_components.csv', components)
    write_table(POSTHOC / 'fp_per_case.csv', cases_out)
    rows_by = {(r['cohort'], r['case_id'], r['model']): r for r in cases_out}

    subsets = {('mcrc', 'all_22'): None, ('mcrc', 'primary_13_slab'): 'primary',
               **{('mcrc', s): s for s in STRATA}, ('validation', 'validation_24'): 'validation'}

    def members(cohort, key):
        ids = sorted({r['case_id'] for r in cases_out if r['cohort'] == cohort})
        if key == 'primary':
            return [p for p in ids if p[-3:] in PRIMARY]
        return [p for p in ids if key in (None, 'validation') or rows_by[cohort, p, MODELS[0]]['stratum'] == key]

    # B. FP summary per model and arm (arm counts = mean of the two seeds, as for empty predictions in summary.csv)
    original = [r for r in read_rows(OUT / 'cases.csv')]
    fp_summary = []
    for (cohort, subset), key in subsets.items():
        ids = members(cohort, key)
        for unit, models in [(m, (m,)) for m in MODELS] + [(a.replace('_arm', ''), ms) for a, ms in ARMS.items()]:
            per_model = [[rows_by[cohort, p, m] for p in ids] for m in models]
            sizes = [c['volume_ml'] for c in components if c['cohort'] == cohort and c['case_id'] in ids
                     and c['model'] in models]
            count = lambda rule: float(np.mean([sum(rule(r) for r in rs) for rs in per_model]))  # noqa: E731
            fp_summary.append(dict(
                source='posthoc (re-stacked follow-up masks)', cohort=cohort, subset=subset, unit=unit,
                n_cases=len(ids), cases_with_fp=count(lambda r: r['n_fp_components'] > 0),
                cases_with_contralateral_fp=count(lambda r: r['n_contralateral_fp'] > 0),
                cases_with_near_contralateral_canal_fp_proxy=count(lambda r: r['n_near_contralateral_canal_fp'] > 0),
                cases_with_fp_either_seed=sum(any(rs[i]['n_fp_components'] > 0 for rs in per_model) for i in range(len(ids))),
                cases_with_contralateral_fp_either_seed=sum(any(rs[i]['n_contralateral_fp'] > 0 for rs in per_model)
                                                            for i in range(len(ids))),
                n_fp_components=count(lambda r: r['n_fp_components']),
                fp_component_ml_median=float(np.median(sizes)) if sizes else '',
                fp_component_ml_max=float(max(sizes)) if sizes else ''))
        for run in [r['run'] for r in runs()]:  # the original 7 models, same component definition (cases.csv)
            rs = [r for r in original if r['cohort'] == cohort and r['run'] == run and r['case_id'] in ids]
            fp_summary.append(dict(source='cases.csv (original 7 models, stored inputs)', cohort=cohort, subset=subset,
                                   unit=run, n_cases=len(rs), cases_with_fp=sum(int(r['n_fp_components']) > 0 for r in rs),
                                   n_fp_components=sum(int(r['n_fp_components']) for r in rs)))
    write_table(POSTHOC / 'fp_summary.csv', fp_summary)

    # B. Contralateral FPs: plain versus augmented, per case (MC-RC and validation)
    contralateral = []
    for cohort, pid in sorted({(r['cohort'], r['case_id']) for r in cases_out}):
        rs = {m: rows_by[cohort, pid, m] for m in MODELS}
        if not any(r['n_contralateral_fp'] for r in rs.values()):
            continue
        arm_has = {arm: any(rs[m]['n_contralateral_fp'] for m in ms) for arm, ms in ARMS.items()}
        row = dict(cohort=cohort, case_id=pid, stratum=rs[MODELS[0]]['stratum'],
                   tumour_side=rs[MODELS[0]]['tumour_side'], reference_ml=rs[MODELS[0]]['reference_ml'])
        for m in MODELS:
            row.update({f'{m}_dice': rs[m]['dice'], f'{m}_n_contralateral_fp': rs[m]['n_contralateral_fp'],
                        f'{m}_contralateral_fp_ml': rs[m]['contralateral_fp_ml'],
                        f'{m}_near_canal_proxy': rs[m]['n_near_contralateral_canal_fp']})
        row['contralateral_fp_in'] = ('both arms' if all(arm_has.values()) else
                                      'augmented arm only' if arm_has['augmented_arm'] else 'plain arm only')
        contralateral.append(row)
    write_table(POSTHOC / 'fp_contralateral_by_case.csv', contralateral)

    # C. Seed stability per case and arm
    seed_rows = []
    for cohort, pid in sorted({(r['cohort'], r['case_id']) for r in cases_out}):
        for arm, (a, b) in ARMS.items():
            ra, rb = rows_by[cohort, pid, a], rows_by[cohort, pid, b]
            empty = {(False, False): 'neither', (True, True): 'both'}.get((ra['pred_empty'], rb['pred_empty']),
                                                                          'one (s42)' if ra['pred_empty'] else 'one (s43)')
            seed_rows.append(dict(cohort=cohort, case_id=pid, stratum=ra['stratum'], primary_slab_case=ra['primary_slab_case'],
                                  arm=arm, dice_s42=ra['dice'], dice_s43=rb['dice'], abs_dice_difference=abs(ra['dice'] - rb['dice']),
                                  mask_dice_s42_vs_s43=seed_overlap[cohort, pid, arm],
                                  empty_predictions=empty))
    write_table(POSTHOC / 'seed_stability.csv', seed_rows)
    seed_summary = []
    for (cohort, subset), key in subsets.items():
        ids = members(cohort, key)
        for arm in ARMS:
            rs = [r for r in seed_rows if r['cohort'] == cohort and r['arm'] == arm and r['case_id'] in ids]
            gaps = np.array([r['abs_dice_difference'] for r in rs])
            overlap = [r['mask_dice_s42_vs_s43'] for r in rs if r['mask_dice_s42_vs_s43'] != '']
            seed_summary.append(dict(cohort=cohort, subset=subset, arm=arm, n_cases=len(rs),
                                     mean_abs_dice_difference=float(gaps.mean()), median_abs_dice_difference=float(np.median(gaps)),
                                     max_abs_dice_difference=float(gaps.max()), cases_gap_above_0_3=int((gaps > SEED_GAP).sum()),
                                     mean_mask_dice_s42_vs_s43=float(np.mean(overlap)), n_mask_dice=len(overlap),
                                     cases_one_seed_empty=sum(r['empty_predictions'].startswith('one') for r in rs),
                                     cases_both_seeds_empty=sum(r['empty_predictions'] == 'both' for r in rs)))
    write_table(POSTHOC / 'seed_stability_summary.csv', seed_summary)

    # D. Robustness of the primary difference (13 slab cases), post hoc
    ids = sorted(p for p in per_case if per_case[p]['primary_slab_case'] == 'True')
    dice = {m: np.array([float(per_case[p][f'dice_{m}']) for p in ids]) for m in MODELS}
    difference = np.array([float(per_case[p]['arm_difference']) for p in ids])
    primary = next(r for r in read_rows(MCRC_FOLLOWUP / 'tables/summary.csv') if r['subset'] == 'primary_13_slab')
    ci = bootstrap_ci(difference)
    per_seed = {}
    for seed in ('42', '43'):
        d = dice[f'slabthick_s{seed}'] - dice[f'plain_s{seed}']
        per_seed[f's{seed}'] = dict(contrast=f'slabthick_s{seed} - plain_s{seed}', mean_difference=float(d.mean()),
                                    ci95=bootstrap_ci(d), plain_mean=float(dice[f'plain_s{seed}'].mean()),
                                    augmented_mean=float(dice[f'slabthick_s{seed}'].mean()),
                                    cases_improved=int((d >= UNCHANGED).sum()), cases_worse=int((d <= -UNCHANGED).sum()))
    loo = {p: float(np.delete(difference, i).mean()) for i, p in enumerate(ids)}
    robustness = dict(
        label='POST HOC, not preregistered. Primary endpoint itself: mcrc_followup/tables/summary.csv, row primary_13_slab.',
        n=len(ids), cases=ids, bootstrap=dict(replicates=10000, seed=42, method=bootstrap_ci.__doc__),
        primary_reproduced=dict(arm_difference_mean=float(difference.mean()), ci95=ci,
                                matches_summary_csv=bool(np.isclose(difference.mean(), float(primary['arm_difference_mean']))
                                                         and np.allclose(ci, [float(primary['arm_difference_ci95_low']),
                                                                              float(primary['arm_difference_ci95_high'])]))),
        per_seed_paired_difference=per_seed,
        direction_by_arm_mean=dict(threshold=f'|arm_difference| < {UNCHANGED} = unchanged',
                                   improved=[p for p, d in zip(ids, difference) if d >= UNCHANGED],
                                   unchanged=[p for p, d in zip(ids, difference) if abs(d) < UNCHANGED],
                                   worse=[p for p, d in zip(ids, difference) if d <= -UNCHANGED]),
        leave_one_out=dict(arm_difference_mean_min=min(loo.values()), case_left_out_at_min=min(loo, key=loo.get),
                           arm_difference_mean_max=max(loo.values()), case_left_out_at_max=max(loo, key=loo.get),
                           per_case_left_out=loo))
    for k in ('improved', 'unchanged', 'worse'):
        robustness['direction_by_arm_mean'][f'n_{k}'] = len(robustness['direction_by_arm_mean'][k])
    (POSTHOC / 'primary_robustness.json').write_text(json.dumps(robustness, indent=2) + '\n')

    # E. Gain versus acquisition (exploratory, n = 13)
    acquisition_rows = {}
    for r in original:
        if r['cohort'] == 'mcrc':
            fields = ('acq_plane', 'n_slices', 'slice_spacing_mm', 'slice_thickness', 'tumor_to_fov_edge_mm')
            if acquisition_rows.setdefault(r['case_id'], {k: r[k] for k in fields}) != {k: r[k] for k in fields}:
                raise ValueError(f"{r['case_id']}: acquisition columns differ between runs in cases.csv")
    gain = []
    for p in ids:
        a = acquisition_rows[p]
        gain.append(dict(case_id=p, stratum=per_case[p]['stratum'], acq_plane=a['acq_plane'],
                         arm_difference=float(per_case[p]['arm_difference']),
                         dice_plain_arm=float(per_case[p]['dice_plain_arm']),
                         dice_augmented_arm=float(per_case[p]['dice_augmented_arm']),
                         coverage_mm=int(a['n_slices']) * float(a['slice_spacing_mm']), n_slices=int(a['n_slices']),
                         slice_thickness=float(a['slice_thickness']), slice_spacing_mm=float(a['slice_spacing_mm']),
                         reference_ml=float(per_case[p]['reference_ml']),
                         tumor_to_fov_edge_mm=float(a['tumor_to_fov_edge_mm'])))
    write_table(POSTHOC / 'gain_vs_acquisition_cases.csv', gain)
    sources = {'coverage_mm': 'cases.csv: n_slices x slice_spacing_mm', 'slice_thickness': 'cases.csv: slice_thickness',
               'slice_spacing_mm': 'cases.csv: slice_spacing_mm', 'reference_ml': 'per_case.csv: reference_ml',
               'tumor_to_fov_edge_mm': 'cases.csv: tumor_to_fov_edge_mm'}
    write_table(POSTHOC / 'gain_vs_acquisition.csv', [
        dict(variable=v, n=len(gain), spearman_rho=float(stats.spearmanr([g[v] for g in gain],
                                                                          [g['arm_difference'] for g in gain])[0]),
             outcome='per_case.csv: arm_difference', source=s,
             note='EXPLORATORY, post hoc; n = 13 is too small for firm conclusions; no p-values computed')
        for v, s in sources.items()])
    cm = 1 / 2.54
    fig, ax = plt.subplots(figsize=(9 * cm, 7.5 * cm))
    fig.subplots_adjust(left=.16, right=.97, top=.9, bottom=.24)
    ax.axhline(0, color='#bdbdbd', lw=.8, ls='--', zorder=1)
    for plane, marker, face in (('axial', 'o', '#6e6e6e'), ('coronal', 's', 'white')):
        g = [r for r in gain if r['acq_plane'] == plane]
        ax.scatter([r['coverage_mm'] for r in g], [r['arm_difference'] for r in g], s=22, marker=marker, facecolors=face,
                   edgecolors='black', linewidths=.7, zorder=3, label=f'{plane} slab (n = {len(g)})')
    for r in gain:
        ax.annotate(r['case_id'][-3:], (r['coverage_mm'], r['arm_difference']),
                    xytext=(3, -9) if r['case_id'].endswith('016') else (3, 3),  # 016 and 100 share a point
                    textcoords='offset points', fontsize=6.5)
    ax.set_xlabel('Coverage along slice direction (mm)', fontsize=8)
    ax.set_ylabel('Dice gain (augmented - plain arm)', fontsize=8)
    ax.set_title('MC-RC slab cases: gain vs coverage (post hoc, n = 13)', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.5, frameon=False, loc='upper left')
    ax.margins(x=.08)
    ax.grid(color='#e5e5e5', lw=.5)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    fig.text(.5, .01, 'Gain: mcrc_followup/tables/per_case.csv, arm_difference\n'
             'Coverage: cases.csv, n_slices x slice_spacing_mm', ha='center', va='bottom', fontsize=5.5)
    for suffix in ('pdf', 'png'):
        fig.savefig(POSTHOC / f'gain_vs_acquisition.{suffix}', dpi=300)
    plt.close(fig)

    # F. Remaining failures (MC-RC)
    expected = ('060', '016', '012', '076', '045', '026', '030')  # author-listed cases, checked, not assumed
    failures = []
    for p in sorted(per_case):
        c, a = per_case[p], acquisition_rows[p]
        rs = {m: rows_by['mcrc', p, m] for m in MODELS}
        augmented, gap = float(c['dice_augmented_arm']), abs(float(c['dice_slabthick_s42']) - float(c['dice_slabthick_s43']))
        criteria = [name for name, hit in (
            ('augmented_dice_below_0.7', augmented < 0.7), ('augmented_seed_gap_above_0.3', gap > SEED_GAP),
            ('arm_difference_below_0', float(c['arm_difference']) < 0),
            ('contralateral_fp_augmented', any(rs[m]['n_contralateral_fp'] for m in ARMS['augmented_arm']))) if hit]
        if not criteria and p[-3:] not in expected:
            continue
        summary = '; '.join(
            f"{m}: {r['n_fp_components']} FP {r['fp_ml']:.3f} ml ({r['n_contralateral_fp']} contralateral, "
            f"{r['n_near_contralateral_canal_fp']} near-canal proxy; largest {r['largest_fp_ml']:.3f} ml "
            f"{r['largest_fp_side']}, slice {r['largest_fp_viewer_slice']})" for m, r in rs.items() if r['n_fp_components'])
        failures.append(dict(
            case_id=p, criteria_met=' '.join(criteria) or 'none (author-listed case only)',
            author_listed=p[-3:] in expected, negligible_difference=abs(float(c['arm_difference'])) < UNCHANGED,
            stratum=c['stratum'], acq_plane=c['acq_plane'], coverage_mm=int(a['n_slices']) * float(a['slice_spacing_mm']),
            slice_thickness=float(a['slice_thickness']), reference_ml=float(c['reference_ml']),
            dice_september_frozen=float(c['dice_september_frozen']),
            **{f'dice_{m}': float(c[f'dice_{m}']) for m in MODELS},
            **{f'empty_{m}': rs[m]['pred_empty'] for m in MODELS},
            dice_plain_arm=float(c['dice_plain_arm']), dice_augmented_arm=augmented,
            arm_difference=float(c['arm_difference']), augmented_seed_gap=gap,
            tumour_side=rs[MODELS[0]]['tumour_side'], tumour_centroid_viewer_slice=rs[MODELS[0]]['tumour_centroid_slice'],
            n_slices=rs[MODELS[0]]['n_slices'], fp_summary=summary or 'no FP components'))
    write_table(POSTHOC / 'remaining_failures.csv', failures)
    print(f'{len(components)} FP components; {len(failures)} remaining-failure rows; outputs in {POSTHOC}')


# Post hoc figure requested 17 Oct 2026 after the author's viewer check (slabthick s42): whole slices, so the
# contralateral spot is visible. plain s43 has the same spots in 030, 076 and 016. 060 uses the author's chosen_slice.
FP_FIGURE_ROWS = ('VS-MC-RC-030', 'VS-MC-RC-076', 'VS-MC-RC-016', 'VS-MC-RC-060')
FP_FIGURE_MODELS = {'plain_s43': 'plain s43', 'slabthick_s42': 'slabthick s42'}


def followup_fp_figure(_args):
    """Post hoc case figure (make_case_figure.py style): image | plain s43 | slabthick s42, whole canonical slices."""
    import importlib.util
    import shutil
    import tempfile
    from scipy import ndimage
    spec = importlib.util.spec_from_file_location('make_case_figure', CONTEXT / 'make_case_figure.py')
    mcf = sys.modules[spec.name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mcf)
    targets = [POSTHOC / f'cases_contralateral.{suffix}' for suffix in ('pdf', 'png')]
    if any(t.exists() for t in targets):
        raise SystemExit(f'Refusing to overwrite {targets}')
    table = {r['case_id']: r for r in read_rows(MCRC_FOLLOWUP / 'tables/per_case.csv')}
    chosen = {'VS-MC-RC-060': '117'}  # case_notes.md chosen_slice (zero-based canonical)

    class Reports:
        @staticmethod
        def metric(_cohort, case, model):
            return table[case][f'dice_{model}'], table[case][f'empty_{model}'] == 'True', 'per_case.csv'

    def legend(fig):
        original_legend(fig)
        fig.texts[-1].set_text('Dice source: mcrc_followup/tables/per_case.csv; re-stacked inputs, masks on the\n'
                               'original grid; whole canonical axial slices (patient right on image left)')

    original_legend, mcf.legend, mcf.model_name = mcf.legend, legend, FP_FIGURE_MODELS.get
    cases = []
    for case_id in FP_FIGURE_ROWS:
        title = 'Sagittal 3D' if table[case_id]['acq_plane'] == 'sagittal' else 'Axial slab'
        case = mcf.Case(mcf.Spec(case_id, chosen.get(case_id, '0'), None, list(FP_FIGURE_MODELS), title),
                        OUT / 'viewer_bundle', Reports)
        if case_id not in chosen:  # slice with an FP component of both models, most reference voxels
            both = np.ones(case.image.shape[2], bool)
            for model in FP_FIGURE_MODELS:
                labels, _ = ndimage.label(case.pred[model], structure=np.ones((3, 3, 3)))
                fp = np.isin(labels, np.setdiff1d(np.unique(labels), np.unique(labels[case.gt])))
                both &= fp.any(axis=(0, 1))
            if not both.any():
                raise ValueError(f'{case_id}: no slice with an FP component in both models')
            case.spec.slice = str(int(np.argmax(np.where(both, case.gt_area + 1, 0))))
        cases.append(case)
    with tempfile.TemporaryDirectory(dir=POSTHOC) as tmp:
        mcf.render_final(cases, Path(tmp))
        for suffix, target in zip(('pdf', 'png'), targets):
            shutil.move(Path(tmp) / f'cases_figure.{suffix}', target)
    for case in cases:
        print(f'{case.spec.case_id}: canonical slice {case.spec.slice} (viewer {int(case.spec.slice) + 1})',
              *sorted(case.notes), sep='\n  ')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['orientation', 'reorient-stage', 'reorient-restore', 'cases', 'bundle',
                                            'stress-build', 'stress-score', 'stress-summary', 'stress-bundle',
                                            'followup-training', 'followup-stress-score', 'followup-mcrc',
                                            'followup-bundle', 'followup-figures', 'followup-posthoc',
                                            'followup-fp-figure'])
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    globals()[args.command.replace('-', '_')](args)


if __name__ == '__main__':
    main()
