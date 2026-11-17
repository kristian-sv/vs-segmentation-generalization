#!/usr/bin/env python3
"""Synthetic scorer/export tests, optionally real development geometry; no model scores."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import tempfile
import unittest
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]
import subprocess

import nibabel as nib
import numpy as np

from evaluate import (FIELDS, binary_mask, check_grid, digest, manual_pipeline,
                     pair_geometry, patient_metrics, prediction_paths, preprocess_image,
                     read_manifest, restore_probabilities, save_mask, score, summary,
                     validate_checkpoint_config, write_csv, write_json)


def volume(data, affine):
    img = nib.Nifti1Image(data, affine)
    img.header.set_xyzt_units('mm')
    return img


class Checks(unittest.TestCase):
    def setUp(self):
        (HERE / 'work').mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=HERE / 'work')
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)

    def test_empty_dice_and_volumes(self):
        empty = np.zeros((9, 10, 11), dtype=bool)
        one = empty.copy()
        one[3, 4, 5] = True
        affine = np.diag([0.7, 1.3, 2.8, 1])
        for pred, ref, expected in [(empty, empty, 1), (empty, one, 0),
                                    (one, empty, 0), (one, one, 1)]:
            self.assertEqual(patient_metrics(pred, ref, affine)['dice'], expected)
        two = one.copy()
        two[3, 4, 6] = True
        self.assertAlmostEqual(patient_metrics(one, two, affine)['dice'], 2 / 3)
        self.assertAlmostEqual(patient_metrics(one, two, affine)['prediction_ml'], 0.7 * 1.3 * 2.8 / 1000)
        self.assertIsNone(summary([1])['sample_sd'])
        self.assertAlmostEqual(summary([0, 1])['sample_sd'], np.sqrt(0.5))

    def test_geometry_and_mask_refusals(self):
        image = volume(np.zeros((8, 9, 10), np.uint8), np.eye(4))
        shifted = np.eye(4)
        shifted[0, 3] = 0.01
        for wrong in [volume(np.zeros((8, 9, 11), np.uint8), np.eye(4)),
                      volume(np.zeros(image.shape, np.uint8), shifted)]:
            with self.assertRaisesRegex(ValueError, 'Grid mismatch'):
                check_grid(image, wrong)
        wrong = volume(np.zeros(image.shape, np.uint8), np.eye(4))
        wrong.header.set_xyzt_units('unknown')
        with self.assertRaisesRegex(ValueError, 'units'):
            check_grid(image, wrong)
        for value in (2, 0.25, np.nan):
            with self.assertRaisesRegex(ValueError, 'binary'):
                binary_mask(volume(np.full(image.shape, value, dtype=np.float32), np.eye(4)))

    def test_checkpoint_config_guard(self):
        from config import load_config
        values = load_config('baseline', 42)
        checkpoint = self.work / 'fixture.pt'
        checkpoint.write_bytes(b'synthetic checkpoint fixture; never loaded')
        with self.assertRaises(FileNotFoundError):
            validate_checkpoint_config(checkpoint, values)
        config_path = self.work / 'experiment_config.json'
        expected = json.loads(json.dumps(values, default=str))
        config_path.write_text(json.dumps(dict(expected, schema_version=2)))
        result = validate_checkpoint_config(checkpoint, values)
        self.assertEqual(result['experiment_config_sha256'], digest(config_path))
        self.assertEqual(result['checkpoint_sha256'], digest(checkpoint))
        for key in expected:
            changed = dict(expected)
            changed.pop(key)
            config_path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, key):
                validate_checkpoint_config(checkpoint, values)
        for key, value in dict(modality='T2', image_col='t2_path', seed=43, dropout=0.1,
                               channels=[16, 32, 64, 128, 256], pixdim=[1, 1, 2],
                               patch_size=[64, 64, 64]).items():
            config_path.write_text(json.dumps(dict(expected, **{key: value})))
            with self.assertRaisesRegex(ValueError, key):
                validate_checkpoint_config(checkpoint, values)

    def make_pairs(self):
        rows = []
        for i, pid in enumerate(['synthetic-B', 'synthetic-A']):
            image = self.work / (pid + '-image.nii.gz')
            reference = self.work / (pid + '-label.nii.gz')
            data = np.zeros((9, 10, 11), dtype=np.uint8)
            data[2 + i, 3, 4] = 1
            nib.save(volume(data.astype(np.float32), np.eye(4)), image)
            nib.save(volume(data, np.eye(4)), reference)
            rows.append(dict(case_id=pid, image=str(image), reference=str(reference),
                             image_sha256=digest(image), reference_sha256=digest(reference)))
        return rows

    def test_patient_pairing_coverage_hashes_and_scores(self):
        rows = self.make_pairs()
        manifest = self.work / 'pairs.csv'
        write_csv(manifest, rows, FIELDS)
        self.assertEqual(read_manifest(manifest), rows)
        pred_dir, compare_dir = self.work / 'pred', self.work / 'compare'
        pred_dir.mkdir()
        compare_dir.mkdir()
        for row in reversed(rows):
            original = nib.load(row['reference'])
            save_mask(binary_mask(original), original, pred_dir / (row['case_id'] + '.nii.gz'))
            save_mask(np.zeros(original.shape, bool), original,
                      compare_dir / (row['case_id'] + '.nii.gz'))
        records, report = score(rows, pred_dir, compare_dir)
        self.assertEqual([r['case_id'] for r in records], ['synthetic-B', 'synthetic-A'])
        self.assertEqual(report['dice']['mean'], 1)
        self.assertEqual(report['dice_difference_prediction_minus_compare']['mean'], 1)
        with self.assertRaisesRegex(ValueError, 'coverage'):
            prediction_paths(rows[:1], pred_dir)
        with self.assertRaisesRegex(ValueError, 'coverage'):
            prediction_paths(rows, self.work / 'missing')
        rows[1]['case_id'] = rows[0]['case_id']
        duplicate = self.work / 'duplicate.csv'
        write_csv(duplicate, rows, FIELDS)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            read_manifest(duplicate)
        rows[0]['image_sha256'] = '0' * 64
        tampered = self.work / 'tampered.csv'
        write_csv(tampered, rows[:1], FIELDS)
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            read_manifest(tampered)

    def test_inverse_continuous_orientation_anisotropy_crop_and_padding(self):
        import torch
        from monai.inferers import sliding_window_inference
        values, transform = manual_pipeline('baseline', 42)
        for affine in [np.diag([-1.8, -0.7, 2.4, 1.0]),
                       np.array([[0., 0., -2.4, 50.], [-1.8, 0., 0., 70.],
                                 [0., 0.7, 0., -40.], [0., 0., 0., 1.]])]:
            image = np.zeros((53, 43, 37), dtype=np.float32)
            image[13:38, 11:33, 9:29] = 1 + np.arange(25)[:, None, None]
            original = volume(image, affine)
            path = self.work / 'image.nii.gz'
            label_path = self.work / 'label.nii.gz'
            nib.save(original, path)
            nib.save(volume((image > 3).astype(np.uint8), affine), label_path)
            transformed = preprocess_image(path, transform)
            training = transform({'image': str(path), 'label': str(label_path)})['image']
            torch.testing.assert_close(transformed.as_tensor(), training.as_tensor(), rtol=0, atol=0)
            self.assertEqual(tuple(transformed.shape[1:]), tuple(values['patch_size']))
            trace = str(transformed.applied_operations)
            for name in ('Orientation', 'SpatialResample', 'CropForeground', 'DivisiblePad', 'SpatialPad'):
                self.assertIn(name, trace)
            # Analytic continuous ramp in ORIGINAL voxel coordinates. Inverse linear
            # interpolation must reproduce it centrally; nearest/argmax-first cannot.
            mapping = np.linalg.inv(affine) @ transformed.affine.cpu().numpy()
            axes = np.ogrid[tuple(slice(n) for n in transformed.shape[1:])]
            original_x = mapping[0, 3] + sum(mapping[0, i] * axes[i] for i in range(3))
            p = np.broadcast_to(np.clip(0.15 + 0.7 * original_x / 52, 0, 1), transformed.shape[1:])
            probabilities = torch.from_numpy(np.stack((1 - p, p)).astype(np.float32))
            restored = restore_probabilities(probabilities, transformed, transform, original)
            expected = np.broadcast_to(0.15 + 0.7 * np.arange(53)[:, None, None] / 52, original.shape)
            interior = np.s_[16:35, 14:30, 12:26]
            np.testing.assert_allclose(restored[1][interior], expected[interior], atol=2e-5, rtol=0)
            np.testing.assert_allclose(restored.sum(axis=0), 1, atol=1e-6)
            np.testing.assert_array_equal(restored[:, 0, 0, 0], [1, 0])
            hard_first = torch.nn.functional.one_hot(probabilities.argmax(0), 2).movedim(-1, 0).float()
            hard_restored = restore_probabilities(hard_first, transformed, transform, original)
            self.assertGreater(float(np.abs(restored[1][interior] - hard_restored[1][interior]).max()), 0.2)
            # Exercise the full sliding-window -> softmax -> inverse -> NIfTI path
            # using a fixed synthetic predictor, without trained weights or scores.
            logits = sliding_window_inference(transformed.as_tensor()[None], values['patch_size'], 1,
                                              lambda x: torch.cat((-x, x), dim=1),
                                              overlap=values['sw_overlap'], mode='constant')
            exported = restore_probabilities(logits[0].softmax(0), transformed, transform, original)
            out = self.work / ('export-' + str(int(affine[0, 0])) + '.nii.gz')
            save_mask(exported.argmax(0), original, out)
            check_grid(original, nib.load(out))
            binary_mask(nib.load(out))
            with self.assertRaises(FileExistsError):
                save_mask(exported.argmax(0), original, out)


def real_checks():
    """All train/val pair geometry, then one train and one val artificial probability inverse."""
    import torch
    from config import DATA
    from data import check_lock
    _, transform = manual_pipeline('baseline', 42)
    lock = check_lock()
    records, restored_splits = [], set()
    for r in lock['patients']:
        if r['split'] not in ('train', 'val'):
            continue
        row = dict(case_id=r['case_id'], image=str(DATA / r['image']), reference=str(DATA / r['label']))
        for key, source in [('image', 'image'), ('reference', 'label')]:
            if digest(row[key]) != r[source + '_sha256']:
                raise ValueError(f'Frozen data mismatch: {r["case_id"]} {key}')
        original, _ = pair_geometry(row)
        record = dict(case_id=r['case_id'], split=r['split'], shape=original.shape,
                      affine=original.affine.tolist(), binary_reference=True, hashes_verified=True)
        if r['split'] not in restored_splits:
            transformed = preprocess_image(row['image'], transform)
            # Artificial uniform field tests real spatial metadata only, never model performance.
            p = torch.empty((2, *transformed.shape[1:]), dtype=torch.float32)
            p[0], p[1] = 0.75, 0.25
            restored = restore_probabilities(p, transformed, transform, original)
            assert restored.shape == (2, *original.shape)
            assert np.isfinite(restored).all() and np.allclose(restored.sum(0), 1)
            assert (restored.argmax(0) == 0).all()
            with tempfile.TemporaryDirectory(dir=HERE / 'work') as tmp:
                path = Path(tmp) / 'geometry-only.nii.gz'
                save_mask(restored.argmax(0), original, path)
                check_grid(original, nib.load(path))
            record['artificial_probability_roundtrip'] = True
            record['transformed_shape'] = list(transformed.shape[1:])
            restored_splits.add(r['split'])
            del p, restored, transformed
        records.append(record)
        if len(records) % 20 == 0:
            print(f'Geometry only: {len(records)} development pairs checked', flush=True)
    if {s: sum(r['split'] == s for r in records) for s in ('train', 'val')} != {'train': 193, 'val': 24}:
        raise ValueError('Unexpected development patient coverage')
    return records


def manual_cli_smoke(directory):
    """Actual checkpoint/export CLI on one val grid; initialized weights, no scores."""
    import torch
    from monai.networks.nets import UNet
    from config import DATA
    from data import check_lock
    directory.mkdir(parents=True, exist_ok=False)
    values, _ = manual_pipeline('baseline', 42)
    row = next(r for r in check_lock()['patients'] if r['split'] == 'val')
    manifest = directory / 'one_val.csv'
    write_csv(manifest, [dict(case_id=row['case_id'], image=str(DATA / row['image']),
                             reference=str(DATA / row['label']),
                             image_sha256=row['image_sha256'], reference_sha256=row['label_sha256'])], FIELDS)
    torch.manual_seed(42)
    model = UNet(spatial_dims=3, in_channels=1, out_channels=2, channels=values['channels'],
                 strides=(2, 2, 2, 2), norm='instance', act='LeakyReLU', dropout=values['dropout'])
    checkpoint = directory / 'UNTRAINED_DIAGNOSTIC_ONLY.pt'
    torch.save(model.state_dict(), checkpoint)
    write_json(directory / 'experiment_config.json',
               dict(values, diagnostic='fresh initialized weights; export plumbing only; do not score'))
    del model
    prediction_dir = directory / 'untrained_export_DO_NOT_SCORE'
    command = [sys.executable, str(ROOT / 'src/evaluate.py'), 'manual-export',
               '--manifest', str(manifest), '--manual', 'baseline', '--seed', '42',
               '--checkpoint', str(checkpoint), '--device', 'cuda', '--output', str(prediction_dir)]
    print('Running initialized-weight diagnostic CLI: ' + ' '.join(command), flush=True)
    subprocess.run(command, check=True)
    original = nib.load(DATA / row['image'])
    prediction = nib.load(prediction_dir / (row['case_id'] + '.nii.gz'))
    check_grid(original, prediction, 'initialized-weight export')
    binary_mask(prediction)
    completed = json.loads((prediction_dir / 'complete.json').read_text())
    assert completed['patient_ids'] == [row['case_id']]
    return dict(case_id=row['case_id'], shape=prediction.shape, device='cuda', command=command,
                checkpoint_sha256=digest(checkpoint), diagnostic='initialized weights; no model score',
                exported_grid_verified=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real', action='store_true', help='Also verify all 217 frozen development pairs')
    parser.add_argument('--manual-cli-smoke', action='store_true',
                        help='GPU: initialize fresh weights, invoke actual export CLI on one val image; no scoring')
    parser.add_argument('--output', type=Path, help='New JSON report; refuses overwrite')
    args = parser.parse_args()
    if args.manual_cli_smoke and not args.output:
        parser.error('--manual-cli-smoke requires --output; artifacts use its stem as a new directory')
    if args.output and args.output.exists():
        raise FileExistsError(args.output)
    import torch
    torch.set_num_threads(2)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    if not result.wasSuccessful():
        raise SystemExit(1)
    report = dict(synthetic_test_groups=result.testsRun, scorer_sha256=digest(ROOT / 'src/evaluate.py'),
                  checks_sha256=digest(__file__),
                  versions={p: importlib.metadata.version(p) for p in ('torch', 'monai', 'numpy', 'nibabel', 'scipy')})
    if args.real:
        report['real_geometry_only'] = real_checks()
    if args.manual_cli_smoke:
        report['manual_cli_smoke'] = manual_cli_smoke(args.output.with_suffix('').resolve())
    if args.output:
        write_json(args.output, report)
    print('PASS: synthetic checks' + (' and 217 development geometry checks; no model scores' if args.real else ''))


if __name__ == '__main__':
    main()
