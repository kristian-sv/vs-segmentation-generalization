"""Check patient exclusion and training-only planning without loading patient images."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'experiments'), str(ROOT / 'src')]
import nnunet_prepare as prepare
from predict_review import assignments


class CrossValidationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lock = json.loads((ROOT / 'experiments/data_lock.json').read_text())

    def test_every_development_patient_held_out_once(self):
        dev = {r['case_id'] for r in self.lock['patients'] if r['split'] in ('train', 'val')}
        test = {r['case_id'] for r in self.lock['patients'] if r['split'] == 'test'}
        held_out = []
        for fold in range(5):
            split = prepare.expected_split(self.lock, fold)[0]
            train, val = set(split['train']), set(split['val'])
            self.assertFalse(train & val)
            self.assertEqual(train | val, dev)
            self.assertFalse((train | val) & test)
            self.assertNotIn('VS-SEG-159', train | val)
            self.assertIn(len(val), (43, 44))
            held_out.extend(val)
        self.assertEqual(len(held_out), 217)
        self.assertEqual(set(held_out), dev)

    def test_assignment_is_stable_and_duplicates_rejected(self):
        reordered = copy.deepcopy(self.lock)
        reordered['patients'].reverse()
        for fold in range(5):
            self.assertEqual(prepare.expected_split(self.lock, fold),
                             prepare.expected_split(reordered, fold))
        reordered['patients'].append(next(r for r in reordered['patients'] if r['split'] == 'train'))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            prepare.expected_split(reordered, 0)
        with self.assertRaises(ValueError):
            prepare.expected_split(self.lock, 5)

    def test_original_experiment_split_is_preserved(self):
        saved = ROOT / 'analysis/reports/training_readiness_20260908/splits_final.json'
        self.assertEqual(prepare.expected_split(self.lock), json.loads(saved.read_text()))
        self.assertEqual(prepare.dataset_name(), 'Dataset901_T1VSComparison')
        self.assertEqual(len({prepare.dataset_name(i) for i in range(5)}), 5)

    def test_review_assigns_every_patient_to_an_untrained_model(self):
        splits = [prepare.expected_split(self.lock, fold)[0] for fold in range(5)]
        rows = assignments(self.lock, splits)
        self.assertEqual(len(rows), 242)
        self.assertEqual(len({r['case_id'] for r in rows}), 242)
        self.assertEqual(sum(r['prediction_source'] == 'existing_out_of_fold' for r in rows), 217)
        for row in rows:
            self.assertNotIn(row['case_id'], splits[row['cv_fold']]['train'])
            if row['prediction_source'] == 'new_inference':
                self.assertEqual(row['cv_fold'], 0)
                self.assertTrue(row['original_split'] == 'test' or row['case_id'] == 'VS-SEG-159')
        unsafe = copy.deepcopy(splits)
        unsafe[0]['train'].append(unsafe[0]['val'][0])
        with self.assertRaisesRegex(ValueError, 'both training and held-out'):
            assignments(self.lock, unsafe)
        unsafe = copy.deepcopy(splits)
        unsafe[0]['val'].append(unsafe[0]['val'][0])
        with self.assertRaisesRegex(ValueError, 'more than once'):
            assignments(self.lock, unsafe)

    def test_planning_excludes_held_out_patients(self):
        split = prepare.expected_split(self.lock, 0)[0]
        staged = []
        api_name = 'nnunetv2.experiment_planning.plan_and_preprocess_api'
        api = types.ModuleType(api_name)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pre = root / 'work/nnUNet_preprocessed' / prepare.dataset_name(0)

            def stage(rows, raw):
                self.assertTrue(all(r['image'].startswith('VS_SEG_T1/') for r in rows))
                staged.extend(r['case_id'] for r in rows)

            def fingerprint(dataset_id, **kwargs):
                self.assertEqual(dataset_id, 902)
                self.assertEqual(staged, split['train'])
                pre.mkdir(parents=True)
                (pre / 'dataset_fingerprint.json').write_text('{}')

            def plan(dataset_id, **kwargs):
                self.assertEqual(staged, split['train'])
                (pre / 'nnUNetPlans.json').write_text('{}')

            def preprocess(dataset_id, **kwargs):
                self.assertEqual(staged, split['train'] + split['val'])
                self.assertEqual(kwargs['configurations'], ['3d_fullres'])
                self.assertEqual(json.loads((pre / 'dataset.json').read_text())['channel_names'], {'0': 'T1'})

            api.extract_fingerprint_dataset = fingerprint
            api.plan_experiment_dataset = plan
            api.preprocess_dataset = preprocess
            with patch.object(prepare, 'ROOT', root), patch.object(prepare, 'WORK', root / 'work'), \
                    patch.object(prepare, 'stage', stage), patch.object(prepare, 'check_lock', return_value=self.lock), \
                    patch.dict(sys.modules, {api_name: api}), patch.dict(prepare.os.environ), \
                    patch.object(sys, 'argv', ['nnunet_prepare.py', '--cv-fold', '0', '--execute']), \
                    contextlib.redirect_stdout(io.StringIO()):
                prepare.main()
                self.assertEqual(prepare.check_preparation(self.lock, 0), pre)
                provenance = pre / 'planning_provenance.json'
                wrong = json.loads(provenance.read_text())
                wrong['planning_patients'].append(split['val'][0])
                provenance.write_text(json.dumps(wrong))
                with self.assertRaisesRegex(ValueError, 'Planning patients'):
                    prepare.check_preparation(self.lock, 0)
                with self.assertRaisesRegex(ValueError, 'refusing to overwrite'):
                    prepare.main()


if __name__ == '__main__':
    unittest.main()
