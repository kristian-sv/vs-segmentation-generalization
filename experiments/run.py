"""Start one planned experiment. Without --execute this only checks and prints settings."""
import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from config import ROOT, load_config
from data import check_lock
from run_logging import snapshot_run


TRAINERS = {'nnUNetTrainer': 'plain', 'nnUNetTrainerSlabThick': 'slabthick'}
EXTERNAL_TRAINERS = ROOT / 'experiments/nnunet_trainers'


def train_nnunet(lock, cv_fold=None, seed=42, trainer='nnUNetTrainer', smoke=False):
    from nnunet_prepare import check_preparation, dataset_name
    pre = check_preparation(lock, cv_fold)
    name = dataset_name(cv_fold)
    if seed != 42 or trainer != 'nnUNetTrainer' or smoke:
        # Preregistered follow-up runs (PREREGISTRATION.md): Dataset901 plans/split, separate results root per run.
        run_name = f"{'smoke_' if smoke else ''}{TRAINERS[trainer]}_s{seed}"
        os.environ['nnUNet_results'] = str(ROOT / 'outputs/t1/nnunet_slab' / run_name)
        os.environ['nnUNet_extTrainer'] = str(EXTERNAL_TRAINERS)
    results = Path(os.environ['nnUNet_results'])
    if cv_fold is not None:
        results = results / name
    if results.exists():
        raise FileExistsError('nnU-Net results already exist; inspect the existing run')
    import importlib.metadata
    if importlib.metadata.version('nnunetv2') != '2.8.1':
        raise ValueError('This comparison uses nnU-Net 2.8.1')
    import numpy as np
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Use the allocated GPU environment')
    os.environ['nnUNet_n_proc_DA'] = '0'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    snapshot = results / 'setup_snapshot'
    snapshot.mkdir(parents=True)
    snapshot_run(snapshot, {'pipeline': trainer, 'seed': seed, 'fold': 0, 'smoke_test': smoke,
                           'cv_fold': cv_fold, 'dataset': name, 'modality': 'T1',
                           'configuration': '3d_fullres',
                           'epochs': int(os.environ.get('T1_SMOKE_EPOCHS', '3')) if smoke else 1000,
                           'augmentation_processes': 0,
                           'checkpoint': 'checkpoint_best.pth' if cv_fold is None else 'checkpoint_final.pth'})
    import nnunetv2
    shutil.copytree(Path(nnunetv2.__file__).parent, snapshot / 'nnunetv2',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    if trainer != 'nnUNetTrainer':
        shutil.copytree(EXTERNAL_TRAINERS, snapshot / 'nnunet_trainers',
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for filename in ('nnUNetPlans.json', 'dataset_fingerprint.json', 'splits_final.json',
                     'dataset.json', 'preparation_complete.json', 'planning_provenance.json'):
        shutil.copyfile(pre / filename, snapshot / filename)
    if smoke:  # a few epochs on Dataset901, sample patches for review; no validation, never a result
        from nnunetv2.run.run_training import get_trainer_from_args
        sys.path.insert(0, str(EXTERNAL_TRAINERS))
        from nnUNetTrainerSlabThick import smoke_test
        model = get_trainer_from_args(name, '3d_fullres', 0, trainer, device=torch.device('cuda'))
        model.num_epochs = int(os.environ.get('T1_SMOKE_EPOCHS', '3'))
        torch.backends.cudnn.deterministic = False  # as nnunetv2.run.run_training sets before training
        torch.backends.cudnn.benchmark = True
        smoke_test(model, ROOT / 'analysis/reports/model_insight_20261012/viewer_bundle/data/augmentation_smoke')
        return
    from nnunetv2.run.run_training import run_training
    run_training(name, '3d_fullres', 0, trainer_class_name=trainer, val_with_best=cv_fold is None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    kind = parser.add_mutually_exclusive_group(required=True)
    kind.add_argument('--manual', choices=['baseline', 'dropout01'])
    kind.add_argument('--nnunet', action='store_true')
    parser.add_argument('--seed', type=int, choices=[42, 43, 44], default=42)
    parser.add_argument('--cv-fold', type=int, choices=range(5), help='Separate T1 five-fold experiment')
    parser.add_argument('--trainer', choices=sorted(TRAINERS), default='nnUNetTrainer',
                        help='nnU-Net only; nnUNetTrainerSlabThick is the preregistered augmentation')
    parser.add_argument('--smoke', action='store_true', help='nnU-Net only: few epochs and sample patches, no result')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.cv_fold is not None and not args.nnunet:
        parser.error('--cv-fold requires --nnunet')
    if args.nnunet and (args.seed == 44 or (args.cv_fold is not None and (args.seed != 42 or args.trainer != 'nnUNetTrainer'))):
        parser.error('nnU-Net: seed 42 (original) or 43 (preregistered follow-up); five-fold runs use seed 42 and the plain trainer')
    if args.manual and (args.trainer != 'nnUNetTrainer' or args.smoke):
        parser.error('--trainer/--smoke apply to nnU-Net only')
    lock = check_lock(full=args.execute)
    if args.manual:
        settings = load_config(args.manual, args.seed)
        print(json.dumps(settings, indent=2))
    elif args.cv_fold is None:
        print(f'nnU-Net 2.8.1, 3d_fullres, fold 0, {args.trainer}, seed {args.seed}; same 193/24 train/val split'
              + ('; SMOKE TEST' if args.smoke else ''))
    else:
        from nnunet_prepare import dataset_name, expected_split
        split = expected_split(lock, args.cv_fold)[0]
        print(f'{dataset_name(args.cv_fold)}: nnU-Net 2.8.1, T1 3d_fullres, seed 42; '
              f'{len(split["train"])}/{len(split["val"])} train/held-out; '
              '1000 epochs, final checkpoint; original test and MC-RC excluded')
    if not args.execute:
        print('CHECK ONLY. No training or test evaluation started.')
        return
    if args.manual:
        from train import main as train_manual
        train_manual(settings, lock)
    else:
        train_nnunet(lock, args.cv_fold, args.seed, args.trainer, args.smoke)


if __name__ == '__main__':
    main()
