"""Exercise the installed nnU-Net trainer for three updates; keep diagnostic outputs separate."""
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'experiments')]
from data import check_lock, digest
from config import EXPERIMENTS as HERE
from nnunet_prepare import NAME, configure_paths, expected_split


def main():
    out = Path(sys.argv[1]).resolve()
    if out.exists() or out.with_suffix('').exists():
        raise FileExistsError(out)
    lock = check_lock()
    configure_paths()
    pre = Path(os.environ['nnUNet_preprocessed']) / NAME
    completed = json.loads((pre / 'preparation_complete.json').read_text())
    assert completed['data_lock_sha256'] == digest(HERE / 'data_lock.json')
    assert json.loads((pre / 'splits_final.json').read_text()) == expected_split(lock)
    for name, value in completed['frozen'].items():
        assert digest(pre / name) == value
    os.environ['nnUNet_results'] = str(out.with_suffix(''))
    os.environ['nnUNet_n_proc_DA'] = '0'
    os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '4'
    import numpy as np
    import torch
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    assert torch.cuda.is_available()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_num_threads(4)
    plans = json.loads((pre / 'nnUNetPlans.json').read_text())
    # Installed trainer's constructor consumes this runtime flag, not a planning field.
    plans['continue_training'] = False
    trainer = nnUNetTrainer(plans, '3d_fullres', 0,
                           json.loads((pre / 'dataset.json').read_text()), torch.device('cuda'))
    trainer.initialize()
    train_loader, val_loader = trainer.get_dataloaders()
    trainer.network.train()
    times = []
    for _ in range(3):
        start = time.monotonic()
        result = trainer.train_step(next(train_loader))
        assert np.isfinite(result['loss']).all()
        torch.cuda.synchronize()
        times.append(time.monotonic() - start)
    trainer.network.eval()
    start = time.monotonic()
    with torch.no_grad():
        result = trainer.validation_step(next(val_loader))
    assert np.isfinite(result['loss']).all()
    torch.cuda.synchronize()
    report = {'status': 'passed', 'diagnostic_only': True, 'training_updates': 3,
              'fold': 0, 'training_patients': 193, 'validation_patients': 24,
              'patch_size': list(trainer.configuration_manager.patch_size),
              'batch_size': trainer.batch_size, 'update_seconds': times,
              'validation_batch_seconds': time.monotonic() - start,
              'peak_gpu_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
              'compiled': trainer._do_i_compile(), 'saved_model': False, 'test_scores': False,
              'numpy_version': np.__version__, 'gpu': torch.cuda.get_device_name(0)}
    with out.open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
