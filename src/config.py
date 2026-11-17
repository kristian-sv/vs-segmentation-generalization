"""Read the one experiment settings file. No training or image loading here."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / 'experiments'
DATA = ROOT / 'data/processed'
WORK = EXPERIMENTS / 'work'


def load_config(family='baseline', seed=42):
    """Return a fresh settings dictionary; callers pass it explicitly to functions."""
    if family not in ('baseline', 'dropout01') or seed not in (42, 43, 44):
        raise ValueError('Use baseline/dropout01 and seed 42, 43 or 44')
    recipe = json.loads((EXPERIMENTS / 'settings.json').read_text())
    settings = dict(recipe['baseline'])
    if family == 'dropout01':
        settings['dropout'] = recipe['lower_dropout']
    settings['seed'] = seed
    settings['modality'] = 'T1'
    settings['image_col'] = 't1_path'
    settings['experiment_name'] = f't1_mvp_{family}_s{seed}'
    return settings
