"""Check the frozen patient list and file identities before an experiment."""
import csv
import hashlib
import json
from config import ROOT, EXPERIMENTS, DATA


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def check_lock(full=False):
    """Verify the original split; only train/val image contents may be hashed."""
    lock = json.loads((EXPERIMENTS / 'data_lock.json').read_text())
    for rel, expected in lock['metadata_sha256'].items():
        if digest(ROOT / rel) != expected:
            raise ValueError(f'Metadata changed: {rel}')
    with (ROOT / 'data/splits/vsseg_baseline.csv').open() as f:
        splits = list(csv.DictReader(f))
    with (ROOT / 'metadata/vs_seg_t1_master.csv').open() as f:
        master = list(csv.DictReader(f))
    ids = [r['case_id'] for r in splits]
    lookup = {r['patient_id']: r for r in master}
    if len(ids) != len(set(ids)) or len(lookup) != len(master):
        raise ValueError('Duplicate/overlapping patient identities')
    if set(ids) - set(lookup) != {'VS-SEG-159'} or set(lookup) - set(ids):
        raise ValueError('Unexpected missing/extra patients')
    rows = []
    for row in splits:
        pid, split = row['case_id'], row['split']
        if split not in ('train', 'val', 'test'):
            raise ValueError('Unknown patient partition')
        if pid == 'VS-SEG-159':
            if split != 'train':
                raise ValueError('Known missing case changed partition')
            continue
        image, label = f'VS_SEG_T1/{pid}/T1.nii.gz', f'VS_SEG_T1/{pid}/label.nii.gz'
        if lookup[pid]['t1_path'] != image or lookup[pid]['seg_path'] != label:
            raise ValueError(f'Changed/reused image or label: {pid}')
        if not all((DATA / p).is_file() for p in (image, label)):
            raise ValueError(f'Missing image/label: {pid}')
        rows.append(dict(case_id=pid, split=split, image=image, label=label))
    expected = [{k: r[k] for k in ('case_id', 'split', 'image', 'label')} for r in lock['patients']]
    if rows != expected:
        raise ValueError('Frozen patient list changed')
    if full:
        for r in lock['patients']:
            if r['split'] in ('train', 'val'):
                for key in ('image', 'label'):
                    if digest(DATA / r[key]) != r[key + '_sha256']:
                        raise ValueError(f'Changed input: {r[key]}')
    return lock


def development_pairs(lock, split):
    if split not in ('train', 'val'):
        raise ValueError('Training loaders accept train/val only; evaluation is separate')
    return [dict(case_id=r['case_id'], image=str(DATA / r['image']), label=str(DATA / r['label']))
            for r in lock['patients'] if r['split'] == split]
