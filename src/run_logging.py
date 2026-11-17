"""Save enough information to explain and reproduce a run, without duplicate bookkeeping."""
import json
import shutil
import subprocess
import sys
from pathlib import Path
from config import ROOT, EXPERIMENTS
from data import digest


def snapshot_run(run_dir, settings):
    """Called after the data check: save effective settings, locked data, code and software."""
    run_dir = Path(run_dir)
    with (run_dir / 'experiment_config.json').open('x') as f:
        json.dump(settings, f, indent=2)
    shutil.copyfile(EXPERIMENTS / 'data_lock.json', run_dir / 'data_lock.json')
    source_dir = run_dir / 'source_code'
    source_dir.mkdir()
    sources = list((ROOT / 'src').glob('*.py')) + [
        p for p in EXPERIMENTS.iterdir() if p.suffix in ('.py', '.sh', '.slurm', '.json', '.md')]
    records = []
    for source in sorted(sources):
        relative = source.relative_to(ROOT)
        target = source_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        records.append({'path': str(relative), 'sha256': digest(target)})
    for relative in ('data/splits/vsseg_baseline.csv', 'metadata/vs_seg_t1_master.csv'):
        shutil.copyfile(ROOT / relative, run_dir / Path(relative).name)
    git = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    provenance = {'git_commit': git, 'argv': sys.argv, 'python': sys.executable,
                  'data_lock_sha256': digest(run_dir / 'data_lock.json'), 'sources': records}
    (run_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    (run_dir / 'environment.txt').write_bytes(
        subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']))
    return settings
