"""Score a completed Astra arm with the original metrics, then export reviewer masks."""
import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import tempfile

import nibabel as nib
import numpy as np

ROOT=Path('/Users/kristiansvendsen/MMIV/llm-test')
spec=importlib.util.spec_from_file_location('opus_score',ROOT/'harness/score.py')
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)

def main(effort, run_root=None, output=None, export=True):
    run_root=run_root or ROOT/'runs'/f'astra_{effort}'
    output=output or ROOT/'results'/f'astra_{effort}'
    records=list(csv.DictReader((ROOT/'private/case_key.csv').open()))
    assert len(records)==22
    assert all((run_root/r['patient_id']/'status.json').exists() for r in records),'Arm is incomplete'
    target=f'Astra {effort}'
    rows=[]; predictions={}
    for r in sorted(records,key=lambda r:r['patient_id']):
        pid=r['patient_id']; ref=nib.load(original.DATA/pid/'label.nii.gz')
        gt=np.asarray(ref.dataobj)>0;sp=ref.header.get_zooms()[:3]
        sources={}
        for name,folder in [(target,run_root),('Opus 5.5 high',ROOT/'runs/run1')]:
            status=json.loads((folder/pid/'status.json').read_text())['status']
            mask=original.load_mask(folder/pid/'label.nii.gz',ref) if status=='ok' else np.zeros_like(gt)
            sources[name]=(mask,status)
        for filename,name in original.COMPARATORS.items():
            sources[name]=(original.load_mask(original.BUNDLE/pid/(filename+'.nii.gz'),ref),'ok')
        for name,(mask,status) in sources.items():
            rows.append({'patient':pid,'run_order_case':r['case'],'stratum':original.stratum(pid),'source':name,'status':status,
                         **{k:round(float(v),4) for k,v in original.metrics(mask,gt,sp).items()}})
        if export:
            mask=sources[target][0]
            im=nib.Nifti1Image(mask.astype(np.uint8),ref.affine,ref.header);im.set_data_dtype(np.uint8)
            dest=original.BUNDLE/pid/f'pred_llm_astra_{effort}_run1.nii.gz'
            with tempfile.NamedTemporaryFile(dir=dest.parent,prefix='.astra-',suffix='.nii.gz',delete=False) as f:temp=Path(f.name)
            try:nib.save(im,temp);os.replace(temp,dest)
            finally:temp.unlink(missing_ok=True)
            predictions[pid]=str(dest)
        print(f'SCORED {effort} {pid}',flush=True)
    output.mkdir(parents=True,exist_ok=True)
    with (output/'scores.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    patients=sorted({r['patient'] for r in rows})
    lookup={(r['patient'],r['source']):r for r in rows}
    arms={target:[target],'Opus 5.5 high':['Opus 5.5 high'],'plain arm':['plain s42','plain s43'],
          'augmented arm':['augmented s42','augmented s43'],'nnU-Net initial':['nnU-Net initial']}
    values={name:np.array([np.mean([lookup[(pid,s)]['dice'] for s in sources]) for pid in patients]) for name,sources in arms.items()}
    lines=[f'# Astra {effort}: 22 MC-RC patients','',
        'Model: `gpt-6-astra`. One independent session per patient. Original Opus prompt, toolbox, image sandbox, case order, and 40-call / 20-image / 60-minute budgets. Invalid or unsaved masks count as empty predictions. Same original Dice, volume-error and HD95 implementations.','',
        'Bootstrap intervals are descriptive: 10,000 patient resamples, seed 42. The prior pilot/development exposure to patient 028 remains a limitation. Model-provider image encoding and system/developer message transport differ between Claude and Codex; image bytes, prompt text, and tool semantics are preserved.','']
    groups={'All patients':None,'Primary slab cases (13)':original.SLAB13,**original.STRATA}
    for group,ids in groups.items():
        selected=np.array([ids is None or pid.split('-')[-1] in ids for pid in patients]);n=int(selected.sum())
        lines += [f'## {group} (n={n})','','| Source | Mean Dice | Median Dice | Empty outputs |','|---|---:|---:|---:|']
        for name,sources in arms.items():
            d=values[name][selected];empty=sum(lookup[(pid,s)]['empty'] for pid,use in zip(patients,selected) if use for s in sources)
            lines.append(f'| {name} | {d.mean():.3f} | {np.median(d):.3f} | {int(empty)}/{n*len(sources)} |')
        lines += ['','| Paired difference (Astra minus) | Mean | 95% CI |','|---|---:|---|']
        for name in arms:
            if name==target:continue
            diff=(values[target]-values[name])[selected];lo,hi=original.boot_ci(diff)
            lines.append(f'| {name} | {diff.mean():+.3f} | [{lo:+.3f}, {hi:+.3f}] |')
        lines.append('')
    lines += ['## Per-patient Dice','','| Patient | Stratum | '+' | '.join(arms)+' |','|---|---|'+'---:|'*len(arms)]
    for i,pid in enumerate(patients):lines.append(f'| {pid} | {original.stratum(pid)} | '+' | '.join(f'{values[a][i]:.3f}' for a in arms)+' |')
    (output/'summary.md').write_text('\n'.join(lines)+'\n')
    (output/'reviewer_exports.json').write_text(json.dumps(predictions,indent=2)+'\n')
    (output/'figures').mkdir(exist_ok=True)
    order=[p for group in original.STRATA for p in patients if original.stratum(p)==group]
    fig,ax=original.plt.subplots(figsize=(12,4))
    for name,marker in [(target,'o'),('Opus 5.5 high','x'),('augmented arm','s'),('plain arm','^')]:
        ax.scatter(np.arange(22),[values[name][patients.index(p)] for p in order],label=name,marker=marker,s=28)
    for boundary in [7,18]:ax.axvline(boundary-.5,color='k',lw=.6,ls=':')
    ax.set_xticks(np.arange(22),[p.split('-')[-1] for p in order],fontsize=8);ax.set_ylim(-.03,1)
    ax.set_ylabel('Dice');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(output/'figures/dice_by_patient.png',dpi=150);original.plt.close(fig)
    print(f'RESULTS {output}; reviewer exports: {len(predictions)}',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--effort',choices=['high','xhigh'],required=True);a=p.parse_args();main(a.effort)
