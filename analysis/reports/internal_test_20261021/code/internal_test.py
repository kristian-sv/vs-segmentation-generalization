#!/usr/bin/env python3
"""Preregistered internal-test evaluation (PROTOCOL.md): the 24 reserved VS-SEG test patients, unchanged and under the
11 stress conditions, with the existing code in analysis/model_insight.py and src/evaluate.py. Writes only below
analysis/reports/internal_test_20261021/ and never overwrites.

    build -> GPU array 'predict' (1-4 nnU-Net follow-up models, 5-10 manual runs) -> score -> summary
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'analysis'))
import model_insight as mi  # noqa: E402

# Point the existing stress-test code at this folder and at the internal-test manifest (nothing else changes).
mi.OUT = HERE
mi.STRESS = HERE / 'stress_test'
mi.COHORTS = {'validation': HERE / 'test_manifest.csv'}
MANIFEST = HERE / 'test_manifest.csv'
MODELS, ARMS = mi.MODELS, mi.ARMS
MANUAL = ('baseline_s42', 'baseline_s43', 'baseline_s44', 'dropout01_s42', 'dropout01_s43', 'dropout01_s44')
PRIMARY_C0, PRIMARY_STRESS = 'C0', 'S40T3'
MARGIN = -0.01
EARLIER = ROOT / 'analysis/reports/model_insight_20261012'


def read_rows(path):
    return mi.read_rows(path)


def build(_args):
    """The validation stress-test builder, unchanged, on the test manifest (side alternates in sorted ID order)."""
    if sorted(r['case_id'] for r in mi.read_manifest(MANIFEST)) != sorted(
            r['case_id'] for r in json.loads((ROOT / 'experiments/data_lock.json').read_text())['patients']
            if r['split'] == 'test'):
        raise SystemExit('Manifest patients differ from the 24 data_lock.json test patients')
    mi.stress_build(None)


def score(_args):
    """Restore re-stacked masks, score every prediction on its own grid (case_metrics = common Dice + FP components)."""
    import torch
    from concurrent.futures import ProcessPoolExecutor
    torch.set_num_threads(1)
    rows = mi.read_manifest(mi.STRESS / 'manifest.csv')
    for model in MODELS:
        restored = mi.STRESS / f'predictions/{model}_restacked'
        restored.mkdir()
        for row in rows:
            if row['case_id'].split('__')[0] in mi.RESTACKED:
                original = mi.nib.load(row['image'])
                mask, _ = mi.restore_from_training_order(
                    mi.nib.load(mi.STRESS / f"predictions_lps/{model}_restacked/{row['case_id']}.nii.gz"), original)
                mi.save_mask(mask, original, restored / f"{row['case_id']}.nii.gz")
    sets = []
    for model in MODELS:
        sets += [(model, 'nnunet', model[-2:], model, None),
                 (model + '_restacked', 'nnunet', model[-2:], model + '_restacked', mi.RESTACKED)]
    for _, _, _, directory, only in sets:  # coverage check before any score
        mi.prediction_paths([r for r in rows if not only or r['case_id'].split('__')[0] in only],
                            mi.STRESS / 'predictions' / directory)
    info = {f"{r['condition']}__{r['case_id']}": r for r in read_rows(mi.STRESS / 'perturbations.csv')}
    with ProcessPoolExecutor(max_workers=8) as pool:
        jobs = [pool.submit(mi.stress_unit, row, info[row['case_id']], sets) for row in rows]
        stress_rows = [r for job in jobs for r in job.result()]
    mi.write_table(mi.STRESS / 'stress_test.csv', stress_rows)

    # Unchanged scans: common scorer (src/evaluate.py score) for the 4 follow-up models and the 6 manual runs;
    # nnU-Net s42 of the original seven is plain_s42 (same checkpoint). case_metrics adds FP components.
    test = mi.read_manifest(MANIFEST)
    c0 = {model: mi.STRESS / f'predictions/{model}' for model in MODELS}
    c0.update({run: HERE / f'original_runs/{run}/predictions' for run in MANUAL})
    scored = HERE / 'scores'
    scored.mkdir()
    out = []
    for label, directory in c0.items():
        if label in MODELS:  # stress predictions are named C0__<id>; link them under the plain patient IDs
            linked = scored / f'c0_links/{label}'
            linked.mkdir(parents=True)
            for row in test:
                (linked / f"{row['case_id']}.nii.gz").symlink_to(directory / f"C0__{row['case_id']}.nii.gz")
            directory = linked
        records, report = mi.score(test, directory)
        mi.write_csv(scored / f'{label}_patients.csv', records)
        mi.write_json(scored / f'{label}_summary.json', report)
        paths = mi.prediction_paths(test, directory)
        for r in records:
            image, reference = mi.nib.load(r['image']), mi.nib.load(r['reference'])
            metrics = mi.case_metrics(mi.binary_mask(mi.nib.load(paths[r['case_id']])), mi.binary_mask(reference), image)
            if metrics['dice'] != r['dice']:
                raise SystemExit(f"{label} {r['case_id']}: case_metrics Dice differs from the common scorer")
            del metrics['empty_status']
            out.append(dict(run=label, case_id=r['case_id'], empty_status=r['empty_status'], **metrics))
    (HERE / 'tables').mkdir()
    mi.write_table(HERE / 'tables/per_case_unchanged_all_runs.csv', out)
    stress_c0 = {(r['run'], r['case_id']): float(r['dice']) for r in read_rows(mi.STRESS / 'stress_test.csv')
                 if r['condition'] == 'C0'}
    mismatch = [(r['run'], r['case_id']) for r in out if r['run'] in MODELS and stress_c0[r['run'], r['case_id']] != r['dice']]
    if mismatch:
        raise SystemExit(f'C0 stress rows differ from the common scorer: {mismatch}')
    print(f'{len(stress_rows)} stress rows; {len(out)} unchanged-scan rows; all checks passed')


def arm_classification(low, high, aug_empty, plain_empty):
    """Same wording and logic as the MC-RC step-B classification (model_insight.followup_mcrc)."""
    return ('inconclusive' if low <= 0 <= high else
            'beneficial' if low > 0 and aug_empty <= plain_empty else
            'not covered by rule: CI excludes 0 ' + ('below' if high < 0 else 'above, but more empty predictions'))


def summary(_args):
    rows = read_rows(mi.STRESS / 'stress_test.csv')
    patients = sorted({r['case_id'] for r in rows})
    if len(patients) != 24:
        raise SystemExit('Expected 24 patients')
    dice, empty, fp, fp_cases = {}, {}, {}, {}
    for r in rows:
        key = (r['condition'], r['run'])
        dice.setdefault(key, {})[r['case_id']] = float(r['dice'])
        empty[key] = empty.get(key, 0) + (r['pred_empty'] == 'True')
        fp[key] = fp.get(key, 0) + int(r['n_fp_components'])
        fp_cases[key] = fp_cases.get(key, 0) + (int(r['n_fp_components']) > 0)
    table, results = [], {}
    for condition in mi.CONDITIONS:
        for variant in ('', '_restacked'):
            if (condition, MODELS[0] + variant) not in dice:
                continue
            per_model = {m: dice[condition, m + variant] for m in MODELS}
            arms = mi.arm_values(per_model, patients)
            row = dict(condition=condition, input='restacked' if variant else 'as_stored', n=len(patients))
            for m in MODELS:
                values = np.array([per_model[m][p] for p in patients])
                k = (condition, m + variant)
                row.update({f'{m}_mean': float(values.mean()), f'{m}_median': float(np.median(values)),
                            f'{m}_empty': empty[k], f'{m}_below_0_1': int((values < .1).sum()),
                            f'{m}_fp_components': fp[k], f'{m}_cases_with_fp': fp_cases[k]})
            for arm, members in ARMS.items():
                row.update({f'{arm}_mean': float(arms[arm].mean()), f'{arm}_median': float(np.median(arms[arm])),
                            f'{arm}_empty': float(np.mean([empty[condition, m + variant] for m in members])),
                            f'{arm}_fp_components': float(np.mean([fp[condition, m + variant] for m in members]))})
            difference = arms['augmented_arm'] - arms['plain_arm']
            low, high = mi.bootstrap_ci(difference)
            row.update(arm_difference_mean=float(difference.mean()), arm_difference_ci95_low=low,
                       arm_difference_ci95_high=high, patients_higher=int((difference > 0).sum()),
                       patients_lower=int((difference < 0).sum()))
            for seed in ('42', '43'):
                d = np.array([per_model[f'slabthick_s{seed}'][p] - per_model[f'plain_s{seed}'][p] for p in patients])
                row[f's{seed}_difference_mean'] = float(d.mean())
                row[f's{seed}_difference_ci95_low'], row[f's{seed}_difference_ci95_high'] = mi.bootstrap_ci(d)
            table.append(row)
            results[condition, row['input']] = (row, arms, difference)
    mi.write_table(HERE / 'tables/summary_by_condition.csv', table)

    # Per-case table for the two primary conditions
    per_case = []
    for condition in (PRIMARY_C0, PRIMARY_STRESS):
        row_, arms, difference = results[condition, 'as_stored']
        for i, p in enumerate(patients):
            r = dict(condition=condition, case_id=p)
            for m in MODELS:
                r[f'dice_{m}'] = dice[condition, m][p]
            r.update(dice_plain_arm=float(arms['plain_arm'][i]), dice_augmented_arm=float(arms['augmented_arm'][i]),
                     arm_difference=float(difference[i]))
            per_case.append(r)
    mi.write_table(HERE / 'tables/per_case_primary.csv', per_case)

    # Primary rules
    a, _, da = results[PRIMARY_C0, 'as_stored']
    b, _, db = results[PRIMARY_STRESS, 'as_stored']
    rule_a = dict(rule='(a) unchanged internal test (C0): augmented-arm minus plain-arm mean Dice >= -0.01',
                  plain_arm_mean=a['plain_arm_mean'], augmented_arm_mean=a['augmented_arm_mean'],
                  difference=a['arm_difference_mean'], ci95=[a['arm_difference_ci95_low'], a['arm_difference_ci95_high']],
                  passes=a['arm_difference_mean'] >= MARGIN,
                  secondary_ci_lower_bound_above_margin=a['arm_difference_ci95_low'] >= MARGIN,
                  plain_arm_empty=a['plain_arm_empty'], augmented_arm_empty=a['augmented_arm_empty'])
    classification = arm_classification(b['arm_difference_ci95_low'], b['arm_difference_ci95_high'],
                                        b['augmented_arm_empty'], b['plain_arm_empty'])
    rule_b = dict(rule='(b) internal test S40T3: paired bootstrap 95% CI of augmented-minus-plain excludes 0 (above) '
                       'and augmented arm has no more empty predictions than plain',
                  plain_arm_mean=b['plain_arm_mean'], augmented_arm_mean=b['augmented_arm_mean'],
                  difference=b['arm_difference_mean'], ci95=[b['arm_difference_ci95_low'], b['arm_difference_ci95_high']],
                  plain_arm_empty=b['plain_arm_empty'], augmented_arm_empty=b['augmented_arm_empty'],
                  empty_per_model={m: b[f'{m}_empty'] for m in MODELS}, classification=classification,
                  passes=classification == 'beneficial')
    (HERE / 'bootstrap.json').write_text(json.dumps(dict(
        method=mi.bootstrap_ci.__doc__, replicates=10000, seed=42, patients=patients,
        arm_value='per-patient mean of the two seed Dice scores; no mask ensembling',
        arm_empty='mean of the two seeds\' empty-prediction counts (as in mcrc_followup/tables/summary.csv)',
        primary_a=rule_a, primary_b=rule_b,
        secondary_by_condition={f'{c} ({i})': dict(difference=r['arm_difference_mean'],
                                                   ci95=[r['arm_difference_ci95_low'], r['arm_difference_ci95_high']],
                                                   plain_arm_empty=r['plain_arm_empty'],
                                                   augmented_arm_empty=r['augmented_arm_empty'])
                                for (c, i), (r, _, _) in results.items()},
        original_runs=original_runs(patients)), indent=2) + '\n')
    comparison(table)
    print(json.dumps(dict(primary_a=rule_a, primary_b=rule_b), indent=2))


def original_runs(patients):
    """Secondary (RQ1 context): the seven original runs on the unchanged internal test."""
    rows = read_rows(HERE / 'tables/per_case_unchanged_all_runs.csv')
    dice = {(r['run'], r['case_id']): float(r['dice']) for r in rows}
    runs = MANUAL + ('plain_s42',)
    table = []
    for run in runs + MODELS[1:]:
        rs = [r for r in rows if r['run'] == run]
        values = np.array([float(r['dice']) for r in rs])
        table.append(dict(run='nnunet_s42 (= plain_s42)' if run == 'plain_s42' else run, n=len(rs),
                          mean_dice=float(values.mean()), median_dice=float(np.median(values)),
                          patient_sd=float(values.std(ddof=1)), dice_below_0_1=int((values < .1).sum()),
                          empty_predictions=sum(r['empty_status'] == 'prediction_empty' for r in rs),
                          fp_components=sum(int(r['n_fp_components']) for r in rs),
                          cases_with_fp=sum(int(r['n_fp_components']) > 0 for r in rs)))
    mi.write_table(HERE / 'tables/unchanged_all_runs_summary.csv', table)
    manual = np.array([np.mean([dice[run, p] for run in MANUAL]) for p in patients])
    nnunet = np.array([dice['plain_s42', p] for p in patients])
    contrasts = {'nnU-Net s42 - manual six-run mean': nnunet - manual}
    for seed in ('42', '43', '44'):
        contrasts[f'dropout01_s{seed} - baseline_s{seed}'] = np.array(
            [dice[f'dropout01_s{seed}', p] - dice[f'baseline_s{seed}', p] for p in patients])
    return {name: dict(mean=float(d.mean()), ci95=mi.bootstrap_ci(d), patients_higher=int((d > 0).sum()),
                       patients_lower=int((d < 0).sum())) for name, d in contrasts.items()}


def comparison(table):
    """Internal test next to the existing validation (step A) and MC-RC (step B) results, per model and arm."""
    validation = {(r['condition'], r['input']): r for r in read_rows(EARLIER / 'stress_test/followup_stress_summary.csv')}
    mcrc = {r['subset']: r for r in read_rows(EARLIER / 'mcrc_followup/tables/summary.csv')}
    out = []
    for r in table:
        v = validation[r['condition'], r['input']]
        for unit in MODELS + tuple(ARMS):
            row = dict(condition=r['condition'], input=r['input'], unit=unit,
                       internal_test_mean=r[f'{unit}_mean'], validation_mean=float(v[f'{unit}_mean']))
            if unit in MODELS:
                row.update(internal_test_empty=r[f'{unit}_empty'], validation_empty=int(v[f'{unit}_empty']))
            else:
                row.update(internal_test_empty=r[f'{unit}_empty'])
            if r['condition'] == 'C0' and r['input'] == 'as_stored':
                for subset in ('primary_13_slab', 'all_22'):
                    row[f'mcrc_{subset}_mean'] = float(mcrc[subset][f'{unit}_mean'])
                    row[f'mcrc_{subset}_empty'] = float(mcrc[subset][f'{unit}_empty'])
            out.append(row)
        out.append(dict(condition=r['condition'], input=r['input'], unit='arm_difference',
                        internal_test_mean=r['arm_difference_mean'], validation_mean=float(v['arm_difference_mean']),
                        internal_test_ci95=f"[{r['arm_difference_ci95_low']:.4f}, {r['arm_difference_ci95_high']:.4f}]",
                        validation_ci95=f"[{float(v['arm_difference_ci95_low']):.4f}, {float(v['arm_difference_ci95_high']):.4f}]"))
    mi.write_table(HERE / 'tables/comparison_with_validation_and_mcrc.csv', out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['build', 'score', 'summary'])
    args = parser.parse_args()
    globals()[args.command](args)


if __name__ == '__main__':
    main()
