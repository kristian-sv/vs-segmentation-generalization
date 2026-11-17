#!/usr/bin/env python3
"""Read-only NIfTI case figures using report Dice; numpy, nibabel, matplotlib only.

Example (run from context):
  python make_case_figure.py --csv figure_cases.csv --bundle /path/to/vs_viewer_bundle --out figures --candidates
  python make_case_figure.py --csv figure_cases.csv --bundle /path/to/vs_viewer_bundle --out figures

Numeric slices are zero-based canonical axial indices (viewer indices are one-based).
Canonical reorientation preserves acquisition tilt; no deobliquing or resampling.
The task supplies display defaults; all study metrics are read from report CSVs.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import os
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'thesis-figure-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import nibabel as nib
import numpy as np

GT_COLOR = '#0072B2'
PRED_COLOR = '#E69F00'
CM = 1 / 2.54
DEFAULT_CROP_MM = 60.0
FP_CASES = {'VS-MC-RC-043', 'VS-MC-RC-026', 'VS-MC-RC-057'}
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 8, 'axes.titlesize': 8,
                     'pdf.fonttype': 42, 'ps.fonttype': 42, 'figure.facecolor': 'white',
                     'savefig.facecolor': 'white', 'axes.facecolor': 'white'})


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def unique_index(rows, keys, name):
    result = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        if key in result:
            raise ValueError(f'Duplicate report row in {name}: {key}')
        result[key] = row
    return result


def model_name(model):
    if model == 'nnunet_s42_reoriented':
        return 'nnU-Net\nre-stacked\nseed 42'
    if model == 'nnunet_s42':
        return 'nnU-Net\noriginal\nseed 42'
    match = re.fullmatch(r'(baseline|dropout01)_s(\d+)', model)
    if match:
        # Keep report run identifiers rather than infer architecture/hyperparameters.
        return f"{match[1]}\nseed {match[2]}"
    return model.replace('_', ' ')


@dataclass
class Spec:
    case_id: str
    slice: str
    crop: float | None
    models: list[str]
    title: str


def read_specs(path):
    rows = read_csv(path)
    required = {'case_id', 'slice', 'crop_mm', 'models', 'title'}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f'{path}: require nonempty CSV with columns {sorted(required)}')
    result = []
    seen = set()
    for row in rows:
        case = row['case_id'].strip()
        if not re.fullmatch(r'[A-Za-z0-9_-]+', case):
            raise ValueError(f'Invalid full case ID: {case!r}')
        if case in seen:
            raise ValueError(f'Duplicate case in selection CSV: {case}')
        seen.add(case)
        models = [m.strip() for m in row['models'].split(';') if m.strip()]
        if not models or len(set(models)) != len(models):
            raise ValueError(f'{case}: supply a nonempty, unique model list')
        if any(not re.fullmatch(r'[A-Za-z0-9_-]+', m) for m in models):
            raise ValueError(f'{case}: invalid model identifier')
        raw_crop = row['crop_mm'].strip().lower()
        crop = None if raw_crop == 'full' else float(raw_crop or DEFAULT_CROP_MM)
        if crop is not None and (not np.isfinite(crop) or crop <= 0):
            raise ValueError(f'{case}: crop_mm must be positive or full')
        sl = row['slice'].strip().lower()
        if sl not in ('', 'fp') and not re.fullmatch(r'\d+', sl):
            raise ValueError(f'{case}: slice must be blank, fp, or a zero-based integer')
        result.append(Spec(case, sl, crop, models, row['title'].strip()))
    return result


class Reports:
    def __init__(self, folder):
        self.folder = folder
        self.primary = unique_index(read_csv(folder / 'cases.csv'),
                                    ('cohort', 'case_id', 'run'), 'cases.csv')
        self.restacked = unique_index(read_csv(folder / 'nnunet_reoriented_scores.csv'),
                                      ('case_id',), 'nnunet_reoriented_scores.csv')

    def metric(self, cohort, case, model):
        if model == 'nnunet_s42_reoriented':
            if cohort != 'mcrc':
                raise ValueError(f'{case}: re-stacked report is for MC-RC only')
            row = self.restacked[(case,)]
            status = row['empty_reoriented']
            if status not in ('nonempty', 'prediction_empty'):
                raise ValueError(f'{case}: unknown re-stacked empty status: {status!r}')
            return row['dice_reoriented'], status == 'prediction_empty', 'nnunet_reoriented_scores.csv'
        row = self.primary[(cohort, case, model)]
        status = row['pred_empty'].lower()
        if status not in ('true', 'false'):
            raise ValueError(f'{case}: unknown pred_empty: {status!r}')
        return row['dice'], status == 'true', 'cases.csv'


class Case:
    def __init__(self, spec, bundle, reports):
        self.spec = spec
        matches = sorted((bundle / 'data').glob(f'*/{spec.case_id}'))
        matches = [p for p in matches if p.is_dir()]
        if len(matches) != 1:
            raise ValueError(f'{spec.case_id}: expected one cohort folder, found {matches}')
        self.folder = matches[0]
        self.cohort = self.folder.parent.name
        image = nib.as_closest_canonical(nib.load(self.folder / 'image.nii.gz'))
        if len(image.shape) != 3 or nib.aff2axcodes(image.affine) != ('R', 'A', 'S'):
            raise ValueError(f'{spec.case_id}: require a canonical three-dimensional image')
        self.image = np.asarray(image.dataobj, dtype=np.float32)
        self.spacing = np.linalg.norm(image.affine[:3, :3], axis=0)
        self.gt = self.mask(self.folder / 'gt.nii.gz', image)
        if not np.any(self.gt):
            raise ValueError(f'{spec.case_id}: empty GT cannot define tumour slice/crop')
        self.pred = {}
        self.metrics = {}
        for model in spec.models:
            self.pred[model] = self.mask(self.folder / f'pred_{model}.nii.gz', image)
            metric = reports.metric(self.cohort, spec.case_id, model)
            score = float(metric[0])
            if not np.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f'{spec.case_id}/{model}: missing or invalid reported Dice')
            if metric[1] != (not np.any(self.pred[model])):
                raise ValueError(f'{spec.case_id}/{model}: reported empty status disagrees with mask')
            self.metrics[model] = metric
        voxels = self.image[np.isfinite(self.image) & (self.image != 0)]
        if not len(voxels):
            raise ValueError(f'{spec.case_id}: no finite nonzero image intensities')
        self.vmin, self.vmax = np.percentile(voxels, (1, 99))
        if self.vmax <= self.vmin:
            self.vmax = self.vmin + max(abs(self.vmin) * .01, 1)
        self.gt_area = self.gt.sum(axis=(0, 1))
        self.tumour_slices = np.flatnonzero(self.gt_area)
        self.fp_area = (self.pred[spec.models[0]] & ~self.gt).sum(axis=(0, 1))
        self.notes = set()
        linear = image.affine[:3, :3] / self.spacing
        if not np.allclose(linear, np.eye(3), atol=.01):
            self.notes.add('Canonical axial slices preserve acquisition obliquity; no tilt correction.')

    @staticmethod
    def mask(path, image):
        mask = nib.as_closest_canonical(nib.load(path))
        if mask.shape != image.shape or not np.allclose(mask.affine, image.affine, rtol=0, atol=1e-4):
            raise ValueError(f'{path}: canonical grid differs from image; ask before resampling')
        data = np.asanyarray(mask.dataobj)
        if not np.all(np.isin(data, (0, 1))):
            raise ValueError(f'{path}: expected binary labels; ask which label defines tumour')
        return data == 1

    def selected_slice(self):
        value = self.spec.slice
        if value == 'fp':
            if not np.any(self.fp_area):
                raise ValueError(f'{self.spec.case_id}: first model has no false positives for slice=fp')
            z = int(np.argmax(self.fp_area))
        elif value:
            z = int(value)
        else:
            z = int(np.argmax(self.gt_area))
        if not 0 <= z < self.image.shape[2]:
            raise ValueError(f'{self.spec.case_id}: slice {z} outside canonical volume')
        return z

    def bounds(self, z, full=False):
        sx, sy = self.spacing[:2]
        nx, ny = self.image.shape[:2]
        extent = (-sx/2, (nx-.5)*sx, -sy/2, (ny-.5)*sy)
        if full or self.spec.crop is None:
            return extent, extent
        points = np.argwhere(self.gt[:, :, z])
        if len(points):
            center = points.mean(axis=0) * self.spacing[:2]
        else:
            # FP-only slice has no slice-specific GT centroid. Keep the same GT
            # location by projecting its volume centroid; always disclose this.
            center = np.argwhere(self.gt).mean(axis=0)[:2] * self.spacing[:2]
            self.notes.add(f'Slice {z}: no GT on slice; crop uses projected volume GT centroid.')
        half = self.spec.crop / 2
        bounds = (center[0]-half, center[0]+half, center[1]-half, center[1]+half)
        if len(points) and outside(points, bounds, self.spacing):
            self.notes.add(f'Slice {z}: crop cuts GT; enlarge crop_mm or use full.')
        if bounds[0] < extent[0] or bounds[1] > extent[1] or bounds[2] < extent[2] or bounds[3] > extent[3]:
            self.notes.add(f'Slice {z}: crop extends beyond acquired field; outside field stays white.')
        return bounds, extent

    def score_label(self, model):
        raw, empty, _ = self.metrics[model]
        # Use report data, never overlap-derived Dice. Rounded only for typography;
        # verbatim values and field-level provenance remain in case_notes.md.
        return 'empty' if empty else f'Dice {float(raw):.3f}'


def outside(points, bounds, spacing):
    x, y = points[:, 0] * spacing[0], points[:, 1] * spacing[1]
    x0, x1, y0, y1 = bounds
    return bool(np.any((x < x0) | (x > x1) | (y < y0) | (y > y1)))


def outline(ax, mask, spacing, color, linewidth=1):
    if not np.any(mask):
        return
    # Padding closes contours when a mask reaches an acquired image boundary.
    padded = np.pad(mask.T.astype(np.uint8), 1)
    x = np.arange(-1, mask.shape[0]+1) * spacing[0]
    y = np.arange(-1, mask.shape[1]+1) * spacing[1]
    ax.contour(x, y, padded, levels=[.5], colors=[color], linewidths=linewidth)


def orient_axes(ax, bounds):
    x0, x1, y0, y1 = bounds
    ax.set_xlim(x1, x0)  # RAS x increases towards patient right: display it on left.
    ax.set_ylim(y0, y1)  # RAS y increases anteriorly: display front at top.
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def panel(ax, case, z, model=None, full=False, orient=True):
    bounds, extent = case.bounds(z, full)
    image = np.ma.masked_invalid(case.image[:, :, z].T)
    ax.imshow(image, origin='lower', extent=extent, cmap='gray',
              vmin=case.vmin, vmax=case.vmax, interpolation='nearest', rasterized=True)
    if model:
        outline(ax, case.gt[:, :, z], case.spacing, GT_COLOR)
        outline(ax, case.pred[model][:, :, z], case.spacing, PRED_COLOR)
    orient_axes(ax, bounds)
    if orient:
        for x, label in ((.02, 'R'), (.98, 'L')):
            ax.text(x, .97, label, transform=ax.transAxes, va='top',
                    ha='left' if x < .5 else 'right', color='black', fontsize=8,
                    bbox={'facecolor':'white', 'edgecolor':'none', 'alpha':.85, 'pad':.4})
        ax.text(.5, .97, 'A', transform=ax.transAxes, va='top', ha='center', color='black',
                bbox={'facecolor':'white', 'edgecolor':'none', 'alpha':.85, 'pad':.4})
    if model and bounds != extent:
        points = np.argwhere(case.pred[model][:, :, z])
        if len(points) and outside(points, bounds, case.spacing):
            case.notes.add(f'Slice {z}, {model}: prediction outside crop; whole-slice inset included.')
            inset = ax.inset_axes([.66, .02, .32, .32])
            inset.imshow(image, origin='lower', extent=extent, cmap='gray',
                         vmin=case.vmin, vmax=case.vmax, interpolation='nearest', rasterized=True)
            outline(inset, case.gt[:, :, z], case.spacing, GT_COLOR, .5)
            outline(inset, case.pred[model][:, :, z], case.spacing, PRED_COLOR, .5)
            x0, x1, y0, y1 = bounds
            inset.add_patch(Rectangle((x0,y0), x1-x0, y1-y0, fill=False, edgecolor='white', linewidth=.8))
            orient_axes(inset, extent)
    return ax


def legend(fig):
    bottom = .85 * CM / fig.get_figheight()
    fig.legend(handles=[Line2D([], [], color=GT_COLOR, lw=1, label='GT'),
                        Line2D([], [], color=PRED_COLOR, lw=1, label='Prediction')],
               loc='lower center', bbox_to_anchor=(.53, bottom), ncol=2, frameon=False,
               fontsize=8, handlelength=1.5)
    fig.text(.53, .08 * CM / fig.get_figheight(), 'Dice sources: hpc_report/cases.csv;\nre-stacked: hpc_report/nnunet_reoriented_scores.csv',
             ha='center', va='bottom', fontsize=8)


def render_candidates(case, out):
    # Exactly six evenly spaced positions over the occupied GT range. Very short
    # tumour spans may round to repeated integer slices; labels make this explicit.
    first, last = case.tumour_slices[[0, -1]]
    indices = np.rint(np.linspace(first, last, 6)).astype(int).tolist()
    columns = [(z, f'Slice {z}', False) for z in indices]
    if len(set(indices)) < len(indices):
        case.notes.add('Tumour spans few axial slices; evenly spaced candidates repeat some indices.')
    if case.spec.case_id in FP_CASES or case.spec.slice == 'fp':
        if np.any(case.fp_area):
            z = int(np.argmax(case.fp_area))
            columns.append((z, f'Slice {z} · FP', True))
        else:
            case.notes.add('No false-positive slice for the first listed model.')
    nrows, ncols = len(case.spec.models)+1, len(columns)
    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             figsize=((2.7*ncols+3)*CM, (3*nrows+2)*CM))
    fig.subplots_adjust(left=.14, right=.995, top=.86, bottom=.13, wspace=.07, hspace=.15)
    for r in range(nrows):
        model = None if r == 0 else case.spec.models[r-1]
        for c, (z, title, full) in enumerate(columns):
            panel(axes[r,c], case, z, model, full=full, orient=r==0)
            if r == 0:
                axes[r,c].set_title(title, pad=5)
        label = 'Image' if model is None else model_name(model)+'\n'+case.score_label(model)
        pos = axes[r,0].get_position()
        fig.text(.01, (pos.y0+pos.y1)/2, label, va='center', ha='left', fontsize=8)
    fig.suptitle(f'{case.spec.case_id} — {case.spec.title}\n'
                 'Canonical axial indices are zero-based · FP column uses whole slice',
                 fontsize=8, y=.98)
    legend(fig)
    target = out / 'candidates' / f'{case.spec.case_id}.png'
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=180, facecolor='white')
    plt.close(fig)
    print(f'{target}\n  candidate indices: {indices}', flush=True)


def render_final(cases, out):
    nrows, ncols = len(cases), 1 + max(len(c.spec.models) for c in cases)
    width_cm, left_cm, right_cm, gap_cm = 15, 1.8, .1, .2
    panel_cm = (width_cm-left_cm-right_cm-gap_cm*(ncols-1))/ncols
    row_cm, footer_cm, top_cm = panel_cm+1.4, 1.6, .2
    height_cm = nrows*row_cm+footer_cm+top_cm
    fig = plt.figure(figsize=(width_cm*CM, height_cm*CM))
    for r, case in enumerate(cases):
        z = case.selected_slice()
        y_cm = height_cm-top_cm-r*row_cm-panel_cm
        for c, model in enumerate([None]+case.spec.models):
            x_cm = left_cm+c*(panel_cm+gap_cm)
            ax = fig.add_axes([x_cm/width_cm, y_cm/height_cm,
                               panel_cm/width_cm, panel_cm/height_cm])
            panel(ax,case,z,model)
            label = 'Image' if model is None else model_name(model)+'\n'+case.score_label(model)
            fig.text((x_cm+panel_cm/2)/width_cm, (y_cm-.15)/height_cm,
                     label,ha='center',va='top',fontsize=8)
            if c == 0:
                row_id=case.spec.case_id.rsplit('-',1)
                row_label=row_id[0]+'-\n'+row_id[1]+'\n'+wrap_words(case.spec.title,10)
                fig.text(.05/width_cm,(y_cm+panel_cm/2)/height_cm,
                         row_label,ha='left',va='center',fontsize=8)
        print(f'{case.spec.case_id}: canonical slice {z}',flush=True)
    legend(fig)
    out.mkdir(parents=True,exist_ok=True)
    # Do not use bbox_inches=tight: the exported PDF must remain exactly 15 cm wide.
    fig.savefig(out/'cases_figure.pdf',dpi=300,facecolor='white')
    fig.savefig(out/'cases_figure.png',dpi=300,facecolor='white')
    plt.close(fig)
    print(f'Saved {out / "cases_figure.pdf"} and {out / "cases_figure.png"}',flush=True)


def wrap_words(text, width):
    import textwrap
    return '\n'.join(textwrap.wrap(text,width=width,break_long_words=False))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--csv',type=Path,required=True,help='Selection CSV; slice blank, fp or zero-based integer')
    parser.add_argument('--bundle',type=Path,required=True,help='Read-only bundle containing data/<cohort>/<case>')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--reports',type=Path,help='Defaults to hpc_report beside the selection CSV')
    parser.add_argument('--candidates',action='store_true',help='Write previews only; never final PDF/PNG')
    args=parser.parse_args(argv)
    bundle=args.bundle.expanduser().resolve()
    selection=args.csv.expanduser().resolve()
    out=args.out.expanduser().resolve()
    reports_path=(args.reports or selection.parent/'hpc_report').expanduser().resolve()
    if out == bundle or bundle in out.parents:
        parser.error('--out must be outside the read-only bundle')
    if out == reports_path or reports_path in out.parents:
        parser.error('--out must be outside the source report directory')
    try:
        specs=read_specs(selection)
        reports=Reports(reports_path)
        cases=[]
        # Validate every requested file/report before writing any figure.
        for spec in specs:
            case=Case(spec,bundle,reports)
            case.selected_slice()
            cases.append(case)
        if args.candidates:
            for case in cases:
                render_candidates(case,out)
        else:
            render_final(cases,out)
        for case in cases:
            for note in sorted(case.notes):
                print(f'NOTE {case.spec.case_id}: {note}',file=sys.stderr)
    except (OSError,ValueError,KeyError) as exc:
        parser.exit(1,f'Cannot render: {exc}\nCheck the source files and selections; no input files were changed.\n')


if __name__=='__main__':
    main()
