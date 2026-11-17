"""Make the three figures shown in the README.

  readme_banner.png    (a) whole axial slice of patient VS-MC-RC-098 with the
                       tumor region boxed, (b, c) that region with the plain
                       and the augmented nnU-Net (seed 42), (d) per-patient
                       arm means of the 13 primary slab cases.
  cases_followup.png   the five patients of the thesis case figure
                       (mcrc_followup/figures/cases_followup.png), same slices
                       and models, drawn with analysis/make_case_figure.py.
  stress_test.png      display copy of stress_test/stress_test.png.

These are display versions in one blue/red palette. The thesis versions under
analysis/reports/ use blue and orange and are left unchanged. All numbers are
read from mcrc_followup/tables/per_case.csv.

The first two figures read images, reference masks and predictions from the
local viewer bundle, which is not distributed (pass --bundle to point to it).
The stress-test copy needs only the stored PNG. Requires nibabel.

Run from the repository root:  python docs/make_readme_figures.py
"""
import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import matplotlib.font_manager as font_manager
import matplotlib.image as mpimg
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'analysis/reports/model_insight_20261012'
DOCS = ROOT / 'docs'
NAVY = '#001D3D'      # text, axes and the manual U-Net lines
SKY = '#5AA9E6'       # reference outline: navy is not visible on a dark scan
RED = '#A8101B'       # predictions and nnU-Net lines
MUTED, FAINT = '#51606F', '#B8BEC6'
ROWS = (('VS-MC-RC-098', '7'), ('VS-MC-RC-081', '6'), ('VS-MC-RC-026', '8'), ('VS-MC-RC-057', '8'),
        ('VS-MC-RC-016', ''))            # as FIGURE_ROWS in analysis/model_insight.py
MODELS = (('plain_s42', 'Plain nnU-Net'), ('slabthick_s42', 'Augmented nnU-Net'))


def load_helper():
    spec = importlib.util.spec_from_file_location('make_case_figure', ROOT / 'analysis/make_case_figure.py')
    helper = sys.modules[spec.name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.GT_COLOR, helper.PRED_COLOR = SKY, RED
    for path in sorted(Path('/usr/share/fonts/truetype').glob('LiberationSans-*.ttf')):
        font_manager.fontManager.addfont(str(path))
    helper.plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Liberation Sans', 'Arial', 'Helvetica', 'DejaVu Sans'],
        'axes.linewidth': 0.8, 'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
        'xtick.major.size': 3, 'ytick.major.size': 3,
    })
    return helper


def read_table():
    with open(REPORT / 'mcrc_followup/tables/per_case.csv', newline='') as handle:
        return {row['case_id']: row for row in csv.DictReader(handle)}


def load_case(helper, bundle, table, case_id, slice_index):
    class Reports:       # the interface make_case_figure.Case expects
        @staticmethod
        def metric(_cohort, case, model):
            return table[case][f'dice_{model}'], table[case][f'empty_{model}'] == 'True', 'per_case.csv'
    spec = helper.Spec(case_id, slice_index, helper.DEFAULT_CROP_MM, tuple(m for m, _ in MODELS), '')
    return helper.Case(spec, bundle, Reports)


def score(table, case_id, model):
    row = table[case_id]
    return 'No tumor predicted' if row[f'empty_{model}'] == 'True' else f"Dice {float(row[f'dice_{model}']):.3f}"


def draw(helper, axis, case, z, model=None, full=False, orient=False):
    helper.panel(axis, case, z, model, full=full, orient=orient)
    for collection, width in zip(axis.collections, (2.0, 1.2)):   # reference under, prediction on top
        collection.set_linewidth(width)


def scale_bar(axis, bounds, length_mm, label):
    x0, _x1, y0, y1 = bounds
    margin = 0.05 * (y1 - y0)
    axis.plot([x0 + margin, x0 + margin + length_mm], [y0 + margin, y0 + margin], color='white', linewidth=2,
              solid_capstyle='butt')
    axis.text(x0 + margin + length_mm / 2, y0 + 1.6 * margin, label, color='white', fontsize=8, ha='center',
              va='bottom')


def outline_legend(figure, x, y, width):
    for offset, color, label in ((0.0, SKY, 'Reference'), (1.05, RED, 'Prediction')):
        start = (x + offset) / width
        figure.add_artist(Line2D([start, start + 0.26 / width], [y, y], color=color, linewidth=2))
        figure.text(start + 0.33 / width, y, label, fontsize=9.5, color=MUTED, va='center')


def banner(helper, bundle, table):
    plt = helper.plt
    case = load_case(helper, bundle, table, *ROWS[0])
    z = case.selected_slice()
    crop, extent = case.bounds(z)
    slabs = [row for row in table.values() if row['primary_slab_case'] == 'True']
    assert len(slabs) == 13
    plain = np.array([float(row['dice_plain_arm']) for row in slabs])
    augmented = np.array([float(row['dice_augmented_arm']) for row in slabs])

    width, height, side = 12.0, 3.55, 2.55
    bottom, left, gap = 0.52, 0.12, 0.16
    top = (bottom + side + 0.09) / height
    figure = plt.figure(figsize=(width, height), dpi=200, facecolor='white')
    panels = (('a', 'Routine T1 MRI of the head', 'Patient VS-MC-RC-098; box: tumor region', None),
              ('b', MODELS[0][1], score(table, case.spec.case_id, MODELS[0][0]), MODELS[0][0]),
              ('c', MODELS[1][1], score(table, case.spec.case_id, MODELS[1][0]), MODELS[1][0]))
    for index, (letter, title, note, model) in enumerate(panels):
        x = left + index * (side + gap)
        axis = figure.add_axes([x / width, bottom / height, side / width, side / height])
        if model is None:
            draw(helper, axis, case, z, full=True, orient=True)
            x0, x1, y0, y1 = crop
            axis.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor='white', linewidth=1.3))
            scale_bar(axis, extent, 50, '5 cm')
        else:
            draw(helper, axis, case, z, model)
            if index == 1:
                scale_bar(axis, crop, 10, '1 cm')
        figure.text(x / width, top, letter, fontsize=12, fontweight='bold', color=NAVY, va='bottom')
        figure.text((x + 0.2) / width, top, title, fontsize=11, color=NAVY, va='bottom')
        figure.text(x / width, (bottom - 0.1) / height, note, fontsize=9.5, color=MUTED, va='top')
    outline_legend(figure, left + side + gap, 0.1 / height, width)

    x = left + 3 * (side + gap) + 0.62
    axis = figure.add_axes([x / width, (bottom + 0.02) / height, (width - x - 0.55) / width, (side - 0.02) / height])
    for a, b in zip(plain, augmented):
        axis.plot([0, 1], [a, b], color=FAINT, linewidth=1, marker='o', markersize=3.2, zorder=2, clip_on=False)
    axis.plot([0, 1], [plain.mean(), augmented.mean()], color=NAVY, linewidth=2.4, marker='o', markersize=6.5,
              zorder=3, clip_on=False)
    for position, value, shift, align in ((0, plain.mean(), -9, 'right'), (1, augmented.mean(), 9, 'left')):
        axis.annotate(f'{value:.3f}', (position, value), xytext=(shift, 0), textcoords='offset points', ha=align,
                      va='center', fontsize=10.5, fontweight='bold', color=NAVY)
    axis.set_xlim(-0.42, 1.42)
    axis.set_ylim(0, 1)
    axis.set_xticks([0, 1])
    axis.set_xticklabels(['Plain', 'Augmented'], fontsize=10, color=NAVY)
    axis.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axis.tick_params(axis='y', labelsize=9, colors=NAVY)
    axis.tick_params(axis='x', length=0, pad=6)
    axis.set_ylabel('Dice per patient', fontsize=10, color=NAVY)
    axis.spines['left'].set_color(NAVY)
    for name in ('top', 'right', 'bottom'):
        axis.spines[name].set_visible(False)
    figure.text((x - 0.5) / width, top, 'd', fontsize=12, fontweight='bold', color=NAVY, va='bottom')
    figure.text((x - 0.3) / width, top, '13 routine slab scans', fontsize=11, color=NAVY, va='bottom')
    figure.text((x - 0.5) / width, 0.1 / height, 'Grey: one patient.  Dark line: mean.', fontsize=9.5, color=MUTED,
                va='center')
    figure.savefig(DOCS / 'readme_banner.png', facecolor='white')
    plt.close(figure)


def case_figure(helper, bundle, table):
    plt = helper.plt
    side, gap_x, gap_y, left, head, foot = 2.05, 0.1, 0.42, 1.15, 0.38, 0.5
    width = left + 3 * side + 2 * gap_x + 0.1
    height = head + len(ROWS) * (side + gap_y) + foot
    figure = plt.figure(figsize=(width, height), dpi=200, facecolor='white')
    for column, title in enumerate(('T1 MRI',) + tuple(label for _, label in MODELS)):
        figure.text((left + column * (side + gap_x) + side / 2) / width, 1 - 0.24 / height, title, fontsize=10.5,
                    color=NAVY, ha='center', va='center')
    for row, (case_id, slice_index) in enumerate(ROWS):
        case = load_case(helper, bundle, table, case_id, slice_index)
        z = case.selected_slice()
        y = height - head - (row + 1) * (side + gap_y) + gap_y
        figure.text(0.08 / width, (y + side / 2) / height, case_id.replace('VS-MC-RC-', 'VS-MC-RC-\n'), fontsize=9.5,
                    color=NAVY, va='center', linespacing=1.3)
        for column, model in enumerate((None,) + tuple(m for m, _ in MODELS)):
            x = left + column * (side + gap_x)
            axis = figure.add_axes([x / width, y / height, side / width, side / height])
            draw(helper, axis, case, z, model, orient=model is None)
            if model:
                figure.text((x + side / 2) / width, (y - 0.08) / height, score(table, case_id, model), fontsize=9,
                            color=MUTED, ha='center', va='top')
    outline_legend(figure, left + side + gap_x, 0.2 / height, width)
    figure.savefig(DOCS / 'cases_followup.png', facecolor='white')
    plt.close(figure)


def recolor(image, replacements):
    """Show stored line colours in the README palette.

    A pixel on an anti-aliased line is a mix of the line colour and a neutral
    background. Its share of the line colour is its chroma relative to the
    pure colour; that share of (new - old) is added.
    """
    rgb = image[..., :3].astype(float)
    hsv = mcolors.rgb_to_hsv(rgb)
    chroma = rgb.max(-1) - rgb.min(-1)
    out = rgb.copy()
    for old, new in replacements:
        old_rgb, new_rgb = np.array(mcolors.to_rgb(old)), np.array(mcolors.to_rgb(new))
        distance = np.abs((hsv[..., 0] - mcolors.rgb_to_hsv(old_rgb)[0] + 0.5) % 1 - 0.5)
        share = np.where((distance < 0.06) & (chroma > 0.03), np.clip(chroma / np.ptp(old_rgb), 0, 1), 0)
        out += share[..., None] * (new_rgb - old_rgb)
    result = image.copy()
    result[..., :3] = np.clip(out, 0, 1)
    return result


def stress_copy():
    stored = mpimg.imread(REPORT / 'stress_test/stress_test.png')
    mpimg.imsave(DOCS / 'stress_test.png', recolor(stored, (('#2A78D6', NAVY), ('#EB6834', RED))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--bundle', type=Path, default=REPORT / 'viewer_bundle',
                        help='local viewer bundle with data/mcrc/<patient>/ image, reference and predictions')
    arguments = parser.parse_args()
    stress_copy()
    module = load_helper()
    scores = read_table()
    banner(module, arguments.bundle, scores)
    case_figure(module, arguments.bundle, scores)
