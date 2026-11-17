"""Make the figures shown in the README from records already in this repository.

No image volumes are read. Three files are written to docs/:

  readme_banner.png    panels a-c are cropped from the first row of
                       mcrc_followup/figures/cases_followup.png (patient
                       VS-MC-RC-098); panel d plots the per-patient arm means
                       of the 13 primary slab cases in
                       mcrc_followup/tables/per_case.csv.
  cases_followup.png   display copy of the case figure.
  stress_test.png      display copy of the stress-test figure.

In all three, the orange used for predictions and for nnU-Net in the stored
figures is shown in red. Only that display colour differs; the stored figures
under analysis/reports/ are the thesis versions and are left unchanged.

Run from the repository root:  python docs/make_readme_figures.py
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as font_manager
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'analysis/reports/model_insight_20261012'
DOCS = ROOT / 'docs'
BLUE, RED, INK, MUTED, FAINT = '#0072B2', '#D62728', '#1A1A1A', '#555555', '#B8BCC2'
RED_RGB = np.array(matplotlib.colors.to_rgb(RED))

for path in sorted(Path('/usr/share/fonts/truetype').glob('LiberationSans-*.ttf')):
    font_manager.fontManager.addfont(str(path))
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Liberation Sans', 'Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'xtick.major.size': 3, 'ytick.major.size': 3,
})


def orange_to_red(image, orange):
    """Return the image with one stored orange shown in red.

    A pixel on an anti-aliased orange line is a mix of that orange and a
    neutral background. Its orange share is its chroma relative to the pure
    colour, and that share of (red - orange) is added.
    """
    rgb = image[..., :3].astype(float)
    source = np.array(matplotlib.colors.to_rgb(orange))
    high, low = rgb.max(-1), rgb.min(-1)
    chroma = high - low
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    is_orange = (chroma > 0.03) & (red >= green) & (green >= blue) & (red - green < 0.75 * chroma + 0.02)
    share = np.where(is_orange, np.clip(chroma / (source.max() - source.min()), 0, 1), 0)
    out = image.copy()
    out[..., :3] = np.clip(rgb + share[..., None] * (RED_RGB - source), 0, 1)
    return out


def display_copy(source, name, orange):
    mpimg.imsave(DOCS / name, orange_to_red(mpimg.imread(source), orange))


def banner():
    cases = orange_to_red(mpimg.imread(REPORT / 'mcrc_followup/figures/cases_followup.png'), '#E69F00')
    rows, columns = slice(23, 523), (slice(213, 713), slice(736, 1236), slice(1260, 1760))
    panels = (('a', 'Routine T1 MRI', 'VS-MC-RC-098, axial slab'),
              ('b', 'Plain nnU-Net', 'No tumor predicted'),
              ('c', 'Augmented nnU-Net', 'Dice 0.914'))
    with open(REPORT / 'mcrc_followup/tables/per_case.csv', newline='') as handle:
        slabs = [row for row in csv.DictReader(handle) if row['primary_slab_case'] == 'True']
    plain = np.array([float(row['dice_plain_arm']) for row in slabs])
    augmented = np.array([float(row['dice_augmented_arm']) for row in slabs])
    assert len(slabs) == 13

    width, height = 12.0, 3.55
    figure = plt.figure(figsize=(width, height), dpi=200, facecolor='white')
    side = 2.55                      # panel edge in inches
    bottom, left, gap = 0.52, 0.12, 0.16
    for index, (columns_, (letter, title, note)) in enumerate(zip(columns, panels)):
        x = left + index * (side + gap)
        axis = figure.add_axes([x / width, bottom / height, side / width, side / height])
        axis.imshow(cases[rows, columns_], interpolation='lanczos')
        axis.set_xticks([]); axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
        top = (bottom + side + 0.09) / height
        figure.text(x / width, top, letter, fontsize=12, fontweight='bold', color=INK, va='bottom')
        figure.text((x + 0.2) / width, top, title, fontsize=11, color=INK, va='bottom')
        figure.text(x / width, (bottom - 0.1) / height, note, fontsize=9.5, color=MUTED, va='top')

    legend_y = 0.1 / height
    for offset, color, label in ((0.0, BLUE, 'Reference'), (1.05, RED, 'Prediction')):
        x = left + side + gap + offset   # under panel b, where the outlines first appear
        figure.add_artist(Line2D([x / width, (x + 0.26) / width], [legend_y, legend_y], color=color, linewidth=2))
        figure.text((x + 0.33) / width, legend_y, label, fontsize=9.5, color=MUTED, va='center')

    x = left + 3 * (side + gap) + 0.62
    plot_width = width - x - 0.55
    axis = figure.add_axes([x / width, (bottom + 0.02) / height, plot_width / width, (side - 0.02) / height])
    for a, b in zip(plain, augmented):
        axis.plot([0, 1], [a, b], color=FAINT, linewidth=1, marker='o', markersize=3.2, zorder=2, clip_on=False)
    axis.plot([0, 1], [plain.mean(), augmented.mean()], color=INK, linewidth=2.2, marker='o', markersize=6.5,
              zorder=3, clip_on=False)
    axis.annotate(f'{plain.mean():.3f}', (0, plain.mean()), xytext=(-9, 0), textcoords='offset points',
                  ha='right', va='center', fontsize=10.5, fontweight='bold', color=INK)
    axis.annotate(f'{augmented.mean():.3f}', (1, augmented.mean()), xytext=(9, 0), textcoords='offset points',
                  ha='left', va='center', fontsize=10.5, fontweight='bold', color=INK)
    axis.set_xlim(-0.42, 1.42); axis.set_ylim(0, 1)
    axis.set_xticks([0, 1]); axis.set_xticklabels(['Plain', 'Augmented'], fontsize=10, color=INK)
    axis.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axis.tick_params(axis='y', labelsize=9, colors=INK)
    axis.tick_params(axis='x', length=0, pad=6)
    axis.set_ylabel('Dice per patient', fontsize=10, color=INK)
    for name in ('top', 'right', 'bottom'):
        axis.spines[name].set_visible(False)
    top = (bottom + side + 0.09) / height
    figure.text((x - 0.5) / width, top, 'd', fontsize=12, fontweight='bold', color=INK, va='bottom')
    figure.text((x - 0.3) / width, top, '13 routine slab scans', fontsize=11, color=INK, va='bottom')
    figure.text((x - 0.5) / width, legend_y, 'Grey: one patient.  Black: mean.', fontsize=9.5, color=MUTED, va='center')
    figure.savefig(DOCS / 'readme_banner.png', facecolor='white')
    plt.close(figure)


if __name__ == '__main__':
    banner()
    display_copy(REPORT / 'mcrc_followup/figures/cases_followup.png', 'cases_followup.png', '#E69F00')
    display_copy(REPORT / 'stress_test/stress_test.png', 'stress_test.png', '#EB6834')
