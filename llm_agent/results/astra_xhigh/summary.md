# Astra xhigh: 22 MC-RC patients

Model: `gpt-6-astra`. One independent session per patient. Original Opus prompt, toolbox, image sandbox, case order, and 40-call / 20-image / 60-minute budgets. Invalid or unsaved masks count as empty predictions. Same original Dice, volume-error and HD95 implementations.

Bootstrap intervals are descriptive: 10,000 patient resamples, seed 42. The prior pilot/development exposure to patient 028 remains a limitation. Model-provider image encoding and system/developer message transport differ between Claude and Codex; image bytes, prompt text, and tool semantics are preserved.

## All patients (n=22)

| Source | Mean Dice | Median Dice | Empty outputs |
|---|---:|---:|---:|
| Astra xhigh | 0.854 | 0.915 | 0/22 |
| Opus 5.5 high | 0.778 | 0.876 | 0/22 |
| plain arm | 0.546 | 0.621 | 10/44 |
| augmented arm | 0.761 | 0.851 | 1/44 |
| nnU-Net initial | 0.483 | 0.679 | 7/22 |

| Paired difference (Astra minus) | Mean | 95% CI |
|---|---:|---|
| Opus 5.5 high | +0.076 | [-0.030, +0.179] |
| plain arm | +0.307 | [+0.172, +0.452] |
| augmented arm | +0.092 | [+0.035, +0.162] |
| nnU-Net initial | +0.371 | [+0.220, +0.529] |

## Primary slab cases (13) (n=13)

| Source | Mean Dice | Median Dice | Empty outputs |
|---|---:|---:|---:|
| Astra xhigh | 0.872 | 0.914 | 0/13 |
| Opus 5.5 high | 0.773 | 0.845 | 0/13 |
| plain arm | 0.364 | 0.379 | 10/26 |
| augmented arm | 0.728 | 0.834 | 1/26 |
| nnU-Net initial | 0.335 | 0.000 | 7/13 |

| Paired difference (Astra minus) | Mean | 95% CI |
|---|---:|---|
| Opus 5.5 high | +0.099 | [+0.038, +0.180] |
| plain arm | +0.508 | [+0.337, +0.670] |
| augmented arm | +0.144 | [+0.052, +0.246] |
| nnU-Net initial | +0.536 | [+0.342, +0.727] |

## axial whole-head (n=7)

| Source | Mean Dice | Median Dice | Empty outputs |
|---|---:|---:|---:|
| Astra xhigh | 0.917 | 0.925 | 0/7 |
| Opus 5.5 high | 0.876 | 0.878 | 0/7 |
| plain arm | 0.898 | 0.896 | 0/14 |
| augmented arm | 0.897 | 0.894 | 0/14 |
| nnU-Net initial | 0.892 | 0.899 | 0/7 |

| Paired difference (Astra minus) | Mean | 95% CI |
|---|---:|---|
| Opus 5.5 high | +0.041 | [+0.015, +0.066] |
| plain arm | +0.019 | [+0.002, +0.035] |
| augmented arm | +0.020 | [+0.004, +0.035] |
| nnU-Net initial | +0.024 | [+0.004, +0.043] |

## axial slab (n=11)

| Source | Mean Dice | Median Dice | Empty outputs |
|---|---:|---:|---:|
| Astra xhigh | 0.858 | 0.899 | 0/11 |
| Opus 5.5 high | 0.745 | 0.841 | 0/11 |
| plain arm | 0.347 | 0.379 | 9/22 |
| augmented arm | 0.731 | 0.834 | 1/22 |
| nnU-Net initial | 0.342 | 0.000 | 6/11 |

| Paired difference (Astra minus) | Mean | 95% CI |
|---|---:|---|
| Opus 5.5 high | +0.113 | [+0.043, +0.204] |
| plain arm | +0.511 | [+0.350, +0.663] |
| augmented arm | +0.128 | [+0.034, +0.238] |
| nnU-Net initial | +0.517 | [+0.310, +0.727] |

## non-axial (n=4)

| Source | Mean Dice | Median Dice | Empty outputs |
|---|---:|---:|---:|
| Astra xhigh | 0.730 | 0.860 | 0/4 |
| Opus 5.5 high | 0.695 | 0.918 | 0/4 |
| plain arm | 0.477 | 0.499 | 1/8 |
| augmented arm | 0.608 | 0.648 | 0/8 |
| nnU-Net initial | 0.153 | 0.007 | 1/4 |

| Paired difference (Astra minus) | Mean | 95% CI |
|---|---:|---|
| Opus 5.5 high | +0.034 | [-0.508, +0.594] |
| plain arm | +0.253 | [+0.016, +0.708] |
| augmented arm | +0.122 | [+0.012, +0.333] |
| nnU-Net initial | +0.577 | [+0.293, +0.860] |

## Per-patient Dice

| Patient | Stratum | Astra xhigh | Opus 5.5 high | plain arm | augmented arm | nnU-Net initial |
|---|---|---:|---:|---:|---:|---:|
| VS-MC-RC-009 | axial slab | 0.933 | 0.929 | 0.458 | 0.937 | 0.000 |
| VS-MC-RC-012 | non-axial | 0.958 | 0.947 | 0.020 | 0.518 | 0.600 |
| VS-MC-RC-016 | axial slab | 0.479 | 0.000 | 0.000 | 0.000 | 0.000 |
| VS-MC-RC-021 | axial whole-head | 0.935 | 0.882 | 0.896 | 0.892 | 0.883 |
| VS-MC-RC-024 | axial slab | 0.963 | 0.951 | 0.379 | 0.779 | 0.758 |
| VS-MC-RC-026 | axial slab | 0.890 | 0.899 | 0.000 | 0.723 | 0.000 |
| VS-MC-RC-028 | axial whole-head | 0.915 | 0.860 | 0.926 | 0.922 | 0.926 |
| VS-MC-RC-030 | axial slab | 0.925 | 0.845 | 0.228 | 0.840 | 0.456 |
| VS-MC-RC-043 | non-axial | 0.788 | 0.000 | 0.770 | 0.778 | 0.000 |
| VS-MC-RC-044 | axial whole-head | 0.892 | 0.905 | 0.847 | 0.846 | 0.837 |
| VS-MC-RC-045 | axial slab | 0.914 | 0.883 | 0.890 | 0.855 | 0.877 |
| VS-MC-RC-057 | axial slab | 0.916 | 0.781 | 0.473 | 0.873 | 0.859 |
| VS-MC-RC-060 | non-axial | 0.242 | 0.926 | 0.227 | 0.228 | 0.013 |
| VS-MC-RC-063 | axial whole-head | 0.884 | 0.875 | 0.894 | 0.891 | 0.899 |
| VS-MC-RC-076 | axial slab | 0.899 | 0.632 | 0.150 | 0.458 | 0.000 |
| VS-MC-RC-081 | axial slab | 0.800 | 0.685 | 0.000 | 0.834 | 0.000 |
| VS-MC-RC-082 | axial whole-head | 0.937 | 0.908 | 0.915 | 0.915 | 0.910 |
| VS-MC-RC-085 | axial whole-head | 0.925 | 0.878 | 0.919 | 0.917 | 0.909 |
| VS-MC-RC-097 | axial slab | 0.856 | 0.749 | 0.818 | 0.828 | 0.808 |
| VS-MC-RC-098 | axial slab | 0.867 | 0.841 | 0.426 | 0.910 | 0.000 |
| VS-MC-RC-099 | axial whole-head | 0.929 | 0.826 | 0.887 | 0.894 | 0.883 |
| VS-MC-RC-100 | non-axial | 0.932 | 0.909 | 0.891 | 0.908 | 0.000 |
