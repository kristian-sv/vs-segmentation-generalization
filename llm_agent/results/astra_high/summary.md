# Astra high: 22 MC-RC patients

Model: `gpt-6-astra`. One independent session per patient. Original Opus prompt, toolbox, image sandbox, case order, and 40-call / 20-image / 60-minute budgets. Invalid or unsaved masks count as empty predictions. Same original Dice, volume-error and HD95 implementations.

Bootstrap intervals are descriptive: 10,000 patient resamples, seed 42. The prior pilot/development exposure to patient 028 remains a limitation. Model-provider image encoding and system/developer message transport differ between Claude and Codex; image bytes, prompt text, and tool semantics are preserved.

## All patients (n=22)

| Source | Mean Dice | Median Dice | Empty outputs |
|---|---:|---:|---:|
| Astra high | 0.845 | 0.907 | 0/22 |
| Opus 5.5 high | 0.778 | 0.876 | 0/22 |
| plain arm | 0.546 | 0.621 | 10/44 |
| augmented arm | 0.761 | 0.851 | 1/44 |
| nnU-Net initial | 0.483 | 0.679 | 7/22 |

| Paired difference (Astra minus) | Mean | 95% CI |
|---|---:|---|
| Opus 5.5 high | +0.068 | [-0.035, +0.165] |
| plain arm | +0.299 | [+0.164, +0.444] |
| augmented arm | +0.084 | [+0.031, +0.148] |
| nnU-Net initial | +0.363 | [+0.211, +0.521] |

## Primary slab cases (13) (n=13)

| Source | Mean Dice | Median Dice | Empty outputs |
|---|---:|---:|---:|
| Astra high | 0.865 | 0.904 | 0/13 |
| Opus 5.5 high | 0.773 | 0.845 | 0/13 |
| plain arm | 0.364 | 0.379 | 10/26 |
| augmented arm | 0.728 | 0.834 | 1/26 |
| nnU-Net initial | 0.335 | 0.000 | 7/13 |

| Paired difference (Astra minus) | Mean | 95% CI |
|---|---:|---|
| Opus 5.5 high | +0.092 | [+0.039, +0.157] |
| plain arm | +0.501 | [+0.334, +0.662] |
| augmented arm | +0.137 | [+0.055, +0.229] |
| nnU-Net initial | +0.530 | [+0.334, +0.722] |

## axial whole-head (n=7)

| Source | Mean Dice | Median Dice | Empty outputs |
|---|---:|---:|---:|
| Astra high | 0.909 | 0.909 | 0/7 |
| Opus 5.5 high | 0.876 | 0.878 | 0/7 |
| plain arm | 0.898 | 0.896 | 0/14 |
| augmented arm | 0.897 | 0.894 | 0/14 |
| nnU-Net initial | 0.892 | 0.899 | 0/7 |

| Paired difference (Astra minus) | Mean | 95% CI |
|---|---:|---|
| Opus 5.5 high | +0.032 | [+0.013, +0.053] |
| plain arm | +0.011 | [-0.009, +0.032] |
| augmented arm | +0.012 | [-0.008, +0.033] |
| nnU-Net initial | +0.016 | [-0.008, +0.041] |

## axial slab (n=11)

| Source | Mean Dice | Median Dice | Empty outputs |
|---|---:|---:|---:|
| Astra high | 0.851 | 0.893 | 0/11 |
| Opus 5.5 high | 0.745 | 0.841 | 0/11 |
| plain arm | 0.347 | 0.379 | 9/22 |
| augmented arm | 0.731 | 0.834 | 1/22 |
| nnU-Net initial | 0.342 | 0.000 | 6/11 |

| Paired difference (Astra minus) | Mean | 95% CI |
|---|---:|---|
| Opus 5.5 high | +0.106 | [+0.046, +0.178] |
| plain arm | +0.503 | [+0.345, +0.657] |
| augmented arm | +0.120 | [+0.039, +0.216] |
| nnU-Net initial | +0.509 | [+0.299, +0.724] |

## non-axial (n=4)

| Source | Mean Dice | Median Dice | Empty outputs |
|---|---:|---:|---:|
| Astra high | 0.719 | 0.847 | 0/4 |
| Opus 5.5 high | 0.695 | 0.918 | 0/4 |
| plain arm | 0.477 | 0.499 | 1/8 |
| augmented arm | 0.608 | 0.648 | 0/8 |
| nnU-Net initial | 0.153 | 0.007 | 1/4 |

| Paired difference (Astra minus) | Mean | 95% CI |
|---|---:|---|
| Opus 5.5 high | +0.024 | [-0.512, +0.569] |
| plain arm | +0.242 | [-0.002, +0.697] |
| augmented arm | +0.111 | [-0.007, +0.324] |
| nnU-Net initial | +0.566 | [+0.284, +0.847] |

## Per-patient Dice

| Patient | Stratum | Astra high | Opus 5.5 high | plain arm | augmented arm | nnU-Net initial |
|---|---|---:|---:|---:|---:|---:|
| VS-MC-RC-009 | axial slab | 0.935 | 0.929 | 0.458 | 0.937 | 0.000 |
| VS-MC-RC-012 | non-axial | 0.947 | 0.947 | 0.020 | 0.518 | 0.600 |
| VS-MC-RC-016 | axial slab | 0.379 | 0.000 | 0.000 | 0.000 | 0.000 |
| VS-MC-RC-021 | axial whole-head | 0.926 | 0.882 | 0.896 | 0.892 | 0.883 |
| VS-MC-RC-024 | axial slab | 0.962 | 0.951 | 0.379 | 0.779 | 0.758 |
| VS-MC-RC-026 | axial slab | 0.886 | 0.899 | 0.000 | 0.723 | 0.000 |
| VS-MC-RC-028 | axial whole-head | 0.902 | 0.860 | 0.926 | 0.922 | 0.926 |
| VS-MC-RC-030 | axial slab | 0.910 | 0.845 | 0.228 | 0.840 | 0.456 |
| VS-MC-RC-043 | non-axial | 0.758 | 0.000 | 0.770 | 0.778 | 0.000 |
| VS-MC-RC-044 | axial whole-head | 0.909 | 0.905 | 0.847 | 0.846 | 0.837 |
| VS-MC-RC-045 | axial slab | 0.923 | 0.883 | 0.890 | 0.855 | 0.877 |
| VS-MC-RC-057 | axial slab | 0.904 | 0.781 | 0.473 | 0.873 | 0.859 |
| VS-MC-RC-060 | non-axial | 0.235 | 0.926 | 0.227 | 0.228 | 0.013 |
| VS-MC-RC-063 | axial whole-head | 0.873 | 0.875 | 0.894 | 0.891 | 0.899 |
| VS-MC-RC-076 | axial slab | 0.884 | 0.632 | 0.150 | 0.458 | 0.000 |
| VS-MC-RC-081 | axial slab | 0.820 | 0.685 | 0.000 | 0.834 | 0.000 |
| VS-MC-RC-082 | axial whole-head | 0.924 | 0.908 | 0.915 | 0.915 | 0.910 |
| VS-MC-RC-085 | axial whole-head | 0.917 | 0.878 | 0.919 | 0.917 | 0.909 |
| VS-MC-RC-097 | axial slab | 0.862 | 0.749 | 0.818 | 0.828 | 0.808 |
| VS-MC-RC-098 | axial slab | 0.893 | 0.841 | 0.426 | 0.910 | 0.000 |
| VS-MC-RC-099 | axial whole-head | 0.909 | 0.826 | 0.887 | 0.894 | 0.883 |
| VS-MC-RC-100 | non-axial | 0.936 | 0.909 | 0.891 | 0.908 | 0.000 |
