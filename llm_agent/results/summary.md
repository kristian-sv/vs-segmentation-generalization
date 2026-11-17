# LLM agent vs nnU-Net on the 22 MC-RC patients

Model: claude-opus-5-5, effort high. Runs: run1. Arms average their models within each patient (no mask ensembling). Empty outputs count individual model predictions. Paired 95% CIs: patient bootstrap, 10,000 resamples, seed 42. A run that never saved a valid mask counts as empty.

### All patients (n = 22)

| Source | Mean Dice | Median Dice | Empty outputs |
|---|---|---|---|
| LLM agent | 0.778 | 0.876 | 0/22 |
| plain arm | 0.546 | 0.621 | 10/44 |
| augmented arm | 0.761 | 0.851 | 1/44 |
| nnU-Net initial | 0.483 | 0.679 | 7/22 |

| Paired difference (LLM minus) | Mean | 95% CI |
|---|---|---|
| augmented arm | +0.017 | [-0.090, +0.123] |
| plain arm | +0.232 | [+0.066, +0.394] |
| nnU-Net initial | +0.295 | [+0.141, +0.462] |

### Primary slab cases (13 slabs, as in RQ3) (n = 13)

| Source | Mean Dice | Median Dice | Empty outputs |
|---|---|---|---|
| LLM agent | 0.773 | 0.845 | 0/13 |
| plain arm | 0.364 | 0.379 | 10/26 |
| augmented arm | 0.728 | 0.834 | 1/26 |
| nnU-Net initial | 0.335 | 0.000 | 7/13 |

| Paired difference (LLM minus) | Mean | 95% CI |
|---|---|---|
| augmented arm | +0.045 | [-0.030, +0.133] |
| plain arm | +0.409 | [+0.234, +0.584] |
| nnU-Net initial | +0.438 | [+0.230, +0.644] |

### Axial whole-head (n = 7)

| Source | Mean Dice | Median Dice | Empty outputs |
|---|---|---|---|
| LLM agent | 0.876 | 0.878 | 0/7 |
| plain arm | 0.898 | 0.896 | 0/14 |
| augmented arm | 0.897 | 0.894 | 0/14 |
| nnU-Net initial | 0.892 | 0.899 | 0/7 |

| Paired difference (LLM minus) | Mean | 95% CI |
|---|---|---|
| augmented arm | -0.021 | [-0.047, +0.011] |
| plain arm | -0.022 | [-0.047, +0.009] |
| nnU-Net initial | -0.016 | [-0.044, +0.016] |

### Axial slab (n = 11)

| Source | Mean Dice | Median Dice | Empty outputs |
|---|---|---|---|
| LLM agent | 0.745 | 0.841 | 0/11 |
| plain arm | 0.347 | 0.379 | 9/22 |
| augmented arm | 0.731 | 0.834 | 1/22 |
| nnU-Net initial | 0.342 | 0.000 | 6/11 |

| Paired difference (LLM minus) | Mean | 95% CI |
|---|---|---|
| augmented arm | +0.014 | [-0.049, +0.082] |
| plain arm | +0.398 | [+0.220, +0.571] |
| nnU-Net initial | +0.403 | [+0.181, +0.635] |

### Non-axial (n = 4)

| Source | Mean Dice | Median Dice | Empty outputs |
|---|---|---|---|
| LLM agent | 0.695 | 0.918 | 0/4 |
| plain arm | 0.477 | 0.499 | 1/8 |
| augmented arm | 0.608 | 0.648 | 0/8 |
| nnU-Net initial | 0.153 | 0.007 | 1/4 |

| Paired difference (LLM minus) | Mean | 95% CI |
|---|---|---|
| augmented arm | +0.088 | [-0.476, +0.563] |
| plain arm | +0.218 | [-0.403, +0.813] |
| nnU-Net initial | +0.542 | [+0.173, +0.911] |

## Per patient (Dice)

| Patient | Stratum | LLM agent | plain arm | augmented arm | nnU-Net initial |
|---|---|---|---|---|---|
| VS-MC-RC-009 | axial slab | 0.929 | 0.458 | 0.937 | 0.000 |
| VS-MC-RC-012 | non-axial | 0.947 | 0.020 | 0.518 | 0.600 |
| VS-MC-RC-016 | axial slab | 0.000 | 0.000 | 0.000 | 0.000 |
| VS-MC-RC-021 | axial whole-head | 0.882 | 0.896 | 0.892 | 0.883 |
| VS-MC-RC-024 | axial slab | 0.951 | 0.379 | 0.779 | 0.758 |
| VS-MC-RC-026 | axial slab | 0.899 | 0.000 | 0.723 | 0.000 |
| VS-MC-RC-028 | axial whole-head | 0.860 | 0.926 | 0.922 | 0.926 |
| VS-MC-RC-030 | axial slab | 0.845 | 0.228 | 0.840 | 0.456 |
| VS-MC-RC-043 | non-axial | 0.000 | 0.770 | 0.778 | 0.000 |
| VS-MC-RC-044 | axial whole-head | 0.905 | 0.847 | 0.846 | 0.837 |
| VS-MC-RC-045 | axial slab | 0.883 | 0.890 | 0.855 | 0.877 |
| VS-MC-RC-057 | axial slab | 0.781 | 0.473 | 0.873 | 0.859 |
| VS-MC-RC-060 | non-axial | 0.926 | 0.227 | 0.228 | 0.013 |
| VS-MC-RC-063 | axial whole-head | 0.875 | 0.894 | 0.891 | 0.899 |
| VS-MC-RC-076 | axial slab | 0.632 | 0.150 | 0.458 | 0.000 |
| VS-MC-RC-081 | axial slab | 0.685 | 0.000 | 0.834 | 0.000 |
| VS-MC-RC-082 | axial whole-head | 0.908 | 0.915 | 0.915 | 0.910 |
| VS-MC-RC-085 | axial whole-head | 0.878 | 0.919 | 0.917 | 0.909 |
| VS-MC-RC-097 | axial slab | 0.749 | 0.818 | 0.828 | 0.808 |
| VS-MC-RC-098 | axial slab | 0.841 | 0.426 | 0.910 | 0.000 |
| VS-MC-RC-099 | axial whole-head | 0.826 | 0.887 | 0.894 | 0.883 |
| VS-MC-RC-100 | non-axial | 0.909 | 0.891 | 0.908 | 0.000 |

Run outcomes: {'ok': 22}
