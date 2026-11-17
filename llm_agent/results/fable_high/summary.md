# Fable 5.2 high: 22 MC-RC patients

Model: `claude-fable-5-2`, effort high, Claude Code CLI on the Claude subscription login. One independent session per patient. Same prompt, 12 tools, Docker image, staged images, case order and 40-call / 20-image / 60-minute limits as run1 (Opus 5.5); only the model name differs. Invalid or unsaved masks count as empty predictions. nnU-Net arms average their two seeds within each patient. Paired 95% intervals: patient bootstrap, 10,000 resamples, seed 42. This arm was added after the Opus and Astra results were known and is not in PROTOCOL.md. Patient 028 was the pilot image.

## All patients (n = 22)

| Source | Mean Dice | Median Dice | Empty outputs |
|---|---:|---:|---:|
| Fable 5.2 high | 0.828 | 0.887 | 0/22 |
| Opus 5.5 high | 0.778 | 0.876 | 0/22 |
| Astra high | 0.845 | 0.907 | 0/22 |
| Astra xhigh | 0.854 | 0.915 | 0/22 |
| plain arm | 0.546 | 0.621 | 10/44 |
| augmented arm | 0.761 | 0.851 | 1/44 |
| nnU-Net initial | 0.483 | 0.679 | 7/22 |

| Paired difference (Fable minus) | Mean | 95% CI | Fable higher in |
|---|---:|---|---:|
| Opus 5.5 high | +0.050 | [+0.007, +0.124] | 17/22 |
| Astra high | -0.017 | [-0.083, +0.066] | 4/22 |
| Astra xhigh | -0.026 | [-0.098, +0.062] | 5/22 |
| plain arm | +0.282 | [+0.148, +0.424] | 16/22 |
| augmented arm | +0.067 | [-0.001, +0.152] | 12/22 |
| nnU-Net initial | +0.345 | [+0.186, +0.512] | 18/22 |

## Primary slab cases (13) (n = 13)

| Source | Mean Dice | Median Dice | Empty outputs |
|---|---:|---:|---:|
| Fable 5.2 high | 0.791 | 0.876 | 0/13 |
| Opus 5.5 high | 0.773 | 0.845 | 0/13 |
| Astra high | 0.865 | 0.904 | 0/13 |
| Astra xhigh | 0.872 | 0.914 | 0/13 |
| plain arm | 0.364 | 0.379 | 10/26 |
| augmented arm | 0.728 | 0.834 | 1/26 |
| nnU-Net initial | 0.335 | 0.000 | 7/13 |

| Paired difference (Fable minus) | Mean | 95% CI | Fable higher in |
|---|---:|---|---:|
| Opus 5.5 high | +0.018 | [-0.004, +0.042] | 10/13 |
| Astra high | -0.074 | [-0.136, -0.028] | 2/13 |
| Astra xhigh | -0.081 | [-0.160, -0.025] | 2/13 |
| plain arm | +0.427 | [+0.254, +0.598] | 11/13 |
| augmented arm | +0.063 | [-0.012, +0.150] | 7/13 |
| nnU-Net initial | +0.456 | [+0.251, +0.664] | 11/13 |

## axial whole-head (n = 7)

| Source | Mean Dice | Median Dice | Empty outputs |
|---|---:|---:|---:|
| Fable 5.2 high | 0.895 | 0.903 | 0/7 |
| Opus 5.5 high | 0.876 | 0.878 | 0/7 |
| Astra high | 0.909 | 0.909 | 0/7 |
| Astra xhigh | 0.917 | 0.925 | 0/7 |
| plain arm | 0.898 | 0.896 | 0/14 |
| augmented arm | 0.897 | 0.894 | 0/14 |
| nnU-Net initial | 0.892 | 0.899 | 0/7 |

| Paired difference (Fable minus) | Mean | 95% CI | Fable higher in |
|---|---:|---|---:|
| Opus 5.5 high | +0.018 | [+0.006, +0.028] | 6/7 |
| Astra high | -0.014 | [-0.032, +0.005] | 1/7 |
| Astra xhigh | -0.022 | [-0.044, -0.002] | 2/7 |
| plain arm | -0.003 | [-0.025, +0.018] | 4/7 |
| augmented arm | -0.002 | [-0.024, +0.020] | 4/7 |
| nnU-Net initial | +0.002 | [-0.022, +0.026] | 5/7 |

## axial slab (n = 11)

| Source | Mean Dice | Median Dice | Empty outputs |
|---|---:|---:|---:|
| Fable 5.2 high | 0.765 | 0.834 | 0/11 |
| Opus 5.5 high | 0.745 | 0.841 | 0/11 |
| Astra high | 0.851 | 0.893 | 0/11 |
| Astra xhigh | 0.858 | 0.899 | 0/11 |
| plain arm | 0.347 | 0.379 | 9/22 |
| augmented arm | 0.731 | 0.834 | 1/22 |
| nnU-Net initial | 0.342 | 0.000 | 6/11 |

| Paired difference (Fable minus) | Mean | 95% CI | Fable higher in |
|---|---:|---|---:|
| Opus 5.5 high | +0.020 | [-0.006, +0.048] | 8/11 |
| Astra high | -0.086 | [-0.155, -0.033] | 1/11 |
| Astra xhigh | -0.093 | [-0.184, -0.028] | 2/11 |
| plain arm | +0.418 | [+0.247, +0.584] | 9/11 |
| augmented arm | +0.035 | [-0.029, +0.108] | 5/11 |
| nnU-Net initial | +0.424 | [+0.200, +0.655] | 9/11 |

## non-axial (n = 4)

| Source | Mean Dice | Median Dice | Empty outputs |
|---|---:|---:|---:|
| Fable 5.2 high | 0.885 | 0.919 | 0/4 |
| Opus 5.5 high | 0.695 | 0.918 | 0/4 |
| Astra high | 0.719 | 0.847 | 0/4 |
| Astra xhigh | 0.730 | 0.860 | 0/4 |
| plain arm | 0.477 | 0.499 | 1/8 |
| augmented arm | 0.608 | 0.648 | 0/8 |
| nnU-Net initial | 0.153 | 0.007 | 1/4 |

| Paired difference (Fable minus) | Mean | 95% CI | Fable higher in |
|---|---:|---|---:|
| Opus 5.5 high | +0.189 | [+0.000, +0.562] | 3/4 |
| Astra high | +0.165 | [-0.017, +0.515] | 2/4 |
| Astra xhigh | +0.155 | [-0.032, +0.508] | 1/4 |
| plain arm | +0.407 | [-0.000, +0.815] | 3/4 |
| augmented arm | +0.277 | [-0.012, +0.566] | 3/4 |
| nnU-Net initial | +0.731 | [+0.492, +0.912] | 4/4 |

## Volume error and HD95, all patients (LLM arms)

| Source | Median abs. volume error (%) | Median signed volume error (%) | Under-segmented | Median HD95 (mm) | Dice = 0 | Dice < 0.5 |
|---|---:|---:|---:|---:|---:|---|
| Fable 5.2 high | 16.4 | -15.1 | 21/22 | 1.42 | 1 | 016 |
| Opus 5.5 high | 21.1 | -19.8 | 19/22 | 1.71 | 2 | 016, 043 |
| Astra high | 10.8 | -7.5 | 16/22 | 1.58 | 0 | 016, 060 |
| Astra xhigh | 9.6 | -5.9 | 15/22 | 1.41 | 0 | 016, 060 |

## Per-patient Dice

| Patient | Stratum | Fable 5.2 high | Opus 5.5 high | Astra high | Astra xhigh | plain arm | augmented arm | nnU-Net initial |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| VS-MC-RC-009 | axial slab | 0.936 | 0.929 | 0.935 | 0.933 | 0.458 | 0.937 | 0.000 |
| VS-MC-RC-012 | non-axial | 0.952 | 0.947 | 0.947 | 0.958 | 0.020 | 0.518 | 0.600 |
| VS-MC-RC-016 | axial slab | 0.000 | 0.000 | 0.379 | 0.479 | 0.000 | 0.000 | 0.000 |
| VS-MC-RC-021 | axial whole-head | 0.908 | 0.882 | 0.926 | 0.935 | 0.896 | 0.892 | 0.883 |
| VS-MC-RC-024 | axial slab | 0.951 | 0.951 | 0.962 | 0.963 | 0.379 | 0.779 | 0.758 |
| VS-MC-RC-026 | axial slab | 0.876 | 0.899 | 0.886 | 0.890 | 0.000 | 0.723 | 0.000 |
| VS-MC-RC-028 | axial whole-head | 0.872 | 0.860 | 0.902 | 0.915 | 0.926 | 0.922 | 0.926 |
| VS-MC-RC-030 | axial slab | 0.783 | 0.845 | 0.910 | 0.925 | 0.228 | 0.840 | 0.456 |
| VS-MC-RC-043 | non-axial | 0.748 | 0.000 | 0.758 | 0.788 | 0.770 | 0.778 | 0.000 |
| VS-MC-RC-044 | axial whole-head | 0.892 | 0.905 | 0.909 | 0.892 | 0.847 | 0.846 | 0.837 |
| VS-MC-RC-045 | axial slab | 0.888 | 0.883 | 0.923 | 0.914 | 0.890 | 0.855 | 0.877 |
| VS-MC-RC-057 | axial slab | 0.804 | 0.781 | 0.904 | 0.916 | 0.473 | 0.873 | 0.859 |
| VS-MC-RC-060 | non-axial | 0.925 | 0.926 | 0.235 | 0.242 | 0.227 | 0.228 | 0.013 |
| VS-MC-RC-063 | axial whole-head | 0.903 | 0.875 | 0.873 | 0.884 | 0.894 | 0.891 | 0.899 |
| VS-MC-RC-076 | axial slab | 0.742 | 0.632 | 0.884 | 0.899 | 0.150 | 0.458 | 0.000 |
| VS-MC-RC-081 | axial slab | 0.719 | 0.685 | 0.820 | 0.800 | 0.000 | 0.834 | 0.000 |
| VS-MC-RC-082 | axial whole-head | 0.922 | 0.908 | 0.924 | 0.937 | 0.915 | 0.915 | 0.910 |
| VS-MC-RC-085 | axial whole-head | 0.910 | 0.878 | 0.917 | 0.925 | 0.919 | 0.917 | 0.909 |
| VS-MC-RC-097 | axial slab | 0.834 | 0.749 | 0.862 | 0.856 | 0.818 | 0.828 | 0.808 |
| VS-MC-RC-098 | axial slab | 0.887 | 0.841 | 0.893 | 0.867 | 0.426 | 0.910 | 0.000 |
| VS-MC-RC-099 | axial whole-head | 0.854 | 0.826 | 0.909 | 0.929 | 0.887 | 0.894 | 0.883 |
| VS-MC-RC-100 | non-axial | 0.913 | 0.909 | 0.936 | 0.932 | 0.891 | 0.908 | 0.000 |

Run outcomes (Fable 5.2 high): {'ok': 22}
