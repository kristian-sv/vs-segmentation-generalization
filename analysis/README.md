# Research evidence

For an overview of the project and its results, start at [the repository guide](../README.md).
This folder holds the analysis scripts and the dated evidence reports:

- `model_insight.py`, run through `model_insight.slurm`: failure analysis,
  stress test and follow-up evaluation; `make_case_figure.py` is its
  case-figure helper.
- `thesis_analysis.py`: the original comparison statistics and the paired
  bootstrap of the seven-run comparison.
- `audit_mcrc_vsseg_scan_overlap.py`: the scan-level overlap audit.

Reports behind the three steps of the thesis:

- [Seven-model original-label T1 evaluation](reports/t1_evaluation_20260912/README.md);
  the statistics derived from it are in `reports/thesis_statistics_20261008/statistics.json`
- [Why MC-RC cases fail, and the stress test on VS-SEG validation](reports/model_insight_20261012/FINDINGS.md)
- [Pre-registration of the slab/thick-slice augmentation experiment](reports/model_insight_20261012/PREREGISTRATION.md);
  its results are under `reports/model_insight_20261012/mcrc_followup/`
- [Internal-test check](reports/internal_test_20261021/REPORT.md) and its
  [protocol](reports/internal_test_20261021/PROTOCOL.md)

Cohort and source checks:

- [Corrected 22-patient retrospective MC-RC evaluation](reports/trustworthy_testing_20260907/REPORT.md)
- [Scan-level overlap between VS-MC-RC and VS-SEG](reports/mcrc_vsseg_scan_overlap_20261103/REPORT.md)
- [Source annotation checks for 217 VS-SEG development cases](reports/vsseg_t1_readiness_mvp/REPORT.md)

Supporting reports; read these only when you need the detail:

- [Completed training-readiness checks](reports/training_readiness_20260908/REPORT.md)
- [What the historical experiments actually show](reports/experiment_history_mvp/README.md)
- [Feasibility of an independent T1 test cohort](reports/independent_test_feasibility_mvp/REPORT.md)

Older reports retain historical commands, paths and hashes; some referenced
local artifacts, reports and retired helpers are not distributed with this
repository. Commit hashes refer to the private development history. Use the
active experiment guide for current commands. Scientific findings, comparison
limitations and necessary experiment records remain included.
