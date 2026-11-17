# Protocol: LLM agent segmentation on the 22 MC-RC patients

Frozen at git tag `freeze-v1`, before any MC-RC case was run. Changes after the
freeze are listed at the end with their reason.

## Question
How does a general-purpose LLM agent, using only classical image tools, segment
vestibular schwannoma on the same 22 MC-RC T1 scans used in the thesis, and does
it fail on the same acquisition types (slabs, non-axial scans) as the trained
nnU-Net models?

## Agent
- Model `claude-opus-5-5`, effort `high`, through Claude Code (`claude -p`, version
  2.1.311, recorded per run in `status.json`) on the author's subscription, started
  with a minimal environment (no inherited variables, no API key).
- One new session per case and run: empty working directory, fixed system prompt
  (`harness/prompt.md`), no built-in tools (`--tools ""`), no settings, no
  auto-memory, no session history, no other MCP servers.
- Tools: the 12 tools in `harness/tools.py` (info, stats, overview, view, review,
  threshold, grow, components, morph, exclude_box, union, save). No arbitrary code.
- Budget: 40 tool calls and 20 images per case; `save` is always allowed.
  Failed calls count toward the budget. Wall-clock limit 60 minutes per case.

## Sandbox
- Tools run in a new Docker container per case (`harness/Dockerfile`, image
  `llm-seg-tools:v1`, image ID `sha256:9560b625afff…`): `--network none`, read-only root, non-root user, all
  capabilities dropped, no-new-privileges, 4 GB memory, 2 CPUs, 256 processes.
- Mounts: that case's image read-only at `/data`, and `runs/runN/<patient>` at
  `/out`. Nothing else. The login token stays on the host.
- The agent's only output is `label.nii.gz` written by `save`; the tool server
  also writes its own log and the rendered images to `/out`.

## Inputs
- `staging/<patient>/image.nii.gz`: the 22 `T1.nii.gz` files from
  `~/MMIV/data/VS_MCRC_T1_evaluation_mm`, run in a fixed random order
  (seed 2026), with free-text header fields cleared and voxels and geometry
  unchanged (checked on copy). The key is in `private/case_key.csv`.
- Reference labels are never copied to `staging/` or mounted into a container.
- Images are re-stacked to canonical orientation inside the tools (axis
  permutation and flips, no resampling), so non-axial scans are shown in the
  correct anatomical planes; `save` maps the mask back to the original grid.
  The follow-up nnU-Net models also received re-stacked inputs.
- The prompt states that the scan contains one vestibular schwannoma. nnU-Net
  receives no such statement; every reference in this cohort is non-empty.

## Runs
- Two independent runs per case (`runs/run1`, `runs/run2`), cases in staging order,
  up to two cases at a time (`--jobs 2`).
- Before the freeze, the harness was tested on the pilot image only (smoke test,
  Dice 0.849). That test found a slice-rounding bug in `view`, fixed before the freeze.
- No human intervention during a run. Errors and timeouts are not retried
  unless the run produced no transcript at all (infrastructure failure), which
  is recorded.
- A run without a valid saved mask counts as an empty prediction and stays in
  every denominator.

## Analysis (`harness/score.py`)
- Common scoring on each patient's original T1 grid against
  `VS_MCRC_T1_evaluation_mm/<patient>/label.nii.gz`: Dice, empty prediction,
  signed volume error (ml, %), HD95 (non-empty predictions only).
- LLM arm = mean of the two runs per patient. Comparators from the thesis:
  plain arm (plain s42, s43), augmented arm (slabthick s42, s43), initial
  nnU-Net s42 (stored inputs).
- Reported for all 22, the 13 slab cases (the RQ3 primary population), and the
  three strata (7 axial whole-head, 11 axial slab, 4 non-axial).
- Paired differences LLM minus comparator with 95% patient-bootstrap intervals
  (10,000 resamples, seed 42). Descriptive; no hypothesis test.

## Known limitations to report
- One model, one prompt, two runs per case.
- The toolbox and prompt were designed by an agent that had read the thesis,
  including per-patient results; the predicting agents had no access to it.
- A single-case pilot (`pilot/`) was run before this protocol, by an agent with
  thesis context, and scored against its reference. The pilot image is
  VS-MC-RC-028 (identified afterwards by voxel comparison), one of the 22 cohort
  patients; both smoke tests also used it. Development therefore saw one cohort
  case and its reference.
- MC-RC was inspected during the thesis; the agent design (re-stacking,
  geometry reporting) was informed by its known failure modes.
- Public datasets may be in the model's training data; this cannot be ruled out.

## Changes after freeze
- 2026-10-27, before any MC-RC case was run: one run per case instead of two
  (author's decision, to limit scope). Only `runs/run1` is produced; run-to-run
  variation of the agent is therefore not estimated and must be reported as a
  limitation.
- 2026-10-27, after run 1: case folders in `staging/` and `runs/run1/` renamed from
  the run-order names (case_01 … case_22) to the MC-RC patient IDs, for clarity.
  Folder names were never visible to the agent (its container only sees
  `/data/image.nii.gz`); `private/case_key.csv` keeps the run order, and each
  `status.json` records its `run_order_case`. Contents were not changed.
