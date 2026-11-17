# Exploratory comparison: tool-using LLM agents

General-purpose language models segmented the vestibular schwannoma on the 22
VS-MC-RC T1 scans of the thesis. Each model worked alone in a sandbox, through
12 classical image tools (thresholding, region growing, morphology, viewing),
with no VS-specific training and no access to reference masks. The masks were
scored in the same way as the nnU-Net predictions. The design is in
[PROTOCOL.md](PROTOCOL.md); the thesis reports the experiment in Sections 3.6.5
and 4.7 and in its appendices.

This folder holds the prompt, the tools, the runner and scoring code, the
protocol, the score tables and, for every case, the status record and the tool
log. It contains no MRI volumes, rendered slices, masks or model transcripts.

## Arms and results

One run per patient in every arm. Values are volume-level Dice (mean/median).

| Arm | Model | Run | All 22 patients | 13 slab cases | Difference from augmented nnU-Net, 22 patients (95% CI) |
|---|---|---|---|---|---|
| Opus 5.5 | `claude-opus-5-5`, effort high | 27 Oct 2026, pre-specified | 0.778 / 0.876 | 0.773 / 0.845 | +0.017 [−0.090, +0.123] |
| Astra high | `gpt-6-astra`, effort high | 8 Nov 2026, post hoc | 0.845 / 0.907 | 0.865 / 0.904 | +0.084 [+0.031, +0.148] |
| Astra xhigh | `gpt-6-astra`, effort xhigh | 8 Nov 2026, post hoc | 0.854 / 0.915 | 0.872 / 0.914 | +0.092 [+0.035, +0.162] |
| Fable 5.2 | `claude-fable-5-2`, effort high | 12 Nov 2026, post hoc | 0.828 / 0.887 | 0.791 / 0.876 | +0.067 [−0.001, +0.152] |
| Augmented nnU-Net (two seeds) | | | 0.761 / 0.851 | 0.728 / 0.834 | |

Intervals are patient bootstraps (10,000 resamples, seed 42) and are
descriptive. The full tables, including the strata and every patient, are in
the `summary.md` files under `results/`.

## Limits

- The prompt states that the scan contains one vestibular schwannoma. nnU-Net
  receives no such statement.
- Each arm is a single run, so run-to-run variation is not estimated.
- Only the Opus arm was specified before its results were known. The Astra and
  Fable arms were added afterwards and are not in the frozen protocol.
- The prompt and tools were designed with knowledge of this cohort, and a
  single-case pilot used VS-MC-RC-028, one of the 22 patients.
- Public datasets may be in the models' training data.

`PROTOCOL.md` lists these and further limits in full.

## What is here

| Path | Contents |
|---|---|
| `PROTOCOL.md` | Protocol of the Opus arm, frozen before any MC-RC case was run, with the changes made after the freeze. |
| `harness/prompt.md` | The system prompt and task text, the same in all four arms. |
| `harness/tools.py`, `harness/Dockerfile` | The 12 tools and the sandbox image they run in. |
| `harness/run.py`, `harness/score.py` | Runner and scorer of the Opus arm. The other arms reuse their functions. |
| `harness/astra/` | Runner and scorer of the two Astra arms, their setup notes (`README.md`), the outcome of the request audit and the scoring check. |
| `harness/fable/` | Runner, scorer and reviewer export of the Fable arm. |
| `private/case_key.csv` | The shuffled run order and the checksum of each input scan. "Private" meant hidden from the agent. |
| `results/` | `scores.csv`, `summary.md` and a per-patient figure for the Opus arm, and the same in `astra_high/`, `astra_xhigh/` and `fable_high/`. |
| `runs/<arm>/<patient>/` | `status.json` (outcome, timing, model and, for the post hoc arms, checksums of prompt, tools and input) and `tool_log.jsonl` (every tool call with its arguments and text result). The Opus arm is `runs/run1/`. |

Checksums (SHA-256) recorded in the post hoc arms and reproducible from the
files here:

- `harness/prompt.md`: `aa9e523489703a4f84152735b939622ecccb130e3faeb77cbf251a942688c975`
- `harness/tools.py`: `658e5160a4e7a8a749c94c393264b35c2f14a83d1358cbd6331f35232f189de5`

## What is not here

Kept privately and available on request:

- the model transcripts, the slice images shown to the models and the saved
  masks of each case (about 4.6 GB);
- the staged input images and the single-case pilot named in the protocol;
- the Opus runner's `run.log` and the progress logs.

Left out for other reasons:

- `harness/astra/astra_catalog.json`, the pinned Codex model catalog that
  `astra_run.py` loads. It consists mostly of the vendor's own instruction
  text.
- `harness/astra/audit-*/request.json`, the two captured requests. They hold
  session and installation identifiers. Their checksums are in the
  `audit_passed.json` files.
- `reviewer_exports.json`, lists of local file paths.

## Notes on the files

- The files are unchanged copies, so the recorded checksums still match. They
  therefore keep paths from the machine the runs were made on (`~/MMIV/...`,
  and absolute paths in the Astra scripts and notes) and cannot be run
  elsewhere without changing those paths and supplying the data.
- The Opus arm was developed in a private repository. The protocol was frozen
  there as tag `freeze-v1` (commit `237a379`), and the run outputs and results
  were committed in `d12d416`. That history is not published.
- The runner code of the Astra and Fable arms was not under version control
  when those arms were run. It is committed here for the first time.
- `harness/astra/astra_run.py` was last edited at 20:54 on 8 November 2026,
  after both Astra arms had finished (15:07 and 15:30). The edit changed only
  how subprocesses and files are closed. The earlier version, with a file time
  of 14:22, one minute before the arms started, is in
  `harness/astra/maintenance_20261108/astra_run_before.py`.
