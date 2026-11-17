# Astra MC-RC extension — 2026-11-08

Two independent arms: `gpt-6-astra` at `high` and `xhigh`, one run for each of
the same 22 MC-RC cases, in the order in `private/case_key.csv`. One worker per
effort; each case starts a new model session and a new Docker sandbox. The two
workers run concurrently and share no model context or segmentation state.

The original Opus harness, prompt, runs, and results are preserved. The frozen
Docker image is pinned by its full ID:
`sha256:9560b625afffe5228b50f09a1a67795848029f6bc12ee0543087bd0f698a5b8f`.
The image's `/tools/tools.py` matches `harness/tools.py` byte-for-byte (SHA-256
`658e5160a4e7a8a749c94c393264b35c2f14a83d1358cbd6331f35232f189de5`).

## Experimental parity

- Original system-prompt text and task text, unchanged.
- Original 12 tools, descriptions, input schemas, image bytes and outputs. The
  host bridge forwards calls to the original MCP server without altering tools.
- Original 40-tool-call and 20-image budgets, including original handling of
  failed calls; `save` remains available after exhaustion.
- Original 60-minute per-case wall-clock budget and mask validation.
- Original read-only input image, writable output directory, no other mounts.
  Network disabled, read-only root, non-root user, all capabilities dropped,
  no-new-privileges, 4 GB RAM, 2 CPUs, 256 processes, 512 MB temporary filesystem.
- No reference labels, patient identifiers, previous predictions, other cases,
  memory, filesystem tools, code execution, browsing, skills, plugins, extra MCP
  tools, or human feedback are exposed to the predicting model.
- Credentials remain on the host. Codex 0.167.0 uses the existing ChatGPT login;
  no API key or API-billing fallback is configured.

Codex's ordinary tool catalog is broader than the experiment allows. A pinned
catalog removes its built-in tools and selects direct function calls. A fresh
Codex app-server receives only the original 12 schemas as dynamic tools. The
trusted host bridge owns the Docker MCP connection, so Codex does not add MCP
resource-discovery tools. The model ID remains `gpt-6-astra` with no fallback.

`audit-high/` and `audit-xhigh/` contain captured model requests sent to a local
mock endpoint, not OpenAI. They verify the exact two prompt texts and exact 12
tools. No inference was performed on the audit image. Standalone connectivity
checks used no dataset inputs. The audit's intentional HTTP 400 terminates the
local capture; `audit_passed.json` records the actual verification outcome.

Cross-provider transport necessarily differs: the original Claude system text
is sent as a Codex developer message, and tools use Codex's function namespace.
Image encoding/tokenization is provider-specific. These are transport
differences, not changes to experimental data access or segmentation tools.

## Outputs and terminal inspection

From `llm-test`:

```sh
tail -F runs/astra_high/progress.log runs/astra_xhigh/progress.log
cat runs/astra_high/progress.json runs/astra_xhigh/progress.json
```

Per case, `runs/astra_<effort>/<patient>/` stores the predicted `label.nii.gz`,
model transcript, original tool log and rendered views, status, timing, exact
tool schemas, session settings, and input/prompt/tool checksums. Logs include
model usage events. Never assume a case is complete until `status.json` exists.

Each arm automatically scores and exports after all 22 cases finish. Results
go to `results/astra_high/` and `results/astra_xhigh/`, with per-case metrics,
summary tables, patient-bootstrap intervals, a Dice figure, and an export map.
The original metric functions and strata are reused. The scoring adapter was
checked against all existing Opus outputs, reproducing the original metrics.

Reviewer exports are `pred_llm_astra_high_run1.nii.gz` and
`pred_llm_astra_xhigh_run1.nii.gz` in each patient's existing MC-RC model-insight
bundle directory. In MMIV Reviewer choose **Model insight · MC-RC**, rescan,
and select the corresponding prediction under Compare.

An unsaved/invalid mask or timeout is retained as a trial outcome; scoring uses
an empty prediction when no valid mask exists. Unexpected bridge or server
failures stop that arm and write `infrastructure_error.json` for inspection;
the runner does not automatically repeat a case or adjust its prompt. High and
xhigh are distinct arms and are not averaged together. The original protocol's
pilot-exposure limitation still applies.

## Running

The supervisor launches both arms, then each worker scores and exports itself:

```sh
/Users/kristiansvendsen/miniforge3/bin/python harness/astra/astra_run.py launch
```

The supervisor holds an exclusive lock against duplicate launches. Completed
cases are skipped. Inspect and resolve any infrastructure error before a
manual restart. The isolated CLI installation is under
`/Users/kristiansvendsen/Documents/Codex/2026-11-08/bu/work/astra/cli/`; the global
Codex installation was not modified.
