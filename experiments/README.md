# T1 experiments

Read [the root guide](../README.md) first. The original comparison and the separate
five-fold nnU-Net experiment below both use T1 only.

## The experiment and its limits

| Run | Change | Seeds |
|---|---|---|
| baseline | Existing manual MONAI 3D U-Net; dropout 0.2 | 42, 43, 44 |
| dropout01 | Only dropout changes to 0.1 | 42, 43, 44 |
| nnU-Net | Installed nnU-Net 2.8.1, standard trainer, `3d_fullres`, fold 0 | 42 |

Lower dropout is a hypothesis suggested by previously inspected external scores.
It is not a new confirmatory hypothesis. Retraining assesses consistency, rather
than counting identical historical checkpoint copies as repetitions.

Both pipelines use the same **193 train / 24 validation** patients. The original
24 internal-test patients remain separate. VS-SEG-159 still lacks its label and
is excluded. `data_lock.json` preserves the original identities, metadata and
217 development image/label hashes; it is not a settings file to edit casually.
The data guard refuses overlaps, missing cases and changed input bytes.

Manual training keeps the existing loss (DiceFocal), AdamW, warmup/cosine schedule,
augmentation and full-volume validation. It stops at 300 epochs or 75 epochs
without improvement, saving the first strictly best validation checkpoint.
Compare all paired seeds. Select the configuration by mean best validation Dice;
ties favour baseline. Seed 42 is the predeclared representative when one is needed.

nnU-Net uses its own preprocessing, loss, optimizer and augmentation. Its 193-case
training-only plans are frozen before adding validation. It trains for 1000 epochs
with 250 updates per epoch, one fixed fold, and single-process augmentation.
Select `checkpoint_best.pth` (moving-average validation pseudo-Dice). Its native
selection metric differs from the manual model's metric. This is a **pipeline
comparison**, not an isolated architecture experiment. No bitwise reproducibility
claim is made for GPU training; one nnU-Net run cannot establish its variability.

Use validation for decisions. No test/external data enter training or planning.
The old 22-patient MC-RC cohort is retrospective. Optional fresh MC-RC2 acquisition
is archived until needed; it is not yet a verified independent test cohort.

## Settings you may want to understand

Edit only [settings.json](settings.json) for manual hyperparameters. Both variants
read it. Do not change the settings between paired runs or tune against external
results; a new research variation needs a clearly recorded experiment plan.

| Key | Meaning |
|---|---|
| `dropout` / `lower_dropout` | Probability of switching off activations during training. |
| `lr` | Size of the optimizer's learning steps. |
| `epochs`, `patience` | Maximum training duration and early-stopping allowance. |
| `patch_size`, `num_samples` | Size and number of patches sampled from each image. |
| `pixdim` | Voxel spacing after resampling, in millimetres. |
| `channels` | Number of features at each network level. |

The remaining defaults preserve the verified recipe. Inactive T2/2.5D, alternative
loss and optional augmentation branches live in the archive. Source edits no
longer require manually updating a hash list: each run saves its actual source
and hashes. Patient-data and nnU-Net planning checks remain enforced.

## Training commands — from the repository root

```bash
mkdir -p logs experiments/work
# Inspect first; no GPU and no training:
python3.11 -B experiments/run.py --manual baseline --seed 42
python3.11 -B experiments/run.py --nnunet

# Submit only when ready for the actual study:
sbatch experiments/train.slurm manual baseline 42
sbatch experiments/train.slurm manual dropout01 42
sbatch --time=4-00:00:00 experiments/train.slurm nnunet
# Repeat the manual pair for seeds 43 and 44 after checking the first runs.
squeue --me
```

One job uses one GPU, 16 CPUs and 128 GiB system memory. Manual jobs request 24 h;
the nnU-Net command requests 96 h. These are allocation limits, not measured
completion times. Inspect early epoch times. A disconnected terminal does not
stop a submitted job; check its status before submitting another. Manual resume
is not implemented. A prematurely interrupted run is incomplete.

Logs: `logs/t1-<jobid>.log`. Manual outputs: `outputs/t1/manual/<run>/`.
nnU-Net outputs: `outputs/t1/nnunet/`. Local CSVs/checkpoints and source snapshots
are the records you need; W&B stays offline by default. Old models are preserved.

## Training submissions

Initial submissions on 8 September 2026 use the implementation at `16bfac0`.
Each run also saves its actual source and settings. Check `squeue --me` and the
job logs for current progress; a submitted job is not a completed result.

| Setup | Seed | Job ID |
|---|---:|---:|
| baseline | 42 | 2191374 |
| dropout01 | 42 | 2191375 |
| baseline | 43 | 2191464 |
| dropout01 | 43 | 2191465 |
| baseline | 44 | 2191466 |
| dropout01 | 44 | 2191467 |
| nnU-Net | 42 | 2191376 |

The seed-43 and seed-44 pairs were submitted after both seed-42 runs completed
five epochs with finite training/validation values and saved checkpoints.
Configuration and patient splits stay fixed. Early scores are progress checks;
compare the completed runs under the selection rules above.

`environment.sh` reuses the existing ARM64 container and overlay on GPU nodes;
it does not run on the x86 login host. nnU-Net prepared data already exist under
`experiments/work/`. `nnunet_prepare.py` is only for reproducing preparation on
a fresh workspace; it refuses to overwrite the existing prepared dataset.

## Separate five-fold T1 nnU-Net experiment

The 217 existing development patients (193 train + 24 validation) are reassigned
to five fixed patient folds. A seed-42 SHA-256 ordering gives 44/44/43/43/43
held-out patients; every development patient is held out exactly once. The
original 24 test patients, VS-SEG-159 without a mask, all MC-RC data, and T2 data
are excluded from training and planning.

Each fold reuses the existing staging, geometry checks, default 8 GiB planner,
nnU-Net 2.8.1 `3d_fullres` trainer, seed 42 and 1000 epochs. Planning uses only
that fold's 173 or 174 training patients; its held-out patients are appended
after the plans are frozen. Dataset IDs 902–906 keep these fold-specific plans
separate from Dataset901. Each dataset has one internal nnU-Net split (`fold_0`);
the outer fold number is in the dataset name and run snapshot.

Use **`checkpoint_final.pth`**, fixed in advance, for this experiment's predictions.
Held-out validation is monitored, but does not choose the checkpoint, training
duration, configuration or postprocessing. The trainer also writes a best
checkpoint; it is not the selected model for this experiment. Native final
validation exports held-out masks using nnU-Net's default mirroring. For comparison
with the original experiment, use the existing inference/scoring workflow to
export without TTA. Previously inspected development data remain retrospective.

```bash
python3.11 -B tests/check_nnunet_cv.py
python3.11 -B experiments/run.py --nnunet --cv-fold 0
# Each array task prepares its own dataset, then starts a fresh training run.
sbatch --array=0-4 --time=4-00:00:00 --job-name=t1-nnunet-cv \
  --output=logs/t1-cv-%A_%a.log experiments/train.slurm nnunet-cv
```

Check `squeue --me` and existing outputs before submitting. Preparation and
training refuse existing fold directories. Results and source/split snapshots:
`outputs/t1/nnunet_cv/Dataset90{2..6}_T1VSCVFold{0..4}/`. Fold-specific raw and
preprocessed data are under the existing `experiments/work/nnUNet_*` directories.
The earlier experiment's checkpoints and predictions remain available.

Submitted on 16 September 2026 as Slurm array **2272511**, tasks **0–4**.
`outputs/t1/nnunet_cv/submission_2272511.json` records the submitted code hashes,
all five patient assignments and the excluded test patients. Check the scheduler
and fold logs for completion; submission does not mean training has finished.

For subsequent inference, a development patient uses only the model from their
held-out fold. Shared MC-RC patient identities must follow that same assignment;
independent MC-RC and original test patients are absent from every training fold.

### Whole VS-SEG T1 review export

`predict_review.py` verifies and copies the 217 existing held-out predictions,
then predicts the 24 original test cases and unlabelled VS-SEG-159 with the final
checkpoint of outer fold 0. Those 25 cases were excluded from every training and
planning fold; fold 0 is fixed before inference. All masks use test-time mirroring.
No reference masks are loaded or new scores computed. The 242 masks, patient/model
manifest and verification record go to `outputs/t1/review/vsseg242_nnunet_cv/`.
Existing output or staging directories cause an error rather than replacement.

```bash
python3.11 -B experiments/predict_review.py
sbatch --job-name=t1-vsseg-review --output=logs/t1-vsseg-review-%j.log \
  experiments/evaluate.slurm vsseg-review
```

## Prediction and scoring — original comparison

The seven September training runs have completed. Their fixed checkpoints,
24-patient validation set and 22-patient direct-T1-reference MC-RC cohort are in
the [original-label evaluation record](../analysis/reports/t1_evaluation_20260912/README.md).
`evaluate.slurm` runs that comparison using the existing exporter and scorer below.

Run Python evaluation commands inside the manual ARM64 environment on an allocated
node: prefix them with `bash experiments/environment.sh manual`. Use
`T1_MVP_GPU=1` for GPU export. Make a validation manifest first:

```bash
python src/evaluate.py manifest --split val --output experiments/work/val.csv
python src/evaluate.py manual-export --manifest experiments/work/val.csv \
  --manual baseline --seed 42 --checkpoint /path/to/run/best_model.pt \
  --output experiments/work/manual_val
```

The checkpoint's saved settings must match the chosen variant and seed. Export
uses the same image preprocessing as training, constant sliding-window blending,
softmax probabilities and linear inverse transforms **before argmax**. Cropped-out
regions become background. Output masks use the original processed T1 grid.

For nnU-Net, stage only the chosen manifest's images as `<case_id>_0000.nii.gz`.
Inside `bash experiments/environment.sh nnunet` on an allocated GPU, use:

```bash
nnUNetv2_predict -i /path/to/input_images -o /path/to/nnunet_predictions \
  -d 901 -c 3d_fullres -f 0 -chk checkpoint_best.pth --disable_tta
```

Retain that invocation. Do not use mirrored native-validation masks as final
comparison outputs. No TTA, ensembling or component filtering for either pipeline.

Both pipelines' original-grid masks use the same scorer:

```bash
python src/evaluate.py score --manifest experiments/work/val.csv \
  --predictions experiments/work/manual_val --compare /path/to/nnunet_predictions \
  --output experiments/work/validation_comparison
```

It validates all IDs, coverage, hashes, binary labels, shapes and affines before
scoring. Per-patient foreground Dice: both empty = 1; one empty = 0. Summaries
contain mean, median, sample SD, volumes and optional paired Dice differences.
All output directories must be new. Extra boundary metrics are archived optional
work; Dice is the current primary metric. Freeze any final-test manifest and model
selection before evaluating it; this tool constructs only development manifests.

## Preregistered slab/thick-slice nnU-Net follow-up (October 2026)

Protocol: [PREREGISTRATION.md](../analysis/reports/model_insight_20261012/PREREGISTRATION.md) (commit db0521a).
Trainer: `nnunet_trainers/nnUNetTrainerSlabThick.py` (commit f25e035), found through `nnUNet_extTrainer`. Each run
reuses the Dataset901 plans, 193/24 split and 1000 epochs, and writes to `outputs/t1/nnunet_slab/<run>/`:

```bash
sbatch --time=2-00:00:00 experiments/train.slurm nnunet-followup nnUNetTrainer 43
sbatch --time=2-00:00:00 experiments/train.slurm nnunet-followup nnUNetTrainerSlabThick 42
sbatch --time=2-00:00:00 experiments/train.slurm nnunet-followup nnUNetTrainerSlabThick 43
```

Note: voxel rounding of the drawn slab width gives an effective range of 29.94–55.5 mm on about 1% of slab draws
(55.5 mm head-feet, 29.94 mm front-back; 200,000 simulated draws). Tumour-keeping widening does trigger whenever a tumour is
longer than the drawn slab. The largest training tumour is 31.5 mm head-feet (45 mm at maximum 1.43× zoom), so head-feet slabs
never exceed 55 mm. Front-back slabs could exceed 55 mm only in rare zoomed and rotated patches of the 7 tumours whose
in-plane diagonal × 1.43 exceeds 55 mm (largest diagonal 42.3 mm). Accepted as is on 12 Oct 2026; no code change.
