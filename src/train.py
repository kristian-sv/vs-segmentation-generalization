"""Manual T1 model: learn from training patches and select using validation only."""
import csv
import random
from datetime import datetime
import numpy as np
import torch
import wandb
from torch.amp import GradScaler, autocast
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from monai.networks.utils import one_hot
from monai.utils import set_determinism
from config import ROOT
from dataset import get_loader
from run_logging import snapshot_run


def seed_everything(seed):
    set_determinism(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(settings):
    return UNet(spatial_dims=3, in_channels=1, out_channels=2,
                channels=settings['channels'], strides=(2, 2, 2, 2),
                norm='instance', act='LeakyReLU', dropout=settings['dropout'])


def train_model(model, train_loader, val_loader, device, run_dir, settings):
    loss_fn = DiceFocalLoss(softmax=True, to_onehot_y=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=settings["warmup_epochs"])
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings["epochs"] - settings["warmup_epochs"])
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[settings["warmup_epochs"]])
    scaler = GradScaler("cuda")
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    best_dice = -1.0
    wait = 0
    log_rows = []

    for epoch in range(1, settings["epochs"] + 1):
        # — Train —
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            imgs = batch["image"].to(device)
            labs = batch["label"].to(device)

            optimizer.zero_grad()
            with autocast("cuda"):
                preds = model(imgs)
                loss = loss_fn(preds, labs)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        scheduler.step()
        mean_loss = epoch_loss / len(train_loader)

        # — Validate (sliding window) —
        model.eval()
        dice_metric.reset()
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                labs = batch["label"].to(device)
                preds = sliding_window_inference(
                    imgs, roi_size=settings["patch_size"],
                    sw_batch_size=settings["sw_batch_size"], predictor=model,
                    overlap=settings["sw_overlap"],
                )
                pred_oh = one_hot(preds.argmax(dim=1, keepdim=True), num_classes=2)
                lab_oh = one_hot(labs.long(), num_classes=2)
                dice_metric(pred_oh, lab_oh)

        val_dice = dice_metric.aggregate().item()
        current_lr = optimizer.param_groups[0]["lr"]
        wandb.log({"train_loss": mean_loss, "val_dice": val_dice, "lr": current_lr, "epoch": epoch})
        print(f'Epoch {epoch}/{settings["epochs"]}  loss={mean_loss:.4f}  val_dice={val_dice:.4f}')
        log_rows.append({"epoch": epoch, "train_loss": mean_loss, "val_dice": val_dice})

        # — Checkpoint —
        if val_dice > best_dice:
            best_dice = val_dice
            wait = 0
            torch.save(model.state_dict(), run_dir / "best_model.pt")
        else:
            wait += 1
            if wait >= settings["patience"]:
                print(f"Early stopping at epoch {epoch}")
                break

    # Save training log
    with open(run_dir / "training_log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_dice"])
        writer.writeheader()
        writer.writerows(log_rows)

    return best_dice


def main(settings, lock):
    seed_everything(settings['seed'])
    if not torch.cuda.is_available():
        raise RuntimeError('Training needs an allocated GPU; use experiments/train.slurm')
    device = torch.device('cuda')
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = ROOT / 'outputs/t1/manual' / f"{settings['experiment_name']}_{stamp}"
    run_dir.mkdir(parents=True)  # never reuse another run's output directory
    snapshot_run(run_dir, settings)
    wandb.init(project='vs-segmentation', config=settings, dir=str(run_dir),
               name=settings['experiment_name'])
    model = build_model(settings).to(device)
    train_loader = get_loader(lock, 'train', settings)
    val_loader = get_loader(lock, 'val', settings)
    best = train_model(model, train_loader, val_loader, device, run_dir, settings)
    wandb.finish()
    print(f'Best validation Dice: {best:.4f}. Outputs: {run_dir}')
    print('Development complete. Test/external evaluation is a separate command.')
