"""Train full-image Real/Fake classification without using test data for selection."""

import argparse
import hashlib
import shutil
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import ImageDataset, build_transform, dataset_summary, read_metadata
from src.model import build_model
from src.utils import binary_metrics, choose_threshold, identity_hashes, is_smoke_data, load_config, plot_history, save_json, seed_everything, seed_worker, select_device


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, amp=False):
    """Run training or validation and collect unthresholded scores for metrics."""
    training = optimizer is not None
    model.train(training)
    total_loss, labels, probabilities, updates = 0.0, [], [], 0
    progress = tqdm(loader, desc="train" if training else "val", leave=False)
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["label"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(images)
                loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss. Check images, learning rate and AMP settings.")
            if training:
                previous_scale = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                updates += int(scaler.get_scale() >= previous_scale)
        total_loss += loss.item() * len(targets)
        labels.extend(targets.detach().cpu().tolist())
        probabilities.extend(logits.detach().float().sigmoid().cpu().tolist())
        progress.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(labels), labels, probabilities, updates


def train(config_path):
    """Train with validation F1 selection, best/last checkpoints and early stopping."""
    config = load_config(config_path)
    settings, data = config["training"], config["data"]
    seed_everything(config["seed"])
    device = select_device(settings["device"])
    df = read_metadata(data["metadata"], data["root"], data.get("leakage_columns", []), data.get("verify_images", True))
    print(dataset_summary(df))
    smoke = bool(is_smoke_data(df))
    if smoke and not settings.get("allow_smoke", False):
        raise ValueError("Synthetic fixtures are blocked in baseline training; use configs/smoke.yaml")
    if smoke:
        print("SYNTHETIC PIPELINE CHECK ONLY: metrics are not detector performance.")
    train_set = ImageDataset(df, "train", build_transform(config["model"], train=True))
    val_set = ImageDataset(df, "val", build_transform(config["model"]))
    if len(train_set) < 2:
        raise ValueError("At least two training images required for backbone BatchNorm")
    if settings["tune_threshold"] and val_set.frame.label.nunique() != 2:
        raise ValueError("Validation requires both Real and Fake to tune a threshold")
    output = Path(settings["output_dir"])
    if any((output / "checkpoints" / name).exists() for name in ["best_model.pth", "last_model.pth"]):
        raise FileExistsError("Output already contains checkpoints. Choose a fresh training.output_dir.")
    for name in ["checkpoints", "logs", "figures", "predictions"]:
        (output / name).mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    shutil.copyfile(data["metadata"], output / "metadata_snapshot.csv")
    from scripts.check_gpu import inspect_environment
    environment = inspect_environment()
    save_json(output / "environment.json", environment)
    protected = identity_hashes(df.loc[df.split.isin(["train", "val"])], data.get("leakage_columns", []))
    generator = torch.Generator().manual_seed(config["seed"])
    batch_size = min(settings["batch_size"], len(train_set))
    drop_last = len(train_set) % batch_size == 1
    if drop_last:
        print("Dropping final singleton training batch for BatchNorm.")
    common = {"num_workers": settings["num_workers"], "pin_memory": device.type == "cuda", "worker_init_fn": seed_worker}
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=generator, drop_last=drop_last, **common)
    val_loader = DataLoader(val_set, batch_size=settings["batch_size"], shuffle=False, **common)
    # No test loader: test is used only by the separate evaluation entry point.
    model = build_model(config["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings["epochs"])
    amp = bool(settings["amp"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=settings.get("amp_init_scale", 1024))
    criterion = nn.BCEWithLogitsLoss()
    best_f1, stale, history = -1.0, 0, []
    print(f"Training: AMP={amp}, batch_size={batch_size}, pretrained={config['model']['pretrained']}")
    for epoch in range(1, settings["epochs"] + 1):
        lr = optimizer.param_groups[0]["lr"]
        train_loss, train_y, train_p, updates = run_epoch(model, train_loader, criterion, device, optimizer, scaler, amp)
        if updates == 0:
            raise ValueError("No optimizer updates: AMP skipped every batch. Lower training.amp_init_scale or set training.amp: false and rerun in a fresh output directory.")
        # Use float32 validation, matching evaluation and single-image inference.
        val_loss, val_y, val_p, _ = run_epoch(model, val_loader, criterion, device, amp=False)
        threshold = choose_threshold(val_y, val_p) if settings["tune_threshold"] else settings["threshold"]
        train_metrics = binary_metrics(train_y, train_p, threshold)
        val_metrics = binary_metrics(val_y, val_p, threshold)
        row = {"epoch": epoch, "optimizer_updates": updates, "learning_rate": lr, "threshold": threshold, "train_loss": train_loss, "val_loss": val_loss}
        for split, metrics in [("train", train_metrics), ("val", val_metrics)]:
            row.update({f"{split}_{key}": metrics[key] for key in ["accuracy", "precision", "recall", "f1"]})
        history.append(row)
        improved = val_metrics["f1"] > best_f1
        best_f1, stale = (val_metrics["f1"], 0) if improved else (best_f1, stale + 1)
        scheduler.step()
        checkpoint = {
            "format_version": 1, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(), "scaler_state": scaler.state_dict(),
            "epoch": epoch, "optimizer_updates": updates, "best_f1": best_f1, "stale_epochs": stale, "threshold": float(threshold),
            "threshold_source": "validation_f1" if settings["tune_threshold"] else "fixed_config",
            "config": config, "is_smoke": smoke, "environment": environment,
            "protected_identities": protected, "metadata_sha256": hashlib.sha256(Path(data["metadata"]).read_bytes()).hexdigest(),
            "torch_rng_state": torch.get_rng_state(), "loader_rng_state": generator.get_state(),
        }
        torch.save(checkpoint, output / "checkpoints" / "last_model.pth")
        if improved:
            torch.save(checkpoint, output / "checkpoints" / "best_model.pth")
        pd.DataFrame(history).to_csv(output / "logs" / "training.csv", index=False)
        plot_history(history, output / "figures" / "training_curves.png")
        print(f"Epoch {epoch}/{settings['epochs']}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_F1={val_metrics['f1']:.4f}, threshold={threshold:.6f}, best={improved}")
        if device.type == "cuda":
            print(f"GPU memory allocated={torch.cuda.memory_allocated()/2**20:.0f} MiB; peak={torch.cuda.max_memory_allocated()/2**20:.0f} MiB")
        if stale >= settings["patience"]:
            print(f"Early stopping: validation F1 did not improve for {stale} epochs")
            break
    print(f"Saved best/last checkpoints and logs to {output}")


def main():
    """Expose training as python -m src.train."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()
    try:
        train(args.config)
    except torch.cuda.OutOfMemoryError:
        parser.exit(1, "CUDA out of memory: reduce training.batch_size in the config.\n")
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
