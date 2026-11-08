"""Configuration, reproducibility, metrics and checkpoint helpers."""

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "outputs" / ".matplotlib"))


def load_config(path):
    """Read YAML and resolve all filesystem settings relative to the project root."""
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    project = path.parent.parent
    for section, key in [("data", "root"), ("data", "metadata"), ("training", "output_dir")]:
        config[section][key] = str((project / config[section][key]).resolve())
    train = config["training"]
    if train["epochs"] < 1 or train["patience"] < 1 or train["batch_size"] < 2 or train["num_workers"] < 0:
        raise ValueError("epochs/patience >=1, batch_size >=2 and num_workers >=0 required")
    if not 0 <= train["threshold"] <= 1 or train["learning_rate"] <= 0 or train["weight_decay"] < 0:
        raise ValueError("Invalid threshold, learning_rate or weight_decay")
    return config


def seed_everything(seed):
    """Seed Python, NumPy, CPU/CUDA torch and deterministic CUDA kernels."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id):
    """Initialize each loader worker reproducibly, including Windows spawn workers."""
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def select_device(requested="auto"):
    """Choose CUDA when available and clearly diagnose CPU fallback."""
    if requested not in {"auto", "cuda", "cpu"}:
        raise ValueError("device must be auto, cuda, or cpu")
    available = torch.cuda.is_available()
    if requested == "cuda" and not available:
        raise RuntimeError("CUDA requested but unavailable. Run python -m scripts.check_gpu.")
    device = torch.device("cuda" if available and requested != "cpu" else "cpu")
    if device.type == "cuda":
        print(f"Device: {torch.cuda.get_device_name(0)}; CUDA runtime={torch.version.cuda}")
    else:
        reason = "CPU explicitly selected" if requested == "cpu" else ("CPU-only PyTorch build" if torch.version.cuda is None else "CUDA driver/device unavailable")
        print(f"Device: CPU ({reason}). Full 384px training can be very slow; NVIDIA CUDA is recommended.")
    return device


def is_smoke_data(df):
    """Recognize the explicit synthetic fixture marker."""
    return "is_smoke" in df and df.is_smoke.astype(str).str.lower().isin(["true", "1", "yes"]).any()


def binary_metrics(labels, probabilities, threshold=0.5, fake_types=None):
    """Compute robust binary metrics; undefined AUC/type recall is JSON null."""
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if labels.size == 0 or labels.shape != probabilities.shape:
        raise ValueError("Nonempty aligned labels/probabilities required")
    if not np.isfinite(probabilities).all() or not ((0 <= probabilities) & (probabilities <= 1)).all():
        raise ValueError("Probabilities must be finite values in [0,1]")
    predictions = (probabilities >= threshold).astype(int)
    result = {
        "count": int(labels.size), "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)) if np.unique(labels).size == 2 else None,
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }
    if fake_types is not None:
        types = np.asarray(fake_types)
        result["fake_type_recall"] = {}
        for kind in ["A", "B", "C"]:
            mask = (types == kind) & (labels == 1)
            count = int(mask.sum())
            result["fake_type_recall"][kind] = {"count": count, "recall": float(predictions[mask].mean()) if count else None}
    return result


def choose_threshold(labels, probabilities):
    """Maximize validation F1 exactly over score boundaries; prefer ties near 0.5."""
    if np.unique(labels).size != 2:
        raise ValueError("Validation requires both Real and Fake to tune a threshold")
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    thresholds = np.append(thresholds, 0.5)
    f1 = np.append(f1, binary_metrics(labels, probabilities, 0.5)["f1"])
    candidates = thresholds[np.isclose(f1, f1.max(), rtol=0, atol=1e-12)]
    return float(sorted(candidates, key=lambda t: (abs(t - 0.5), t))[0])


def save_json(path, value):
    """Write strict JSON without NaN/Infinity values."""
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def identity_hashes(df, leakage_columns):
    """Store split identity sets to prevent held-out evaluation on seen sources."""
    result = {}
    for col in dict.fromkeys(["source_id", *leakage_columns, "_resolved_path"]):
        if col not in df:
            continue
        values = df[col].astype(str).str.strip()
        if col == "_resolved_path":
            values = values.map(os.path.normcase)
        result[col] = sorted({hashlib.sha256(value.encode("utf-8")).hexdigest() for value in values if value})
    return result


def check_held_out(df, checkpoint):
    """Reject test identities that overlap checkpoint training/validation identities."""
    protected = checkpoint["protected_identities"]
    missing = [col for col, values in protected.items() if values and col not in df]
    if missing:
        raise ValueError(f"Cannot verify evaluation leakage: missing identity columns {missing}")
    current = identity_hashes(df, [c for c in protected if c != "_resolved_path"])
    for column, old_hashes in protected.items():
        if set(old_hashes) & set(current.get(column, [])):
            raise ValueError(f"Evaluation leakage against checkpoint train/val: {column}")


def load_checkpoint(path, allow_smoke=False):
    """Load only trained project checkpoints with a valid frozen threshold."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Trained checkpoint not found: {path}. Train the model first.")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    required = {"model_state", "config", "threshold", "epoch", "is_smoke", "protected_identities", "optimizer_updates"}
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint) or checkpoint["epoch"] < 1:
        raise ValueError("Not a trained TrueTrade checkpoint")
    if not np.isfinite(checkpoint["threshold"]) or not 0 <= checkpoint["threshold"] <= 1:
        raise ValueError("Invalid checkpoint threshold")
    if checkpoint["optimizer_updates"] < 1:
        raise ValueError("Checkpoint has no optimizer updates")
    if checkpoint["is_smoke"] and not allow_smoke:
        raise ValueError("SYNTHETIC smoke checkpoint: use --allow-smoke only for pipeline checks")
    return checkpoint


def plot_history(rows, output):
    """Save train/validation loss and F1 curves using a headless renderer."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = [r["epoch"] for r in rows]
    for ax, metric in zip(axes, ["loss", "f1"]):
        for split in ["train", "val"]:
            ax.plot(epochs, [r[f"{split}_{metric}"] for r in rows], label=split, marker="o")
        ax.set(xlabel="Epoch", ylabel=metric.upper())
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
