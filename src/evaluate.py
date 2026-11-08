"""Evaluate held-out full images using the checkpoint's frozen validation threshold."""

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import ImageDataset, build_transform, dataset_summary, read_metadata
from src.model import build_model
from src.utils import binary_metrics, check_held_out, is_smoke_data, load_checkpoint, load_config, save_json, select_device


@torch.inference_mode()
def collect_predictions(model, loader, device, threshold):
    """Preserve image/source/type metadata beside each full-image prediction."""
    model.eval()
    rows = []
    for batch in tqdm(loader, desc="test"):
        probabilities = model(batch["image"].to(device, non_blocking=True)).float().sigmoid().cpu().tolist()
        for i, probability in enumerate(probabilities):
            row = {"image_path": batch["image_path"][i], "label": int(batch["label"][i]),
                   "fake_type": batch["fake_type"][i], "fake_probability": probability,
                   "prediction": int(probability >= threshold), "source_id": batch["source_id"][i]}
            if "domain" in batch:
                row["domain"] = batch["domain"][i]
            rows.append(row)
    return pd.DataFrame(rows)


def evaluate(checkpoint_path, config_path=None, output_dir=None, device_name=None, allow_smoke=False):
    """Report global, A/B/C and optional domain metrics without threshold tuning."""
    checkpoint = load_checkpoint(checkpoint_path, allow_smoke)
    config = load_config(config_path) if config_path else checkpoint["config"]
    data, settings = config["data"], config["training"]
    device = select_device(device_name or settings["device"])
    df = read_metadata(data["metadata"], data["root"], data.get("leakage_columns", []), data.get("verify_images", True))
    print(dataset_summary(df))
    smoke = bool(checkpoint["is_smoke"] or is_smoke_data(df))
    if smoke and not allow_smoke:
        raise ValueError("Synthetic evaluation requires --allow-smoke; these metrics are not detector performance")
    if smoke:
        print("SYNTHETIC PIPELINE CHECK ONLY: not real-world detector performance.")
    test_set = ImageDataset(df, "test", build_transform(checkpoint["config"]["model"]))
    check_held_out(test_set.frame, checkpoint)
    loader = DataLoader(test_set, batch_size=settings["batch_size"], shuffle=False,
                        num_workers=settings["num_workers"], pin_memory=device.type == "cuda")
    model = build_model(checkpoint["config"]["model"], pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    threshold = checkpoint["threshold"]
    predictions = collect_predictions(model, loader, device, threshold)
    metrics = binary_metrics(predictions.label, predictions.fake_probability, threshold, predictions.fake_type)
    report = {"checkpoint": str(Path(checkpoint_path).resolve()), "epoch": checkpoint["epoch"],
              "is_smoke": smoke, "threshold_source": checkpoint["threshold_source"],
              "confusion_matrix_order": ["Real=0", "Fake=1"], "overall": metrics, "domains": {}}
    records = [{"scope": "overall", "domain": "", "is_smoke": smoke, **metrics}]
    if "domain" in predictions:
        for domain, group in predictions.groupby("domain", sort=True):
            domain_metrics = binary_metrics(group.label, group.fake_probability, threshold, group.fake_type)
            report["domains"][str(domain)] = domain_metrics
            records.append({"scope": "domain", "domain": str(domain), "is_smoke": smoke, **domain_metrics})
    output = Path(output_dir) if output_dir else Path(settings["output_dir"]) / "predictions"
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / "test_predictions.csv", index=False)
    save_json(output / "test_metrics.json", report)
    pd.json_normalize(records).to_csv(output / "test_metrics.csv", index=False)
    print(f"Test: n={len(predictions)}, threshold={threshold:.8f} (frozen), F1={metrics['f1']:.4f}, ROC-AUC={metrics['roc_auc']}")
    print(f"Saved metrics and image predictions: {output}")
    return report


def main():
    """Expose test evaluation; no test-time threshold override is accepted."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", help="Optional new data/device settings; checkpoint preprocessing stays fixed")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    try:
        evaluate(args.checkpoint, args.config, args.output_dir, args.device, args.allow_smoke)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
