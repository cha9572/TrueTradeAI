"""Predict an entire image with a trained Model 1 checkpoint."""

import argparse
from pathlib import Path

import torch
from PIL import Image

from src.dataset import build_transform
from src.model import build_model
from src.utils import load_checkpoint, select_device


@torch.inference_mode()
def predict(image_path, checkpoint_path, device_name="auto", allow_smoke=False):
    """Return a Fake score and binary decision using stored preprocessing/threshold."""
    checkpoint = load_checkpoint(checkpoint_path, allow_smoke)
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise ValueError("Supported image formats: JPG, JPEG, PNG")
    device = select_device(device_name)
    model = build_model(checkpoint["config"]["model"], pretrained=False).to(device).eval()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    with Image.open(path) as image:
        tensor = build_transform(checkpoint["config"]["model"])(image).unsqueeze(0).to(device)
    probability = float(model(tensor).float().sigmoid().item())
    if not 0 <= probability <= 1:
        raise ValueError("Model produced a non-finite probability")
    if checkpoint["is_smoke"]:
        print("SYNTHETIC PIPELINE CHECK ONLY: not a trained real-data detector.")
    print(f"Fake Probability: {probability:.6f}")
    print(f"Prediction: {'Fake' if probability >= checkpoint['threshold'] else 'Real'}")
    return {"fake_probability": probability, "prediction": int(probability >= checkpoint["threshold"]),
            "threshold": checkpoint["threshold"], "is_smoke": checkpoint["is_smoke"]}


def main():
    """Expose the single-image prediction CLI with clear missing-checkpoint errors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    try:
        predict(args.image, args.checkpoint, args.device, args.allow_smoke)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
