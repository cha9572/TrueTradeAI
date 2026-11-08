"""Predict an entire image with a trained Model 1 checkpoint."""

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageOps

from src.dataset import build_transform
from src.model import build_model
from src.utils import load_checkpoint, select_device


def predict(image_path, checkpoint_path, device_name="auto", allow_smoke=False, explain=False, explain_dir=None):
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
        original = ImageOps.exif_transpose(image).convert("RGB")
        tensor = build_transform(checkpoint["config"]["model"])(original).unsqueeze(0).to(device)
    with torch.inference_mode():
        probability = float(model(tensor).float().sigmoid().item())
    if not 0 <= probability <= 1:
        raise ValueError("Model produced a non-finite probability")
    if checkpoint["is_smoke"]:
        print("SYNTHETIC PIPELINE CHECK ONLY: not a trained real-data detector.")
    print(f"Fake Probability: {probability:.6f}")
    print(f"Prediction: {'Fake' if probability >= checkpoint['threshold'] else 'Real'}")
    prediction = {"fake_probability": probability, "prediction": int(probability >= checkpoint["threshold"]),
                  "threshold": checkpoint["threshold"], "is_smoke": checkpoint["is_smoke"]}
    if explain:
        from src.explain import generate_gradcam, save_explanation
        result = generate_gradcam(model, tensor)
        if abs(result["fake_probability"] - probability) > 1e-5:
            raise RuntimeError("Grad-CAM forward score differs from prediction; explanation not saved.")
        report = save_explanation(original, result, probability, checkpoint["threshold"],
                                  checkpoint["config"]["model"]["input_size"], path, checkpoint_path,
                                  explain_dir, checkpoint["is_smoke"])
        prediction["explanation"] = report
        print(report["explanation_ko"])
        print(f"Explanation saved: {report['files']['explanation.png']}")
    return prediction


def main():
    """Expose the single-image prediction CLI with clear missing-checkpoint errors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--explain", action="store_true", help="Save Fake-logit Grad-CAM; not a manipulation mask")
    parser.add_argument("--explain-dir", help="Output folder; default outputs/explanations/<image+checkpoint hashes>")
    args = parser.parse_args()
    if args.explain_dir and not args.explain:
        parser.error("--explain-dir requires --explain")
    try:
        predict(args.image, args.checkpoint, args.device, args.allow_smoke, args.explain, args.explain_dir)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
