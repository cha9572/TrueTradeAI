"""Verify optimizer updates and a trained checkpoint round trip."""

import argparse
import torch
from src.model import build_model
from src.utils import load_checkpoint, seed_everything


def main():
    """Validate actual weight changes and restore all model parameters strictly."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="outputs/smoke/checkpoints/best_model.pth")
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    checkpoint = load_checkpoint(args.checkpoint, args.allow_smoke)
    assert checkpoint["optimizer_updates"] > 0
    assert checkpoint["optimizer_state"]["state"]
    seed_everything(checkpoint["config"]["seed"])
    model = build_model(checkpoint["config"]["model"], pretrained=False)
    if not checkpoint["config"]["model"]["pretrained"]:
        name, parameter = next(model.named_parameters())
        initial = parameter.detach().clone()
        trained = checkpoint["model_state"][name]
        assert not torch.equal(initial, trained), "Backbone weights did not change"
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    size = checkpoint["config"]["model"]["input_size"]
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, size, size))
    assert output.shape == (1,) and torch.isfinite(output).all()
    print(f"Checkpoint reload OK: epoch={checkpoint['epoch']}, optimizer_updates={checkpoint['optimizer_updates']}, threshold={checkpoint['threshold']:.8f}")


if __name__ == "__main__":
    main()
