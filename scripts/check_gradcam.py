"""Integration check: trained EfficientNet forward/backward, score and artifact parity."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from src.explain import file_sha256
from src.predict import predict


def main():
    """Verify normal/explained predictions match and the checkpoint stays unchanged."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--output-dir", default="outputs/explanations/integration_check")
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    before = file_sha256(args.checkpoint)
    normal = predict(args.image, args.checkpoint, args.device, args.allow_smoke)
    explained = predict(args.image, args.checkpoint, args.device, args.allow_smoke, True, args.output_dir)
    assert abs(normal["fake_probability"] - explained["fake_probability"]) <= 1e-6
    assert normal["prediction"] == explained["prediction"]
    assert normal["threshold"] == explained["threshold"]
    report = explained["explanation"]
    assert report["target"] == "fake_logit" and report["target_layer"] == "network.conv_head"
    assert report["gradient_l1"] > 0 and np.isfinite(report["gradient_l1"])
    assert report["activation_shape"][0:2] == [1, 1280]
    values = np.load(report["files"]["heatmap_values.npy"], allow_pickle=False)
    with Image.open(args.image) as image:
        expected_size = ImageOps.exif_transpose(image).size
    assert values.shape == (expected_size[1], expected_size[0])
    assert np.isfinite(values).all() and values.min() >= 0 and values.max() <= 1
    for path in report["files"].values():
        assert Path(path).is_file()
    for name in ["original.png", "heatmap.png", "overlay.png"]:
        with Image.open(report["files"][name]) as image:
            assert image.size == expected_size
    assert file_sha256(args.checkpoint) == before
    verification = {"image": args.image, "device": args.device, "checkpoint_unchanged": True,
                    "prediction_unchanged": True, "fake_probability": normal["fake_probability"],
                    "prediction": normal["prediction"], "activation_shape": report["activation_shape"],
                    "gradient_l1": report["gradient_l1"], "spatial_status": report["spatial_summary"]["status"],
                    "original_size": list(expected_size), "heatmap_size_matches": True,
                    "target_always_fake_logit": True, "artifacts_verified": True}
    (Path(args.output_dir) / "verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    print("Actual EfficientNet Grad-CAM integration check PASSED")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
