"""Check Grad-CAM mathematics, state isolation, geometry and truthful empty maps."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn

from src.explain import align_cam, explanation_text, generate_gradcam, gradcam_target, letterbox_geometry, save_explanation, spatial_summary


class ToyBinaryModel(nn.Module):
    """Known Fake score: mean(red) - mean(blue) - 10."""

    def __init__(self):
        super().__init__()
        self.network = nn.Module()
        self.network.conv_head = nn.Conv2d(3, 2, 1, bias=False)
        self.head = nn.Linear(2, 1)
        with torch.no_grad():
            self.network.conv_head.weight.zero_()
            self.network.conv_head.weight[0, 0, 0, 0] = 1
            self.network.conv_head.weight[1, 2, 0, 0] = 1
            self.head.weight.copy_(torch.tensor([[1., -1.]]))
            self.head.bias.fill_(-10)

    def forward(self, x):
        return self.head(self.network.conv_head(x).mean(dim=(2, 3))).squeeze(-1)


class GradCAMTests(unittest.TestCase):
    def setUp(self):
        self.model = ToyBinaryModel()
        self.inputs = torch.zeros(1, 3, 6, 6)
        self.inputs[:, 0, :3, :3] = 1
        self.inputs[:, 2, 3:, 3:] = 1

    def test_fake_logit_formula_even_for_real_prediction(self):
        result = generate_gradcam(self.model, self.inputs)
        expected = ((self.inputs[:, :1] - self.inputs[:, 2:3]) / 36).relu()
        torch.testing.assert_close(result["cam"], expected)
        self.assertLess(result["fake_probability"], .5)
        self.assertGreater(result["gradient_l1"], 0)
        self.assertEqual(result["activation_shape"], [1, 2, 6, 6])
        # Blue negative evidence is not converted to Fake evidence by absolute values.
        self.assertEqual(float(result["cam"][0, 0, 4, 4]), 0)

    def test_model_flags_weights_gradients_and_hooks_are_preserved(self):
        self.model.train()
        self.model.network.conv_head.eval()
        before_flags = [m.training for m in self.model.modules()]
        before_weights = {k: v.clone() for k, v in self.model.state_dict().items()}
        for parameter in self.model.parameters():
            parameter.grad = torch.ones_like(parameter)
        hook_count = len(self.model.network.conv_head._forward_hooks)
        generate_gradcam(self.model, self.inputs)
        self.assertEqual(before_flags, [m.training for m in self.model.modules()])
        self.assertEqual(hook_count, len(self.model.network.conv_head._forward_hooks))
        for name, weight in self.model.state_dict().items():
            torch.testing.assert_close(weight, before_weights[name], rtol=0, atol=0)
        for parameter in self.model.parameters():
            torch.testing.assert_close(parameter.grad, torch.ones_like(parameter), rtol=0, atol=0)

    def test_frozen_model_inside_inference_mode(self):
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        with torch.inference_mode():
            result = generate_gradcam(self.model, self.inputs)
        self.assertGreater(result["gradient_l1"], 0)

    def test_hook_removed_on_forward_failure(self):
        hooks = len(self.model.network.conv_head._forward_hooks)
        with self.assertRaises(RuntimeError):
            generate_gradcam(self.model, torch.zeros(1, 4, 6, 6))
        self.assertEqual(hooks, len(self.model.network.conv_head._forward_hooks))

    def test_unsupported_target_is_clear_error(self):
        with self.assertRaisesRegex(ValueError, "conv_head"):
            gradcam_target(nn.Linear(2, 1))


class GeometryAndSummaryTests(unittest.TestCase):
    def test_letterbox_matches_pillow_with_odd_padding(self):
        for original_size in [(13, 7), (7, 13), (11, 11), (100, 41)]:
            mask = ImageOps.pad(Image.new("L", original_size, 1), (32, 32), color=0)
            geometry = letterbox_geometry(original_size, 32)
            expected = np.zeros((32, 32), dtype=np.uint8)
            expected[geometry["top"]:geometry["bottom"], geometry["left"]:geometry["right"]] = 1
            np.testing.assert_array_equal(np.asarray(mask), expected)
            cam = torch.from_numpy(np.asarray(mask).copy()).reshape(1, 1, 32, 32).float()
            aligned, _, padding_share = align_cam(cam, original_size, 32)
            self.assertEqual(aligned.shape, (original_size[1], original_size[0]))
            np.testing.assert_allclose(aligned, 1)
            self.assertEqual(padding_share, 0)

    def test_padding_response_does_not_become_an_image_hotspot(self):
        mask = np.asarray(ImageOps.pad(Image.new("L", (13, 7), 1), (32, 32), color=0))
        cam = torch.from_numpy((1 - mask).copy()).reshape(1, 1, 32, 32).float()
        aligned, _, padding_share = align_cam(cam, (13, 7), 32)
        np.testing.assert_array_equal(aligned, np.zeros((7, 13)))
        self.assertEqual(padding_share, 1)

    def test_grid_summary_uses_only_measured_positions(self):
        heatmap = np.zeros((9, 9), dtype=np.float32)
        heatmap[:3, 6:] = 1
        summary = spatial_summary(heatmap)
        self.assertEqual(summary["top_regions"][0]["region"], "top_right")
        self.assertEqual(summary["top_regions"][0]["heatmap_mass_share"], 1)
        text = explanation_text(summary, .1, .5, 0)
        self.assertIn("우상단", text)
        self.assertIn("예측은 Real", text)
        self.assertIn("확률에 대한 기여율이 아닙니다", text)

    def test_zero_and_uniform_maps_do_not_invent_locations(self):
        for value, status in [(0, "no_positive_response"), (1, "uniform_positive_response")]:
            summary = spatial_summary(np.full((8, 8), value, dtype=np.float32))
            self.assertEqual(summary["status"], status)
            self.assertEqual(summary["top_regions"], [])

    def test_zero_overlay_preserves_original_and_artifacts_are_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            original = Image.new("RGB", (13, 7), (40, 80, 120))
            image_path = tmp / "input.png"
            original.save(image_path)
            checkpoint = tmp / "fixture.pth"
            checkpoint.write_bytes(b"hash-only fixture; not loaded")
            result = {"cam": torch.zeros(1, 1, 2, 2), "fake_probability": .1,
                      "target_layer": "network.conv_head", "activation_shape": [1, 2, 2, 2],
                      "gradient_l1": 1., "native_cam_max": 0.}
            report = save_explanation(original, result, .1, .5, 32, image_path, checkpoint, tmp / "result")
            for path in report["files"].values():
                self.assertTrue(Path(path).is_file())
            with Image.open(report["files"]["overlay.png"]) as overlay:
                np.testing.assert_array_equal(overlay, original)
            saved = json.loads(Path(report["files"]["explanation.json"]).read_text(encoding="utf-8"))
            self.assertEqual(saved["target"], "fake_logit")
            self.assertEqual(saved["spatial_summary"]["status"], "no_positive_response")


if __name__ == "__main__":
    unittest.main()
