"""Fake-logit Grad-CAM and factual spatial summaries; never a manipulation mask."""

import hashlib
import json
import os
import textwrap
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "outputs" / ".matplotlib"))

DISCLAIMER = (
    "Grad-CAM은 이 모델의 Fake logit에 대한 양의 응답을 거칠게 시각화한 것입니다. "
    "강조 영역이 실제 AI 생성·조작 영역이라는 증거나 segmentation 결과는 아닙니다. "
    "색의 강도는 이미지 내부에서 정규화한 상대값이며 조작 확률이 아닙니다."
)
REGIONS = (("top_left", "좌상단"), ("top_center", "상단 중앙"), ("top_right", "우상단"),
           ("middle_left", "왼쪽 중앙"), ("center", "중앙"), ("middle_right", "오른쪽 중앙"),
           ("bottom_left", "좌하단"), ("bottom_center", "하단 중앙"), ("bottom_right", "우하단"))


def gradcam_target(model):
    """Select timm EfficientNet's last 1x1 conv, before final BN/activation and GAP."""
    layer = getattr(getattr(model, "network", None), "conv_head", None)
    if not isinstance(layer, nn.Conv2d):
        raise ValueError("Default Grad-CAM requires EfficientNet network.conv_head (Conv2d).")
    return "network.conv_head", layer


def generate_gradcam(model, tensor, target_layer=None):
    """Backpropagate the single Fake logit, without accumulating parameter .grad.

    alpha_k = spatial_mean(d(fake_logit)/d(A_k)); CAM = ReLU(sum_k(alpha_k*A_k)).
    Model flags, existing parameter gradients and all hooks are preserved/cleaned.
    """
    if tensor.ndim != 4 or tensor.shape[0] != 1:
        raise ValueError("Grad-CAM requires one image with shape [1,C,H,W].")
    if target_layer is None:
        name, layer = gradcam_target(model)
    else:
        name, layer = target_layer
    states = [(module, module.training) for module in model.modules()]
    captured = []
    handle = layer.register_forward_hook(lambda module, inputs, output: captured.append(output))
    try:
        model.eval()
        with torch.inference_mode(False), torch.enable_grad():
            # Input gradients also enable the graph for a fully frozen backbone.
            inputs = tensor.detach().clone().float().requires_grad_(True)
            with torch.autocast(device_type=inputs.device.type, enabled=False):
                logits = model(inputs)
                if logits.shape != (1,) or not torch.isfinite(logits).all():
                    raise ValueError("Expected one finite Fake logit.")
                if len(captured) != 1 or not isinstance(captured[0], torch.Tensor) or captured[0].ndim != 4:
                    raise ValueError("Target must produce one [N,C,H,W] feature tensor.")
                activations = captured[0]
                gradients, = torch.autograd.grad(logits[0], activations, retain_graph=False)
                if not torch.isfinite(activations).all() or not torch.isfinite(gradients).all():
                    raise ValueError("Non-finite Grad-CAM activation/gradient.")
                weights = gradients.mean(dim=(2, 3), keepdim=True)
                cam = (weights * activations).sum(dim=1, keepdim=True).relu()
            return {"cam": cam.detach().float().cpu(), "fake_logit": float(logits[0].detach()),
                    "fake_probability": float(logits[0].detach().sigmoid()), "target_layer": name,
                    "activation_shape": list(activations.shape),
                    "gradient_l1": float(gradients.detach().abs().sum()),
                    "native_cam_max": float(cam.detach().max())}
    finally:
        handle.remove()
        captured.clear()
        for module, was_training in states:
            module.training = was_training


def letterbox_geometry(original_size, input_size):
    """Match Pillow ImageOps.contain/pad's rounding and centered square placement."""
    width, height = original_size
    if min(width, height, input_size) < 1:
        raise ValueError("Image dimensions must be positive.")
    if width > height:
        resized_w, resized_h = input_size, round(height / width * input_size)
    elif height > width:
        resized_w, resized_h = round(width / height * input_size), input_size
    else:
        resized_w = resized_h = input_size
    if min(resized_w, resized_h) < 1:
        raise ValueError("Image aspect ratio is too extreme for the configured input size.")
    left = round((input_size - resized_w) * 0.5)
    top = round((input_size - resized_h) * 0.5)
    return {"input_size": input_size, "left": left, "top": top,
            "right": left + resized_w, "bottom": top + resized_h,
            "original_width": width, "original_height": height}


def align_cam(cam, original_size, input_size):
    """Remove letterbox padding before resizing CAM to the oriented original image."""
    if cam.ndim != 4 or tuple(cam.shape[:2]) != (1, 1) or not torch.isfinite(cam).all() or (cam < 0).any():
        raise ValueError("Expected one finite nonnegative CAM of shape [1,1,H,W].")
    geometry = letterbox_geometry(original_size, input_size)
    square = F.interpolate(cam.float(), size=(input_size, input_size), mode="bilinear", align_corners=False)
    crop = square[:, :, geometry["top"]:geometry["bottom"], geometry["left"]:geometry["right"]]
    total = float(square.sum())
    padding_share = max(0.0, min(1.0, 1 - float(crop.sum()) / total)) if total > 0 else None
    original = F.interpolate(crop, size=(original_size[1], original_size[0]), mode="bilinear", align_corners=False)[0, 0]
    maximum = float(original.max())
    normalized = original / maximum if maximum > 0 else torch.zeros_like(original)
    return normalized.numpy(), geometry, padding_share


def spatial_summary(heatmap):
    """Describe coordinates only, using normalized CAM mass in nine grid cells."""
    heatmap = np.asarray(heatmap, dtype=np.float32)
    if heatmap.ndim != 2 or heatmap.size == 0 or not np.isfinite(heatmap).all() or (heatmap < 0).any():
        raise ValueError("Expected a finite nonnegative 2D heatmap.")
    total = float(heatmap.sum(dtype=np.float64))
    if total == 0:
        status = "no_positive_response"
    elif float(heatmap.max() - heatmap.min()) <= 1e-6:
        status = "uniform_positive_response"
    else:
        status = "spatial_response"
    height, width = heatmap.shape
    y_edges = np.linspace(0, height, 4, dtype=int)
    x_edges = np.linspace(0, width, 4, dtype=int)
    cells = []
    for index, (name, ko_name) in enumerate(REGIONS):
        row, col = divmod(index, 3)
        x0, x1 = int(x_edges[col]), int(x_edges[col + 1])
        y0, y1 = int(y_edges[row]), int(y_edges[row + 1])
        mass = float(heatmap[y0:y1, x0:x1].sum(dtype=np.float64))
        cells.append({"region": name, "region_ko": ko_name, "bounds_xyxy": [x0, y0, x1, y1],
                      "heatmap_mass_share": mass / total if total > 0 else None})
    ranked = sorted(cells, key=lambda cell: -(cell["heatmap_mass_share"] or 0))
    return {"status": status, "grid": cells,
            "top_regions": [cell for cell in ranked if cell["heatmap_mass_share"] > 0][:3] if status == "spatial_response" else []}


def explanation_text(summary, probability, threshold, padding_share):
    """Generate Korean text from measured positions, without inventing image features."""
    prediction = "Fake" if probability >= threshold else "Real"
    parts = [f"Fake Probability: {probability:.6f}, 저장된 임계값: {threshold:.6f}, 예측: {prediction}."]
    if prediction == "Real":
        parts.append("예측은 Real이지만 이 지도는 Fake logit을 대상으로 계산했습니다.")
    if summary["status"] == "no_positive_response":
        parts.append("원본 이미지에 대응하는 영역에서 양의 Grad-CAM 응답이 관찰되지 않았습니다. 이를 Real임을 입증하는 결과로 해석할 수 없습니다.")
    elif summary["status"] == "uniform_positive_response":
        parts.append("양의 Grad-CAM 응답이 거의 균일하여 특정 위치의 강조를 구분하기 어렵습니다.")
    else:
        top = summary["top_regions"][0]
        parts.append(f"원본을 3×3 격자로 나눠 집계했을 때 {top['region_ko']}의 heatmap 값 합이 가장 컸습니다(전체 양의 heatmap 값 합의 {top['heatmap_mass_share']:.1%}).")
        parts.append("이 비율은 시각화 값의 공간 분포이며 예측 확률에 대한 기여율이 아닙니다.")
    if padding_share is not None and padding_share > 0:
        parts.append(f"정사각형 모델 입력 heatmap 값 합의 {padding_share:.1%}는 letterbox 여백에 있었으며 원본 overlay에서는 제외했습니다.")
    parts.append(DISCLAIMER)
    return "\n".join(parts)


def file_sha256(path):
    """Hash image/checkpoint content for traceable, collision-resistant output names."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_explanation(original, result, probability, threshold, input_size, image_path, checkpoint_path,
                     output_dir=None, is_smoke=False):
    """Save original, heatmap, transparent overlay, figure and machine-readable evidence."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    heatmap, geometry, padding_share = align_cam(result["cam"], original.size, input_size)
    summary = spatial_summary(heatmap)
    text = explanation_text(summary, probability, threshold, padding_share)
    image_sha, checkpoint_sha = file_sha256(image_path), file_sha256(checkpoint_path)
    if output_dir is None:
        output_dir = Path("outputs/explanations") / f"{Path(image_path).stem[:35]}_{image_sha[:8]}_{checkpoint_sha[:8]}"
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    colored = plt.get_cmap("inferno")(heatmap)[..., :3].astype(np.float32)
    # Zero-response pixels remain exactly the original color; opacity scales with CAM.
    alpha = (0.5 * heatmap)[..., None]
    pixels = np.asarray(original.convert("RGB"), dtype=np.float32) / 255
    overlay = np.rint(np.clip((1 - alpha) * pixels + alpha * colored, 0, 1) * 255).astype(np.uint8)
    original.save(output / "original.png")
    Image.fromarray(np.rint(colored * 255).astype(np.uint8)).save(output / "heatmap.png")
    Image.fromarray(overlay).save(output / "overlay.png")
    np.save(output / "heatmap_values.npy", heatmap, allow_pickle=False)
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    try:
        axes[0].imshow(original)
        axes[0].set_title("Original (EXIF oriented)")
        shown = axes[1].imshow(heatmap, cmap="inferno", vmin=0, vmax=1)
        axes[1].set_title("Fake-logit Grad-CAM")
        axes[2].imshow(overlay)
        axes[2].set_title("Grad-CAM overlay")
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        colorbar = fig.colorbar(shown, ax=axes[1], fraction=.045, pad=.04)
        colorbar.set_label("Relative CAM intensity (not probability)", fontsize=9)
        fig.suptitle(f"Fake Probability: {probability:.6f} | Prediction: {'Fake' if probability >= threshold else 'Real'} | Threshold: {threshold:.6f}", fontsize=13)
        if summary["status"] == "spatial_response":
            top = summary["top_regions"][0]
            spatial = f"Largest 3x3 cell CAM mass: {top['region']} ({top['heatmap_mass_share']:.1%}). This is not a share of prediction probability."
        elif summary["status"] == "no_positive_response":
            spatial = "No positive Grad-CAM response in the original image region. This does not prove the image is real."
        else:
            spatial = "Almost uniform positive response; a distinct important location cannot be identified."
        note = spatial + "\nTarget: Fake logit, including Real predictions. Highlighted regions are NOT verified AI manipulation regions."
        if is_smoke:
            note += "\nSYNTHETIC PIPELINE CHECK ONLY"
        fig.text(.5, .035, "\n".join(textwrap.fill(line, width=130) for line in note.splitlines()), ha="center", fontsize=9)
        fig.subplots_adjust(left=.025, right=.98, bottom=.22, top=.87, wspace=.25)
        fig.savefig(output / "explanation.png", dpi=150)
    finally:
        plt.close(fig)
    artifacts = {name: str((output / name).resolve()) for name in
                 ["original.png", "heatmap.png", "overlay.png", "heatmap_values.npy", "explanation.png", "explanation.json", "explanation.txt"]}
    report = {"method": "Grad-CAM", "target": "fake_logit", "target_layer": result["target_layer"],
              "fake_probability": probability, "gradcam_forward_probability": result["fake_probability"],
              "prediction": int(probability >= threshold), "threshold": threshold, "is_smoke": is_smoke,
              "image_path": str(Path(image_path).resolve()), "image_sha256": image_sha,
              "checkpoint_path": str(Path(checkpoint_path).resolve()), "checkpoint_sha256": checkpoint_sha,
              "activation_shape": result["activation_shape"], "gradient_l1": result["gradient_l1"],
              "native_cam_max": result["native_cam_max"], "letterbox": geometry,
              "padding_cam_mass_share": padding_share, "normalization": "ReLU CAM / original-region maximum; per image",
              "spatial_summary": summary, "explanation_ko": text, "disclaimer_ko": DISCLAIMER, "files": artifacts}
    (output / "explanation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    (output / "explanation.txt").write_text(text + "\n", encoding="utf-8")
    return report
