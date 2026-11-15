"""
PatchCore V6 vision inference module for the Smart AI Industrial Inspection Robot.

This module reproduces the V6 inference pipeline from the supplied live-inspection
script, without Streamlit UI dependencies.

Expected deployment layout:
project/
    predict_vision.py
    vision_patchcore_v6_results/
        run_20261021_211025/
            v6_model_package.joblib
"""

from pathlib import Path
from functools import lru_cache

import cv2
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.neighbors import NearestNeighbors
from torchvision import models, transforms


ROOT = Path(__file__).resolve().parent
RUN_DIR = (
    ROOT
    / "vision_patchcore_v6_results"
    / "run_20261021_211025"
)
PACKAGE_PATH = RUN_DIR / "v6_model_package.joblib"
OUTPUT_DIR = ROOT / "vision_inference_results"

IMAGE_SIZE = 256
K_NEIGHBORS = 3
SCORE_PERCENTAGE = 5
DEFAULT_THRESHOLD_PERCENTILE = "P80"
DEVICE = torch.device("cpu")

torch.set_num_threads(4)

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class FeatureExtractor(nn.Module):
    """ResNet18 layer2 + layer3 feature extractor used by PatchCore V6."""

    def __init__(self):
        super().__init__()

        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        for parameter in self.parameters():
            parameter.requires_grad = False

        self.eval()

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)

        f2 = self.layer2(x)
        f3 = self.layer3(f2)

        f3 = F.interpolate(
            f3,
            size=f2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        f2 = F.avg_pool2d(f2, 3, stride=1, padding=1)
        f3 = F.avg_pool2d(f3, 3, stride=1, padding=1)

        return torch.cat([f2, f3], dim=1)


@lru_cache(maxsize=1)
def load_package():
    if not PACKAGE_PATH.exists():
        raise FileNotFoundError(
            f"PatchCore V6 package not found: {PACKAGE_PATH}"
        )

    package = joblib.load(PACKAGE_PATH)

    required = {"feature_mean", "feature_std", "memory_bank", "thresholds"}
    missing = required.difference(package.keys())
    if missing:
        raise ValueError(
            f"V6 package is missing required fields: {sorted(missing)}"
        )

    thresholds = package["thresholds"]
    if "top_5_percent" not in thresholds:
        raise ValueError("V6 package has no top_5_percent thresholds.")

    if DEFAULT_THRESHOLD_PERCENTILE not in thresholds["top_5_percent"]:
        raise ValueError(
            f"V6 package has no {DEFAULT_THRESHOLD_PERCENTILE} threshold."
        )

    return package


@lru_cache(maxsize=1)
def load_extractor():
    model = FeatureExtractor().to(DEVICE)
    model.eval()
    return model


@lru_cache(maxsize=1)
def load_neighbor_index():
    package = load_package()
    memory = np.ascontiguousarray(package["memory_bank"], dtype=np.float32)

    if memory.ndim != 2 or memory.shape[0] == 0:
        raise ValueError("V6 memory_bank must be a non-empty 2D array.")

    neighbor = NearestNeighbors(
        n_neighbors=min(K_NEIGHBORS, len(memory)),
        algorithm="brute",
        metric="euclidean",
        n_jobs=2,
    )
    neighbor.fit(memory)
    return neighbor


def _extract_feature(image: Image.Image, model: nn.Module) -> np.ndarray:
    image_tensor = IMAGE_TRANSFORM(image.convert("RGB")).unsqueeze(0)

    with torch.inference_mode():
        feature = model(image_tensor.to(DEVICE))

    return feature.cpu().numpy()[0].astype(np.float32)


def _inspect_image(image: Image.Image, model, package, neighbor):
    feature = _extract_feature(image, model)
    channels, height, width = feature.shape

    # V6 patch layout: [channels, pixels] -> [pixels, channels].
    patches = feature.reshape(channels, -1).T.copy()

    mean = np.asarray(package["feature_mean"], dtype=np.float32)
    std = np.asarray(package["feature_std"], dtype=np.float32)

    if len(mean) != patches.shape[1] or len(std) != patches.shape[1]:
        raise ValueError(
            "V6 feature-statistics dimension mismatch: "
            f"live={patches.shape[1]}, mean={len(mean)}, std={len(std)}"
        )

    # Match the saved V6 feature standardization.
    patches = ((patches - mean) / std).astype(np.float32)

    distances, _ = neighbor.kneighbors(patches, return_distance=True)
    patch_scores = distances.mean(axis=1)

    anomaly_map = patch_scores.reshape(height, width)
    anomaly_map = gaussian_filter(anomaly_map, sigma=0.5).astype(np.float32)

    flat = anomaly_map.ravel()
    count = max(1, int(np.ceil(len(flat) * SCORE_PERCENTAGE / 100)))
    score = float(np.partition(flat, -count)[-count:].mean())

    return score, anomaly_map


def _make_visuals(image: Image.Image, anomaly_map: np.ndarray):
    original = image.convert("RGB")

    resized_map = cv2.resize(
        anomaly_map,
        (original.width, original.height),
        interpolation=cv2.INTER_LINEAR,
    )

    low = float(resized_map.min())
    high = float(resized_map.max())
    normalized = (resized_map - low) / (high - low + 1e-8)

    heat_uint8 = np.uint8(np.clip(normalized * 255, 0, 255))
    heat_bgr = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(
        np.asarray(original),
        0.60,
        heat_rgb,
        0.40,
        0,
    )

    return Image.fromarray(heat_uint8), Image.fromarray(overlay)


def predict(image_path, threshold_percentile=DEFAULT_THRESHOLD_PERCENTILE):
    """
    Run PatchCore V6 on an image.

    Returns a dictionary compatible with the dashboard's existing result
    handling, including legacy ensemble_score/threshold aliases.
    """
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    package = load_package()
    try:
        threshold = float(
            package["thresholds"]["top_5_percent"][threshold_percentile]
        )
    except KeyError as exc:
        raise ValueError(
            f"Threshold percentile {threshold_percentile!r} is unavailable "
            "in the saved V6 package."
        ) from exc

    image = Image.open(image_path).convert("RGB")
    model = load_extractor()
    neighbor = load_neighbor_index()

    score, anomaly_map = _inspect_image(image, model, package, neighbor)
    heatmap, overlay = _make_visuals(image, anomaly_map)

    prediction = "DEFECTIVE" if score > threshold else "NORMAL"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    heatmap_path = OUTPUT_DIR / f"{image_path.stem}_v6_heatmap.png"
    overlay_path = OUTPUT_DIR / f"{image_path.stem}_v6_overlay.png"

    heatmap.save(heatmap_path)
    overlay.save(overlay_path)

    return {
        "model": "PatchCore V6",
        "status": "SUCCESS",
        "prediction": prediction,
        "anomaly_score": score,
        "threshold": threshold,
        "threshold_percentile": threshold_percentile,
        "score_method": "top_5_percent",
        "heatmap_path": str(heatmap_path),
        "overlay_path": str(overlay_path),
        # Compatibility aliases for the existing dashboard's older schema.
        "ensemble_score": score,
        "visualization_path": str(overlay_path),
        "source_image": str(image_path),
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="PatchCore V6 image inference")
    parser.add_argument("--image", required=True, help="Path to an image")
    parser.add_argument(
        "--threshold",
        default=DEFAULT_THRESHOLD_PERCENTILE,
        choices=["P80", "P90", "P95"],
    )
    args = parser.parse_args()

    result = predict(args.image, args.threshold)
    print(json.dumps(result, indent=2))
