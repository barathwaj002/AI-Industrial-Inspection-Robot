"""
V6 PatchCore vision predictor for the integrated AI Industrial Inspection app.

This module reproduces the inference pipeline from vision_inspection_app.py:
256x256 ImageNet preprocessing -> ResNet18 layer2/layer3 features ->
saved feature standardization -> k-nearest-neighbour distances ->
Gaussian smoothing -> top-5% anomaly score -> saved P80/P90/P95 threshold.

Keep this file beside inspection_dashboard.py and multimodal_inspection.py.
"""

from functools import lru_cache
from pathlib import Path

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


# ---------------------------------------------------------------------
# Paths and V6 settings
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent

PACKAGE_PATH = (
    ROOT
    / "vision_patchcore_v6_results"
    / "run_20261021_211025"
    / "v6_model_package.joblib"
)

OUTPUT_DIR = ROOT / "vision_inference_results"

IMAGE_SIZE = 256
K_NEIGHBORS = 3
SCORE_PERCENTAGE = 5

DEVICE = torch.device("cpu")
torch.set_num_threads(4)

PREPROCESS = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ---------------------------------------------------------------------
# Saved V6 package
# ---------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_package():
    if not PACKAGE_PATH.is_file():
        raise FileNotFoundError(
            "V6 model package was not found. Expected file: "
            f"{PACKAGE_PATH}"
        )

    package = joblib.load(PACKAGE_PATH)

    required = ("feature_mean", "feature_std", "memory_bank", "thresholds")
    missing = [key for key in required if key not in package]
    if missing:
        raise ValueError(f"V6 package is missing fields: {missing}")

    thresholds = package["thresholds"]
    if "top_5_percent" not in thresholds:
        raise ValueError("V6 package has no top_5_percent thresholds.")

    for percentile in ("P80", "P90", "P95"):
        if percentile not in thresholds["top_5_percent"]:
            raise ValueError(
                f"V6 package is missing the {percentile} threshold."
            )

    memory = np.asarray(package["memory_bank"])
    if memory.ndim != 2 or memory.shape[0] == 0:
        raise ValueError(
            f"Invalid V6 memory bank shape: {memory.shape}"
        )

    return package


# ---------------------------------------------------------------------
# Exact V6 ResNet18 feature extractor
# ---------------------------------------------------------------------

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        # Same pretrained ResNet18 weights as the original working app.
        backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

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

        f2 = F.avg_pool2d(
            f2, kernel_size=3, stride=1, padding=1
        )
        f3 = F.avg_pool2d(
            f3, kernel_size=3, stride=1, padding=1
        )

        return torch.cat([f2, f3], dim=1)


@lru_cache(maxsize=1)
def load_extractor():
    model = FeatureExtractor().to(DEVICE)
    model.eval()
    return model


# ---------------------------------------------------------------------
# Exact V6 nearest-neighbour memory bank
# ---------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_neighbor_index():
    package = load_package()

    memory = np.ascontiguousarray(
        package["memory_bank"],
        dtype=np.float32,
    )

    k = min(
        int(package.get("k_neighbors", K_NEIGHBORS)),
        len(memory),
    )

    if k < 1:
        raise ValueError("V6 memory bank contains no neighbour vectors.")

    neighbor = NearestNeighbors(
        n_neighbors=k,
        algorithm="brute",
        metric="euclidean",
        n_jobs=2,
    )
    neighbor.fit(memory)
    return neighbor


# ---------------------------------------------------------------------
# V6 threshold
# ---------------------------------------------------------------------

def get_threshold(percentile="P80"):
    key = str(percentile).upper()
    if not key.startswith("P"):
        key = f"P{key}"

    if key not in ("P80", "P90", "P95"):
        raise ValueError("Threshold must be P80, P90, or P95.")

    try:
        return float(
            load_package()["thresholds"]["top_5_percent"][key]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Could not read {key} threshold from the V6 package."
        ) from exc


# ---------------------------------------------------------------------
# Exact V6 scoring pipeline
# ---------------------------------------------------------------------

def inspect_image(image):
    """
    Returns (top_5_percent_score, smoothed_anomaly_map).

    The scoring operations intentionally match the original V6 app.
    """
    package = load_package()
    extractor = load_extractor()
    neighbor = load_neighbor_index()

    tensor = PREPROCESS(image.convert("RGB")).unsqueeze(0)

    with torch.inference_mode():
        feature = extractor(tensor.to(DEVICE))

    feature = feature.cpu().numpy()[0].astype(np.float32)

    channels, height, width = feature.shape
    patches = feature.reshape(channels, -1).T.copy()

    mean = np.asarray(
        package["feature_mean"], dtype=np.float32
    ).reshape(-1)
    std = np.asarray(
        package["feature_std"], dtype=np.float32
    ).reshape(-1)

    if patches.shape[1] != len(mean):
        raise ValueError(
            "V6 feature dimension mismatch: "
            f"live={patches.shape[1]}, saved mean={len(mean)}."
        )
    if patches.shape[1] != len(std):
        raise ValueError(
            "V6 feature dimension mismatch: "
            f"live={patches.shape[1]}, saved std={len(std)}."
        )
    if np.any(std == 0):
        raise ValueError(
            "Saved V6 feature_std contains zero values; "
            "cannot standardize features safely."
        )

    patches = ((patches - mean) / std).astype(np.float32)

    distances, _ = neighbor.kneighbors(
        patches,
        return_distance=True,
    )
    patch_scores = distances.mean(axis=1)

    anomaly_map = patch_scores.reshape(height, width)
    anomaly_map = gaussian_filter(
        anomaly_map, sigma=0.5
    ).astype(np.float32)

    flat = anomaly_map.ravel()
    count = max(
        1,
        int(np.ceil(len(flat) * SCORE_PERCENTAGE / 100)),
    )
    score = float(np.partition(flat, -count)[-count:].mean())

    return score, anomaly_map


# ---------------------------------------------------------------------
# V6 heatmap and overlay
# ---------------------------------------------------------------------

def make_visuals(image, anomaly_map):
    original = image.convert("RGB")

    resized_map = cv2.resize(
        anomaly_map,
        (original.width, original.height),
        interpolation=cv2.INTER_LINEAR,
    )

    low = float(resized_map.min())
    high = float(resized_map.max())

    # Normalization is for visualization only, not model scoring.
    normalized = (resized_map - low) / (high - low + 1e-8)
    heat_uint8 = np.uint8(np.clip(normalized * 255, 0, 255))

    heat_bgr = cv2.applyColorMap(
        heat_uint8,
        cv2.COLORMAP_JET,
    )
    heat_rgb = cv2.cvtColor(
        heat_bgr,
        cv2.COLOR_BGR2RGB,
    )

    overlay = cv2.addWeighted(
        np.asarray(original),
        0.60,
        heat_rgb,
        0.40,
        0,
    )

    return Image.fromarray(heat_rgb), Image.fromarray(overlay)


# ---------------------------------------------------------------------
# Public function called by multimodal_inspection.py
# ---------------------------------------------------------------------

def predict(image_path, threshold_percentile="P80"):
    """
    Predict a local image file and return the keys expected by the
    integrated dashboard and multimodal report generator.
    """
    source_path = Path(image_path)

    if not source_path.is_file():
        raise FileNotFoundError(
            f"Vision input image was not found: {source_path}"
        )

    percentile = str(threshold_percentile).upper()
    if not percentile.startswith("P"):
        percentile = f"P{percentile}"

    threshold = get_threshold(percentile)

    try:
        with Image.open(source_path) as opened:
            image = opened.convert("RGB")
    except Exception as exc:
        raise ValueError(
            f"Could not open the uploaded vision image: {exc}"
        ) from exc

    score, anomaly_map = inspect_image(image)
    heatmap, overlay = make_visuals(image, anomaly_map)

    prediction = "DEFECTIVE" if score > threshold else "NORMAL"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    safe_stem = "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_"
        for ch in source_path.stem
    )[:80] or "vision_input"

    heatmap_path = OUTPUT_DIR / f"{safe_stem}_heatmap.png"
    overlay_path = OUTPUT_DIR / f"{safe_stem}_overlay.png"

    heatmap.save(heatmap_path)
    overlay.save(overlay_path)

    result = {
        "model": "V6 ResNet18 PatchCore",
        "status": prediction,
        "prediction": prediction,
        "anomaly_score": float(score),
        "threshold": float(threshold),
        "threshold_percentile": percentile,
        "score_method": "top_5_percent",
        "heatmap_path": str(heatmap_path),
        "overlay_path": str(overlay_path),
        "source_image": str(source_path),

        # Compatibility keys used by the integrated dashboard.
        "ensemble_score": float(score),
        "visualization_path": str(heatmap_path),
    }

    return result
