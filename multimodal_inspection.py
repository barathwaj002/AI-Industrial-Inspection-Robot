
from pathlib import Path
from datetime import datetime
import argparse
import json
import re

import joblib
import numpy as np
import pandas as pd

from scipy.io import wavfile
from scipy.signal import stft
from scipy.fft import dct

import predict_vision
import predict_vibration


# =========================================================
# CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parent

OUTPUT_DIR = ROOT / "multimodal_results"

ACOUSTIC_MODEL_PATH = (
    ROOT
    / "acoustic_supervised_results"
    / "sectionwise_acoustic_models.joblib"
)


# =========================================================
# SHARED UTILITIES
# =========================================================

def json_safe(value):
    """Convert common NumPy values to JSON-compatible values."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]

    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if np.isfinite(value) else None

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if isinstance(value, Path):
        return str(value)

    return value


# =========================================================
# ACOUSTIC FEATURE EXTRACTION
# =========================================================

def parse_section(path):
    """Infer one of the trained model sections from section_00/01/02 in the WAV name."""
    text = str(path).replace("\\", "/").lower()
    match = re.search(r"section[_-]?(\d+)", text)
    if match:
        section = int(match.group(1))
        if section in (0, 1, 2):
            return section
    raise ValueError(
        "Cannot infer the trained MIMII section from this WAV filename. "
        "Rename the file to include section_00, section_01 or section_02 "
        "according to the section used when preparing the model dataset."
    )


def read_wav(path):
    sampling_rate, raw = wavfile.read(str(path))
    original_dtype = raw.dtype

    if raw.ndim == 2:
        signal = raw.astype(np.float32).mean(axis=1)
    else:
        signal = raw.astype(np.float32)

    if np.issubdtype(original_dtype, np.integer):
        info = np.iinfo(original_dtype)
        scale = max(abs(info.min), abs(info.max))
        signal /= float(scale)

    # Match the training pipeline's maximum recording duration.
    if len(signal) > sampling_rate * 10:
        signal = signal[:sampling_rate * 10]

    if len(signal) < 1024:
        signal = np.pad(signal, (0, 1024 - len(signal)))

    signal = np.nan_to_num(
        signal, nan=0.0, posinf=0.0, neginf=0.0
    )

    return int(sampling_rate), signal.astype(np.float32)


def mel_filterbank(sampling_rate, n_fft, n_mels=40):

    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_points = np.linspace(
        hz_to_mel(0),
        hz_to_mel(sampling_rate / 2),
        n_mels + 2
    )

    hz_points = mel_to_hz(mel_points)

    bins = np.floor(
        (n_fft + 1) * hz_points / sampling_rate
    ).astype(int)

    bins = np.clip(bins, 0, n_fft // 2)

    filters = np.zeros(
        (n_mels, n_fft // 2 + 1),
        dtype=np.float32
    )

    for m in range(1, n_mels + 1):
        left = bins[m - 1]
        center = bins[m]
        right = bins[m + 1]

        if center > left:
            filters[m - 1, left:center] = (
                np.arange(left, center) - left
            ) / (center - left)

        if right > center:
            filters[m - 1, center:right] = (
                right - np.arange(center, right)
            ) / (right - center)

    return filters


def acoustic_features(path):
    sampling_rate, signal = read_wav(path)

    # Match the training pipeline's downsampling.
    if sampling_rate > 32000 and sampling_rate % 2 == 0:
        signal = signal[::2]
        sampling_rate //= 2

    signal = signal - np.mean(signal)

    peak = np.max(np.abs(signal)) + 1e-10
    signal = signal / peak

    nperseg = min(
        2048,
        max(
            256,
            2 ** int(
                np.floor(
                    np.log2(min(len(signal), 2048))
                )
            )
        )
    )

    noverlap = nperseg // 2

    frequencies, _, z = stft(
        signal,
        fs=sampling_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        boundary=None
    )

    power = (np.abs(z) ** 2).astype(np.float32)

    filters = mel_filterbank(
        sampling_rate, nperseg, n_mels=40
    )

    mel = filters @ power
    logmel = np.log(mel + 1e-10)

    mfcc = dct(
        logmel, type=2, axis=0, norm="ortho"
    )[:10]

    centroid = (
        np.sum(frequencies[:, None] * power, axis=0)
        / (np.sum(power, axis=0) + 1e-10)
    )

    bandwidth = np.sqrt(
        np.sum(
            (
                frequencies[:, None]
                - centroid[None, :]
            ) ** 2 * power,
            axis=0
        )
        / (np.sum(power, axis=0) + 1e-10)
    )

    cumulative = np.cumsum(power, axis=0)
    totals = cumulative[-1] + 1e-10

    roll_idx = np.argmax(
        cumulative >= 0.85 * totals[None, :],
        axis=0
    )

    rolloff = frequencies[roll_idx]

    flatness = (
        np.exp(np.mean(np.log(power + 1e-10), axis=0))
        / (np.mean(power, axis=0) + 1e-10)
    )

    rms = np.sqrt(np.mean(signal * signal) + 1e-10)
    abs_signal = np.abs(signal)

    stats = np.array([
        np.mean(signal),
        np.std(signal),
        rms,
        np.mean(abs_signal),
        np.max(abs_signal),
        np.percentile(abs_signal, 95),
        np.percentile(abs_signal, 99),
        np.mean(np.diff(signal) ** 2)
        if len(signal) > 1 else 0.0,
        np.mean(centroid),
        np.std(centroid),
        np.mean(bandwidth),
        np.std(bandwidth),
        np.mean(rolloff),
        np.std(rolloff),
        np.mean(flatness),
        np.std(flatness),
        np.mean(power),
        np.std(power)
    ], dtype=np.float32)

    base_features = np.concatenate([
        logmel.mean(axis=1),
        logmel.std(axis=1),
        mfcc.mean(axis=1),
        mfcc.std(axis=1),
        stats
    ])

    section = parse_section(path)

    section_hot = np.array([
        float(section == s) for s in (0, 1, 2)
    ], dtype=np.float32)

    return np.nan_to_num(
        np.concatenate([base_features, section_hot]),
        nan=0.0, posinf=0.0, neginf=0.0
    )


# =========================================================
# ACOUSTIC PREDICTION
# =========================================================

def predict_acoustic(path, package):
    """Run the section-specific estimator from the new model bundle.

    CatBoost and ExtraTrees return anomaly probabilities. The section-2
    RBF-SVM was trained without probability=True, so its decision margin
    is reported as a decision score (not misrepresented as a probability).
    """
    section = parse_section(path)
    key = str(section)
    if key not in package:
        raise KeyError(f"No acoustic estimator found for section {section}.")

    features = acoustic_features(path)
    if len(features) != 121:
        raise ValueError(f"Expected 121 acoustic features; generated {len(features)}.")

    model_entry = package[key]
    estimator = model_entry["estimator"]
    X = np.asarray(features, dtype=np.float32).reshape(1, -1)
    classes = list(getattr(estimator, "classes_", [0, 1]))
    if 1 not in classes:
        raise ValueError("Acoustic estimator does not contain anomaly class label 1.")

    if hasattr(estimator, "predict_proba"):
        scores = estimator.predict_proba(X)[0]
        anomaly_score = float(scores[classes.index(1)])
        score_type = "anomaly_probability"
        threshold = 0.5
        predicted = int(anomaly_score >= threshold)
    elif hasattr(estimator, "decision_function"):
        raw_score = np.asarray(estimator.decision_function(X)).reshape(-1)
        anomaly_score = float(raw_score[0])
        score_type = "decision_function_margin"
        threshold = 0.0
        predicted = int(anomaly_score >= threshold)
    else:
        predicted = int(np.asarray(estimator.predict(X)).reshape(-1)[0])
        anomaly_score = None
        score_type = "class_prediction_only"
        threshold = None

    return {
        "status": "ANOMALY" if predicted == 1 else "NORMAL",
        "predicted_label": predicted,
        "model_name": model_entry.get("model", type(estimator).__name__),
        "section": section,
        "score": anomaly_score,
        "score_type": score_type,
        "anomaly_probability": anomaly_score if score_type == "anomaly_probability" else None,
        "threshold": threshold,
        "feature_count": int(len(features)),
        "evidence_quality": "LIMITED_DOMAIN_VALIDATION",
        "warning": (
            "For section 02, the RBF-SVM decision margin is not a probability. "
            "Model outputs require validation on recordings from the target machine."
        )
    }



# =========================================================
# VIBRATION PREDICTION
# =========================================================

def predict_vibration_file(path):
    package = predict_vibration.load_model()

    signal, sampling_rate = (
        predict_vibration.load_mat_signal(path)
    )

    results = predict_vibration.predict_recording(
        package, signal, Path(path).name
    )

    if results.empty or "prediction" not in results.columns:
        raise ValueError(
            "No valid vibration window predictions were produced."
        )

    predictions = results["prediction"].astype(str)
    total = len(predictions)

    counts = predictions.value_counts().to_dict()

    percentages = {
        str(label): round(100.0 * count / total, 2)
        for label, count in counts.items()
    }

    dominant_class = str(predictions.mode().iloc[0])

    probability_columns = [
        col for col in results.columns
        if col.startswith("probability_")
    ]

    mean_probabilities = {
        col.replace("probability_", ""):
        round(float(results[col].mean()), 4)
        for col in probability_columns
    }

    normal_labels = {"normal"}

    fault_mask = ~predictions.str.strip().str.lower().isin(
        normal_labels
    )

    fault_windows = int(fault_mask.sum())

    return {
        "source_file": str(path),
        "sampling_rate_hz": float(sampling_rate),
        "total_windows": int(total),
        "predicted_class": dominant_class,
        "dominant_class_percentage": percentages[dominant_class],
        "class_counts": {
            str(label): int(count)
            for label, count in counts.items()
        },
        "class_percentages": percentages,
        "mean_probabilities": mean_probabilities,
        "fault_windows": fault_windows,
        "fault_window_percentage": round(
            100.0 * fault_windows / total, 2
        ),
        "status": (
            "FAULT_INDICATED"
            if fault_windows > 0
            else "NORMAL"
        ),
        "warning": (
            "A non-Normal window is a model indication, "
            "not a confirmed physical fault."
        )
    }


# =========================================================
# EVIDENCE INTERPRETATION
# =========================================================

def interpret_vision(vision):
    if vision is None:
        return {
            "available": False,
            "status": "UNAVAILABLE"
        }

    prediction = str(
        vision.get("prediction", "UNKNOWN")
    ).upper()

    flagged = prediction in {
        "ANOMALOUS", "DEFECTIVE", "FAULT"
    }

    return {
        "available": True,
        "status": prediction,
        "anomaly_score": vision.get("ensemble_score"),
        "decision_threshold": vision.get("threshold"),
        "flagged": flagged,
        "evidence_quality": "PRELIMINARY_UNVERIFIED",
        "warning": (
            "Vision feature extraction and score aggregation "
            "still require verification against the original "
            "training pipeline."
        )
    }


def interpret_acoustic(acoustic):
    if acoustic is None:
        return {
            "available": False,
            "status": "UNAVAILABLE"
        }

    return {
        "available": True,
        "status": acoustic["status"],
        "anomaly_probability": acoustic.get("anomaly_probability"),
        "score": acoustic.get("score"),
        "score_type": acoustic.get("score_type"),
        "threshold": acoustic.get("threshold"),
        "flagged": acoustic["status"] == "ANOMALY",
        "evidence_quality": "LIMITED_DOMAIN_VALIDATION",
        "warning": acoustic["warning"]
    }


def interpret_vibration(vibration):
    if vibration is None:
        return {
            "available": False,
            "status": "UNAVAILABLE"
        }

    return {
        "available": True,
        "status": vibration["status"],
        "predicted_class": vibration["predicted_class"],
        "dominant_class_percentage":
            vibration["dominant_class_percentage"],
        "class_counts": vibration["class_counts"],
        "class_percentages": vibration["class_percentages"],
        "mean_probabilities": vibration["mean_probabilities"],
        "fault_windows": vibration["fault_windows"],
        "total_windows": vibration["total_windows"],
        "flagged": vibration["status"] == "FAULT_INDICATED",
        "evidence_quality": "RECORDING_LEVEL_MODEL_OUTPUT",
        "warning": vibration["warning"]
    }


# =========================================================
# FAULT DIAGNOSIS, SEVERITY AND RECOMMENDATION
# =========================================================

def build_assessment(vision, acoustic, vibration, errors):
    evidence = {
        "vision": interpret_vision(vision),
        "acoustic": interpret_acoustic(acoustic),
        "vibration": interpret_vibration(vibration)
    }

    available = [
        name for name, item in evidence.items()
        if item["available"]
    ]

    flagged = [
        name for name, item in evidence.items()
        if item.get("flagged", False)
    ]

    missing = [
        name for name, item in evidence.items()
        if not item["available"]
    ]

    # Diagnosis should be based on actual model outputs.
    if vibration is not None and vibration["status"] == "FAULT_INDICATED":
        diagnosis = vibration["predicted_class"]
        diagnosis_source = "vibration_model"

    elif vision is not None and evidence["vision"].get("flagged"):
        diagnosis = "Visual anomaly suspected"
        diagnosis_source = "vision_model"

    elif acoustic is not None and acoustic["status"] == "ANOMALY":
        diagnosis = "Acoustic anomaly suspected"
        diagnosis_source = "acoustic_model"

    else:
        diagnosis = "No specific fault identified"
        diagnosis_source = None

    # Explicit rule-based decision. No raw scores are averaged.
    if missing:
        decision = "NEEDS_REVIEW"
        reason = (
            "One or more modalities are unavailable. "
            "A complete inspection decision cannot be made."
        )

    elif "vibration" in flagged or "vision" in flagged:
        decision = "FAULT_SUSPECTED"
        reason = (
            "The vibration or vision model flagged a potential issue. "
            "Confirm the result with a targeted inspection."
        )

    elif "acoustic" in flagged:
        decision = "NEEDS_REVIEW"
        reason = (
            "Only the acoustic model flagged an anomaly. "
            "Review the recording because cross-domain acoustic "
            "performance is limited."
        )

    else:
        decision = "NO_ANOMALY_FLAGGED"
        reason = (
            "None of the current model decision rules flagged "
            "a concern. This is not proof that the equipment "
            "is defect-free."
        )

    # Provisional severity is a transparent policy, not a
    # validated engineering risk score.
    if missing:
        severity = "UNDETERMINED"
        severity_reason = (
            "Incomplete sensor evidence prevents a full assessment."
        )

    elif "vibration" in flagged and len(flagged) >= 2:
        severity = "HIGH"
        severity_reason = (
            "Multiple model alerts, including vibration. "
            "High is a provisional triage priority, not a measured "
            "failure probability or safety rating."
        )

    elif "vibration" in flagged or "vision" in flagged:
        severity = "MODERATE"
        severity_reason = (
            "At least one vision or vibration model raised a concern. "
            "Severity is provisional pending physical verification."
        )

    elif "acoustic" in flagged:
        severity = "LOW"
        severity_reason = (
            "Acoustic-only alert with limited domain validation. "
            "Review the recording before escalating."
        )

    else:
        severity = "NO_ALERT"
        severity_reason = (
            "No model alert was raised; equipment health is not certified."
        )

    # Practical next actions based on available evidence.
    if missing:
        recommendations = [
            "Repeat the missing sensor measurement.",
            "Check input file paths and model execution logs.",
            "Do not treat the incomplete result as a clean inspection."
        ]

    elif "vibration" in flagged:
        recommendations = [
            "Inspect the bearing and its mounting for the predicted fault.",
            "Repeat the vibration measurement to confirm repeatability.",
            "Compare against a known-good recording under matching conditions."
        ]

        if vibration["predicted_class"] == "Outer_Race_Fault":
            recommendations.insert(
                0,
                "Prioritize physical inspection of the bearing outer race."
            )

        elif vibration["predicted_class"] == "Inner_Race_Fault":
            recommendations.insert(
                0,
                "Prioritize physical inspection of the bearing inner race."
            )

        if "acoustic" in flagged:
            recommendations.append(
                "Review the acoustic alert as supplementary evidence; "
                "its transfer performance is limited."
            )

        if "vision" in flagged:
            recommendations.append(
                "Inspect the visual anomaly and verify that the image "
                "belongs to the same machine and inspection event."
            )

    elif "vision" in flagged:
        recommendations = [
            "Review the vision heatmap and inspect the suspected region.",
            "Capture a repeat image with consistent lighting and viewpoint.",
            "Verify the image against a labelled inspection example."
        ]

    elif "acoustic" in flagged:
        recommendations = [
            "Record the sound again under comparable operating conditions.",
            "Check for background noise and microphone placement changes.",
            "Compare against labelled recordings from the same machine type."
        ]

    else:
        recommendations = [
            "Continue the planned inspection and maintenance schedule.",
            "Repeat measurements under comparable operating conditions.",
            "Do not interpret the absence of alerts as proof of zero risk."
        ]

    warnings = [
        "This is a prototype decision-support report, not a safety-certified diagnosis.",
        "The image, WAV and MAT datasets may represent different machines and events.",
        "Severity is a rule-based triage label, not a calibrated failure-risk probability.",
        "Model outputs require independent validation before operational deployment."
    ]

    if vision is not None:
        warnings.append(
            "The vision model is marked preliminary until its reconstructed "
            "feature extraction and score aggregation are verified."
        )

    if acoustic is not None:
        warnings.append(
            "The acoustic model has shown limited transfer performance "
            "across recording domains."
        )

    if errors:
        warnings.append(
            "One or more models produced errors; see the errors section."
        )

    return {
        "decision": decision,
        "reason": reason,
        "suspected_fault": diagnosis,
        "diagnosis_source": diagnosis_source,
        "provisional_severity": severity,
        "severity_explanation": severity_reason,
        "modalities_available": available,
        "modalities_missing": missing,
        "modalities_flagging_concern": flagged,
        "sensor_evidence": evidence,
        "recommended_actions": recommendations,
        "warnings": warnings
    }


# =========================================================
# MAIN PIPELINE
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="AI multimodal industrial inspection report generator"
    )

    parser.add_argument(
        "--image", required=True,
        help="Path to the inspection image"
    )

    parser.add_argument(
        "--wav", required=True,
        help="Path to the MIMII acoustic WAV file"
    )

    parser.add_argument(
        "--mat", required=True,
        help="Path to the MFPT vibration MAT file"
    )

    args = parser.parse_args()

    inputs = {
        "image": Path(args.image),
        "wav": Path(args.wav),
        "mat": Path(args.mat)
    }

    for name, path in inputs.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"{name.upper()} file not found: {path}"
            )

    if not ACOUSTIC_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Acoustic model not found: {ACOUSTIC_MODEL_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "system": "AI Multimodal Industrial Inspection System",
        "report_version": "2.0",
        "generated_at": datetime.now().astimezone().isoformat(),
        "inspection_id": datetime.now().strftime(
            "INS-%Y%m%d-%H%M%S"
        ),
        "inputs": {
            name: str(path.resolve())
            for name, path in inputs.items()
        },
        "vision": None,
        "acoustic": None,
        "vibration": None,
        "errors": {}
    }

    # 1. Vision
    print("\n[1/3] Running vision model...")

    try:
        vision_result = predict_vision.predict(
            str(inputs["image"])
        )

        if isinstance(vision_result, dict):
            report["vision"] = vision_result

        else:
            vision_json = (
                ROOT
                / "vision_inference_results"
                / f"{inputs['image'].stem}_prediction.json"
            )

            if vision_json.is_file():
                report["vision"] = json.loads(
                    vision_json.read_text(encoding="utf-8")
                )
            else:
                raise RuntimeError(
                    "Vision predictor did not return a result "
                    "or create its expected JSON output."
                )

    except Exception as exc:
        report["errors"]["vision"] = str(exc)
        print("Vision failed:", exc)

    # 2. Acoustic
    print("\n[2/3] Running acoustic model...")

    try:
        acoustic_package = joblib.load(ACOUSTIC_MODEL_PATH)

        report["acoustic"] = predict_acoustic(
            inputs["wav"], acoustic_package
        )

        print(
            "Acoustic prediction:",
            report["acoustic"]["status"]
        )

    except Exception as exc:
        report["errors"]["acoustic"] = str(exc)
        print("Acoustic failed:", exc)

    # 3. Vibration
    print("\n[3/3] Running vibration model...")

    try:
        report["vibration"] = predict_vibration_file(
            inputs["mat"]
        )

        print(
            "Dominant vibration class:",
            report["vibration"]["predicted_class"]
        )

        print(
            "Class percentages:",
            json.dumps(
                report["vibration"]["class_percentages"],
                indent=2
            )
        )

    except Exception as exc:
        report["errors"]["vibration"] = str(exc)
        print("Vibration failed:", exc)

    # 4. Assessment
    report["assessment"] = build_assessment(
        report["vision"],
        report["acoustic"],
        report["vibration"],
        report["errors"]
    )

    report["errors_summary"] = {
        "failed_modalities": list(report["errors"].keys()),
        "failure_count": len(report["errors"])
    }

    # 5. Save JSON
    output_path = OUTPUT_DIR / "latest_inspection.json"

    report = json_safe(report)

    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    # 6. Human-readable console report
    assessment = report["assessment"]

    print("\n" + "=" * 64)
    print("      AI INDUSTRIAL INSPECTION REPORT")
    print("=" * 64)

    print("Inspection ID :", report["inspection_id"])
    print("Generated at  :", report["generated_at"])
    print("Decision      :", assessment["decision"])
    print("Suspected fault:", assessment["suspected_fault"])
    print("Severity      :", assessment["provisional_severity"])
    print("Reason        :", assessment["reason"])

    print("\n--- SENSOR EVIDENCE ---")

    for name, item in assessment["sensor_evidence"].items():
        print(f"\n{name.upper()}")

        for key in (
            "status",
            "anomaly_score",
            "anomaly_probability",
            "decision_threshold",
            "predicted_class",
            "dominant_class_percentage",
            "fault_windows",
            "total_windows"
        ):
            if key in item:
                print(f"  {key}: {item[key]}")

    print("\n--- RECOMMENDED ACTIONS ---")

    for index, action in enumerate(
        assessment["recommended_actions"], start=1
    ):
        print(f"{index}. {action}")

    print("\n--- WARNINGS ---")

    for warning in assessment["warnings"]:
        print(f"- {warning}")

    if report["errors"]:
        print("\n--- MODEL ERRORS ---")
        print(json.dumps(report["errors"], indent=2))

    print("\nFull JSON report:")
    print(output_path)


if __name__ == "__main__":
    main()
