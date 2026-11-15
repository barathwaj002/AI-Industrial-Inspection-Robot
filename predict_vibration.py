from pathlib import Path
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import skew, kurtosis


# =========================================================
# CONFIGURATION
# =========================================================

# Deployment-relative paths: model files stay inside the GitHub project folder.
ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    ROOT
    / "bearing_results"
    / "mfpt_bearing_random_forest.joblib"
)

OUTPUT_DIR = ROOT / "vibration_inference_results"

WINDOW_SIZE = 4096

FEATURE_NAMES = [
    "mean",
    "std",
    "rms",
    "abs_mean",
    "peak_to_peak",
    "crest_factor",
    "shape_factor",
    "skewness",
    "kurtosis",
    "zero_crossing_rate",
]


# =========================================================
# LOAD SAVED MODEL PACKAGE
# =========================================================

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    package = joblib.load(MODEL_PATH)

    if not isinstance(package, dict) or "model" not in package:
        raise ValueError(
            "Unexpected model package structure."
        )

    model = package["model"]

    if model.n_features_in_ != len(FEATURE_NAMES):
        raise ValueError(
            f"Expected {model.n_features_in_} features, "
            f"but this script defines {len(FEATURE_NAMES)}."
        )

    saved_features = package.get("feature_names")

    if saved_features is not None:
        if list(saved_features) != FEATURE_NAMES:
            raise ValueError(
                "Feature names/order do not match the "
                "saved training package."
            )

    if int(package.get("window_size", WINDOW_SIZE)) != WINDOW_SIZE:
        raise ValueError(
            "Window size does not match the training package."
        )

    return package


# =========================================================
# FEATURE EXTRACTION
# MATCHES train_bearing_model.py
# =========================================================

def extract_features(window):
    x = np.asarray(window, dtype=np.float64).squeeze()

    if x.ndim != 1 or len(x) != WINDOW_SIZE:
        raise ValueError(
            f"Expected a 1-D window of {WINDOW_SIZE} samples."
        )

    if not np.all(np.isfinite(x)):
        raise ValueError(
            "Window contains non-finite values."
        )

    # IMPORTANT: exactly as in the original training script.
    # Remove DC offset before calculating all features.
    x = x - np.mean(x)

    abs_x = np.abs(x)

    rms = float(np.sqrt(np.mean(x ** 2)))
    abs_mean = float(np.mean(abs_x))

    peak = float(np.max(abs_x))
    peak_to_peak = float(np.ptp(x))

    crest_factor = (
        peak / rms if rms > 1e-12 else 0.0
    )

    shape_factor = (
        rms / abs_mean if abs_mean > 1e-12 else 0.0
    )

    # Match the original training script exactly.
    skewness_value = float(
        skew(x, bias=False)
    )

    # Pearson kurtosis, NOT Fisher/excess kurtosis.
    kurtosis_value = float(
        kurtosis(x, fisher=False, bias=False)
    )

    zero_crossing_rate = float(
        np.mean(np.diff(np.signbit(x)))
    )

    return [
        float(np.mean(x)),
        float(np.std(x)),
        rms,
        abs_mean,
        peak_to_peak,
        crest_factor,
        shape_factor,
        skewness_value,
        kurtosis_value,
        zero_crossing_rate,
    ]


# =========================================================
# LOAD MFPT SIGNAL
# =========================================================

def load_mat_signal(mat_path):
    mat_path = Path(mat_path)

    if not mat_path.exists():
        raise FileNotFoundError(
            f"MAT file not found: {mat_path}"
        )

    data = loadmat(
        mat_path,
        squeeze_me=True,
        struct_as_record=False
    )

    if "bearing" not in data:
        raise ValueError(
            "MAT file does not contain the expected "
            "'bearing' structure."
        )

    bearing = data["bearing"]

    # MFPT training script uses bearing["gs"].
    if hasattr(bearing, "gs"):
        signal = np.asarray(
            bearing.gs,
            dtype=np.float64
        ).squeeze()

    elif isinstance(bearing, np.void) and bearing.dtype.names:
        signal = np.asarray(
            bearing["gs"],
            dtype=np.float64
        ).squeeze()

    else:
        raise ValueError(
            "Could not read the 'gs' signal from the "
            "bearing structure."
        )

    if signal.ndim != 1:
        raise ValueError(
            f"Expected a 1-D vibration signal; "
            f"got shape {signal.shape}."
        )

    # The original training script removes non-finite samples.
    if not np.all(np.isfinite(signal)):
        signal = signal[np.isfinite(signal)]

    if len(signal) < WINDOW_SIZE:
        raise ValueError(
            f"Signal has only {len(signal)} finite samples; "
            f"at least {WINDOW_SIZE} are required."
        )

    sampling_rate = None

    if hasattr(bearing, "sr"):
        try:
            sampling_rate = float(bearing.sr)
        except (TypeError, ValueError):
            pass

    return signal, sampling_rate


# =========================================================
# PREDICT ONE WINDOW
# =========================================================

def predict_window(package, window):
    model = package["model"]

    features = extract_features(window)

    X = pd.DataFrame(
        [features],
        columns=FEATURE_NAMES
    )

    predicted_class = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]

    probability_map = {
        str(label): float(probability)
        for label, probability in zip(
            model.classes_,
            probabilities
        )
    }

    return {
        "prediction": str(predicted_class),
        "probabilities": probability_map,
        "features": {
            name: float(value)
            for name, value in zip(
                FEATURE_NAMES,
                features
            )
        },
    }


# =========================================================
# PROCESS THE RECORDING
# =========================================================

def predict_recording(package, signal, source_name):
    total_windows = len(signal) // WINDOW_SIZE

    rows = []

    print(f"\nRecording: {source_name}")
    print(f"Signal samples: {len(signal):,}")
    print(f"Window size: {WINDOW_SIZE}")
    print(f"Complete windows: {total_windows}")
    print("Window overlap: None")

    for index in range(total_windows):
        start = index * WINDOW_SIZE
        end = start + WINDOW_SIZE

        window = signal[start:end]

        result = predict_window(
            package,
            window
        )

        row = {
            "source": str(source_name),
            "window": index,
            "start_sample": start,
            "end_sample": end,
            "prediction": result["prediction"],
        }

        for label, probability in result[
            "probabilities"
        ].items():
            row[f"probability_{label}"] = probability

        for name, value in result["features"].items():
            row[f"feature_{name}"] = value

        rows.append(row)

        if (index + 1) % 100 == 0 or index + 1 == total_windows:
            print(
                f"Processed {index + 1}/{total_windows} windows"
            )

    return pd.DataFrame(rows)


# =========================================================
# SAVE RESULTS
# =========================================================

def save_results(results, source_name, sampling_rate):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    stem = Path(source_name).stem

    csv_path = OUTPUT_DIR / f"{stem}_predictions.csv"
    json_path = OUTPUT_DIR / f"{stem}_summary.json"

    results.to_csv(
        csv_path,
        index=False
    )

    counts = results["prediction"].value_counts()

    summary = {
        "source": str(source_name),
        "sampling_rate_hz": sampling_rate,
        "window_size": WINDOW_SIZE,
        "total_windows": int(len(results)),
        "predicted_window_counts": {
            str(label): int(count)
            for label, count in counts.items()
        },
        "predicted_window_percentages": {
            str(label): round(
                float(count / len(results) * 100),
                2
            )
            for label, count in counts.items()
        },
        "csv_path": str(csv_path),
    }

    json_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    print("\n========== VIBRATION SUMMARY ==========")
    print("Recording:", source_name)
    print("Sampling rate:", sampling_rate)
    print("Total windows:", len(results))

    print("\nPredicted window counts:")
    print(counts.to_string())

    print("\nPredicted percentages:")
    for label, percentage in summary[
        "predicted_window_percentages"
    ].items():
        print(f"{label}: {percentage:.2f}%")

    print("\nPredictions CSV:", csv_path)
    print("Summary JSON:", json_path)

    return summary


# =========================================================
# MAIN
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Predict bearing conditions using the saved "
            "MFPT Random Forest model."
        )
    )

    parser.add_argument(
        "--mat",
        required=True,
        help="Path to an MFPT vibration .mat file"
    )

    args = parser.parse_args()

    print("Loading saved bearing model...")
    package = load_model()

    print("Model loaded successfully.")
    print(
        "Classes:",
        list(package["model"].classes_)
    )

    signal, sampling_rate = load_mat_signal(args.mat)

    results = predict_recording(
        package,
        signal,
        args.mat
    )

    save_results(
        results,
        args.mat,
        sampling_rate
    )


if __name__ == "__main__":
    main()