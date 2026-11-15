
from pathlib import Path
from datetime import datetime
from io import BytesIO
import json
import tempfile
import re
import textwrap
import uuid

import joblib
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

from PIL import Image, ImageOps

import predict_vision
import multimodal_inspection as inspection


# =========================================================
# PROJECT PATHS
# =========================================================

ROOT = Path(__file__).resolve().parent

OUTPUT_DIR = ROOT / "multimodal_results"

ACOUSTIC_MODEL_PATH = (
    ROOT / "acoustic_supervised_results"
    / "sectionwise_acoustic_models.joblib"
)

VISION_OUTPUT_DIR = ROOT / "vision_inference_results"


st.set_page_config(
    page_title="AI Industrial Inspection | Smart Robot",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# STREAMLIT HTML RENDERING FIX
# Streamlit's Markdown parser can treat indented HTML after
# blank lines as code blocks. Normalize every HTML-enabled
# markdown block before passing it to Streamlit.
# =========================================================

_original_st_markdown = st.markdown

def _render_markdown_without_html_code_blocks(body, *args, **kwargs):
    if isinstance(body, str) and kwargs.get("unsafe_allow_html", False):
        # Remove common indentation first.
        body = textwrap.dedent(body)
        # Keep each HTML fragment as one continuous raw HTML block.
        # Blank lines inside <div>, <style>, etc. can terminate the
        # Markdown HTML block and make the remaining tags look like code.
        body = re.sub(r"\n[ \t]*\n+", "\n", body)
        body = body.strip()
    return _original_st_markdown(body, *args, **kwargs)

st.markdown = _render_markdown_without_html_code_blocks


# =========================================================
# LIGHT THEME + ROUNDED NAVIGATION
# =========================================================

st.markdown(
    """
    <style>

    :root {
      --navy:#173b5e;
      --blue:#2878b8;
      --cyan:#38a9c8;
      --ink:#203449;
      --muted:#60758a;
      --line:#d8e5ef;
    }

    .stApp {
      background:
        radial-gradient(
          circle at 8% 0%,
          rgba(221,242,252,.9),
          transparent 32%
        ),
        radial-gradient(
          circle at 92% 5%,
          rgba(229,237,255,.8),
          transparent 30%
        ),
        linear-gradient(
          135deg,
          #f7fbff 0%,
          #eef5fb 52%,
          #f8fbff 100%
        );

      color:var(--ink);
    }

    [data-testid="stHeader"] {
      background:rgba(247,251,255,.92);
    }

    [data-testid="stSidebar"] {
      background:linear-gradient(
        180deg,
        #ffffff 0%,
        #edf5fb 100%
      );

      border-right:1px solid #dce8f1;
    }

    [data-testid="stSidebar"] * {
      color:#263f56;
    }

    .block-container {
      padding-top:3.2rem;
      padding-bottom:2.8rem;
      max-width:1500px;
    }

    /* Main title header */

    .hero {
      padding:3.1rem 2.8rem 2.8rem;

      border:1px solid #d5e7f2;
      border-radius:26px;

      margin-top:1.25rem;
      margin-bottom:2.4rem;

      background:linear-gradient(
        115deg,
        #ffffff 0%,
        #edf7fc 55%,
        #f2f5ff 100%
      );

      box-shadow:
        0 12px 34px rgba(37,82,119,.09);
    }

    .hero-kicker {
      color:#287da4;
      letter-spacing:.18em;
      font-size:.76rem;
      font-weight:850;
      margin-bottom:1.55rem;
    }

    .hero-title {
      color:#173b5e;
      font-size:2.5rem;
      line-height:1.4;
      font-weight:850;
      margin:0 0 1.45rem;
    }

    .hero-sub {
      color:#526b80;
      font-size:1.02rem;
      line-height:1.85;
      margin-top:.7rem;
      max-width:900px;
    }

    @media (max-width: 768px) {

      .hero {
        padding:2rem 1.35rem;
      }

      .hero-title {
        font-size:1.8rem;
      }

      .hero-sub {
        font-size:.94rem;
      }
    }

    .section-heading {
      color:#173b5e;
      font-size:1.25rem;
      font-weight:850;
      margin:1.5rem 0 .8rem;
    }

    .muted {
      color:#60758a;
      font-size:.9rem;
    }

    /* Metric cards */

    div[data-testid="stMetric"] {
      background:#ffffff;
      border:1px solid #dce8f1;
      border-radius:16px;
      padding:14px 16px;

      box-shadow:
        0 5px 16px rgba(37,82,119,.06);
    }

    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
      color:#203f5b;
    }

    /* Upload boxes */

    div[data-testid="stFileUploader"] section {
      background:#ffffff;
      border:1px dashed #9fc6dc;
      border-radius:17px;
      transition:all .22s ease;
    }

    div[data-testid="stFileUploader"] section:hover {
      border-color:#3298bd;

      box-shadow:
        0 0 0 3px rgba(56,169,200,.10);

      transform:translateY(-2px);
    }

    /* Buttons */

    div.stButton > button,
    div.stDownloadButton > button {
      border:1px solid #2878b8;
      border-radius:13px;

      background:linear-gradient(
        100deg,
        #2878b8,
        #3b91c4
      );

      color:white;
      font-weight:800;
      transition:all .2s ease;
    }

    div.stButton > button:hover,
    div.stDownloadButton > button:hover {
      border-color:#1b669f;
      color:white;

      transform:translateY(-2px);

      box-shadow:
        0 7px 18px rgba(40,120,184,.20);
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
      border-color:#dce8f1;
    }

    /* Information notice */

    .notice {
      padding:1.1rem 1.25rem;
      border-radius:15px;

      border-left:4px solid #38a9c8;

      background:#eaf5fb;
      color:#29465f;

      margin:.65rem 0 1.15rem;
      line-height:1.65;
    }

    /* Precaution cards */

    .precaution {
      padding:.95rem 1.05rem;
      border-radius:14px;
      margin:.6rem 0;

      background:linear-gradient(
        100deg,
        #f1f8fc,
        #ffffff
      );

      border:1px solid #d7e7f0;
      color:#29465f;

      box-shadow:
        0 3px 10px rgba(37,82,119,.04);
    }

    .status-pill {
      display:inline-block;
      padding:.3rem .7rem;

      border-radius:99px;
      font-size:.76rem;
      font-weight:800;

      background:#e6f6ef;
      color:#18734e;

      border:1px solid #bce6d3;
    }

    /* Rounded sidebar navigation boxes */

    [data-testid="stRadio"] [role="radiogroup"] {
      display:flex;
      flex-direction:column;
      gap:.75rem;
      width:100%;
    }

    [data-testid="stRadio"] [role="radiogroup"] label {
      background:#ffffff;

      border:1px solid #d5e4ee;
      border-radius:16px;

      padding:.9rem 1rem;
      margin:0;
      width:100%;

      box-shadow:
        0 3px 9px rgba(37,82,119,.045);

      transition:all .18s ease;
    }

    [data-testid="stRadio"] [role="radiogroup"] label:hover {
      border-color:#65a9ca;
      background:#f3faff;
    }

    [data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) {
      background:#e6f3fb;
      border:1.5px solid #4c9bc2;

      box-shadow:
        0 4px 12px rgba(40,120,184,.10);
    }

    /* Hide circular radio indicator */

    [data-testid="stRadio"] [role="radiogroup"]
    label > div:first-child {
      display:none;
    }

    [data-testid="stRadio"] [role="radiogroup"] label p {
      color:#24445f;
      font-weight:750;
      margin:0;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# CACHED ACOUSTIC MODEL PACKAGE
# =========================================================

@st.cache_resource(
    show_spinner="Loading section-wise acoustic models..."
)
def load_acoustic_package(
    model_path: str,
    modified_time: float
):

    path = Path(model_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Acoustic model bundle not found: {path}"
        )

    package = joblib.load(path)

    if not isinstance(package, dict):
        raise ValueError(
            "Acoustic model bundle must be a dictionary."
        )

    required = {"0", "1", "2"}

    if not required.issubset(
        {str(key) for key in package.keys()}
    ):
        raise ValueError(
            "Acoustic bundle must contain section keys 0, 1 and 2."
        )

    return package


# =========================================================
# JSON CONVERSION
# =========================================================

def safe_json(value):

    def clean(item):

        if isinstance(item, dict):
            return {
                str(key): clean(val)
                for key, val in item.items()
            }

        if isinstance(item, (list, tuple)):
            return [
                clean(val)
                for val in item
            ]

        if isinstance(item, np.ndarray):
            return clean(item.tolist())

        if isinstance(item, np.generic):
            return clean(item.item())

        if isinstance(item, float) and not np.isfinite(item):
            return None

        return item

    return clean(value)


# =========================================================
# VISION MODEL RESULT
# =========================================================

def get_vision_result(image_path: Path):

    VISION_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    prediction_json = (
        VISION_OUTPUT_DIR
        / f"{image_path.stem}_prediction.json"
    )

    if prediction_json.exists():
        prediction_json.unlink()

    returned = predict_vision.predict(
        str(image_path)
    )

    if isinstance(returned, dict):
        return returned

    if prediction_json.exists():

        return json.loads(
            prediction_json.read_text(
                encoding="utf-8"
            )
        )

    raise RuntimeError(
        "Vision predictor returned no result "
        "and created no prediction JSON."
    )


# =========================================================
# RUN ALL THREE MODELS
# =========================================================

def run_inspection(
    image_path: Path,
    wav_path: Path,
    mat_path: Path,
    acoustic_package: dict
):

    report = {

        "system": (
            "AI Smart Automated Robot - "
            "Multimodal Industrial Inspection"
        ),

        "report_version": "3.0",

        "inspection_id": (
            "INS-"
            + datetime.now().strftime("%Y%m%d-%H%M%S")
            + "-"
            + uuid.uuid4().hex[:6]
        ),

        "generated_at": (
            datetime.now().astimezone().isoformat()
        ),

        "inputs": {
            "image": image_path.name,
            "wav": wav_path.name,
            "mat": mat_path.name
        },

        "vision": None,
        "acoustic": None,
        "vibration": None,
        "errors": {},
    }

    # Vision model

    try:

        report["vision"] = get_vision_result(
            image_path
        )

    except Exception as exc:

        report["errors"]["vision"] = (
            f"{type(exc).__name__}: {exc}"
        )

    # Acoustic model

    try:

        report["acoustic"] = (
            inspection.predict_acoustic(
                wav_path,
                acoustic_package
            )
        )

    except Exception as exc:

        report["errors"]["acoustic"] = (
            f"{type(exc).__name__}: {exc}"
        )

    # Vibration model

    try:

        report["vibration"] = (
            inspection.predict_vibration_file(
                mat_path
            )
        )

    except Exception as exc:

        report["errors"]["vibration"] = (
            f"{type(exc).__name__}: {exc}"
        )

    # Rule-based multimodal assessment

    report["assessment"] = (
        inspection.build_assessment(
            report["vision"],
            report["acoustic"],
            report["vibration"],
            report["errors"]
        )
    )

    report["errors_summary"] = {

        "failed_modalities": list(
            report["errors"].keys()
        ),

        "failure_count": len(
            report["errors"]
        ),
    }

    return safe_json(report)


# =========================================================
# STATUS ICON
# =========================================================

def status_icon(status):

    value = str(status).upper()

    if value in {
        "NORMAL",
        "NO_ALERT",
        "NO_ANOMALY_FLAGGED"
    }:
        return "🟢"

    if value in {
        "ANOMALY",
        "ANOMALOUS",
        "DEFECTIVE",
        "FAULT_INDICATED",
        "FAULT_SUSPECTED",
        "ALERT"
    }:
        return "🔴"

    return "🟠"


# =========================================================
# IMAGE RESIZING
# =========================================================

def fit_image_bytes(
    image_bytes: bytes,
    size=(645, 420)
):

    try:

        with Image.open(
            BytesIO(image_bytes)
        ) as image:

            image = image.convert("RGB")

            fitted = ImageOps.contain(
                image,
                size
            )

            canvas = Image.new(
                "RGB",
                size,
                "#f3f8fc"
            )

            x = (
                size[0] - fitted.width
            ) // 2

            y = (
                size[1] - fitted.height
            ) // 2

            canvas.paste(
                fitted,
                (x, y)
            )

            output = BytesIO()

            canvas.save(
                output,
                format="PNG"
            )

            return output.getvalue()

    except Exception:

        return image_bytes


# =========================================================
# PLACEHOLDER IMAGE
# =========================================================

def make_placeholder_bytes(message: str):

    fig, ax = plt.subplots(
        figsize=(4.3, 2.8),
        dpi=150
    )

    fig.patch.set_facecolor("#f7fbff")
    ax.set_facecolor("#ffffff")

    ax.text(
        .5,
        .5,
        message,
        ha="center",
        va="center",
        color="#60758a",
        fontsize=10,
        wrap=True
    )

    ax.set_axis_off()

    fig.tight_layout()

    output = BytesIO()

    fig.savefig(
        output,
        format="png",
        facecolor=fig.get_facecolor(),
        bbox_inches="tight"
    )

    plt.close(fig)

    output.seek(0)

    return output.getvalue()


# =========================================================
# DIAGNOSTIC CHART GENERATION
# =========================================================

def make_chart_bytes(kind: str, result: dict):

    fig, ax = plt.subplots(
        figsize=(4.3, 2.8),
        dpi=150
    )

    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f4f8fc")

    for spine in ax.spines.values():
        spine.set_color("#d4e2ed")

    ax.tick_params(
        colors="#344e65",
        labelsize=8
    )

    ax.title.set_color("#173b5e")
    ax.xaxis.label.set_color("#344e65")
    ax.yaxis.label.set_color("#344e65")

    # -----------------------------------------------------
    # ACOUSTIC CHART
    # -----------------------------------------------------

    if kind == "acoustic":

        score = result.get("score")
        threshold = result.get("threshold")

        score_type = result.get(
            "score_type",
            ""
        )

        if score is None:

            ax.text(
                .5,
                .5,
                "Acoustic Prediction\n"
                + str(result.get("status", "Unavailable")),
                ha="center",
                va="center",
                color="#2878b8",
                fontsize=11,
                transform=ax.transAxes
            )

            ax.set_axis_off()

        else:

            score = float(score)
            threshold = float(threshold or 0)

            if score_type == "anomaly_probability":

                score_label = "Anomaly score"

                ax.bar(
                    [score_label],
                    [score],
                    color="#319bc5",
                    width=.48
                )

                ax.axhline(
                    threshold,
                    color="#e39b35",
                    linestyle="--",
                    linewidth=2,
                    label=f"Decision limit: {threshold:.2f}"
                )

                ax.set_ylim(0, 1.05)

                ax.set_ylabel(
                    "Anomaly probability"
                )

            elif score_type == "decision_function_margin":

                score_label = "SVM margin"

                ax.bar(
                    [score_label],
                    [score],
                    color="#319bc5",
                    width=.48
                )

                ax.axhline(
                    0,
                    color="#e39b35",
                    linestyle="--",
                    linewidth=2,
                    label="Decision boundary: 0"
                )

                ax.set_ylabel(
                    "Decision function margin"
                )

            else:

                ax.text(
                    .5,
                    .5,
                    "Class prediction: "
                    + str(result.get("status")),
                    ha="center",
                    va="center",
                    color="#2878b8",
                    transform=ax.transAxes
                )

                ax.set_axis_off()

            if score_type in (
                "anomaly_probability",
                "decision_function_margin"
            ):

                ax.set_title(
                    "Acoustic Model Result",
                    fontsize=10,
                    fontweight="bold",
                    pad=12
                )

                ax.legend(
                    fontsize=7,
                    frameon=False,
                    loc="best"
                )

                ax.grid(
                    axis="y",
                    alpha=.2
                )

    # -----------------------------------------------------
    # VIBRATION CHART
    # -----------------------------------------------------

    elif kind == "vibration":

        values = (
            result.get("class_percentages", {})
            or {}
        )

        if values:

            labels = list(values.keys())

            amounts = [
                float(values[x])
                for x in labels
            ]

            ax.barh(
                labels,
                amounts,
                color="#48b69b"
            )

            ax.set_xlim(0, 100)

            ax.set_title(
                "Vibration Fault Classification",
                fontsize=10,
                fontweight="bold",
                pad=12
            )

            ax.set_xlabel(
                "Predicted windows (%)"
            )

            ax.grid(
                axis="x",
                alpha=.2
            )

            ax.tick_params(
                axis="y",
                labelsize=7
            )

        else:

            ax.text(
                .5,
                .5,
                "Vibration distribution unavailable",
                ha="center",
                va="center",
                color="#60758a",
                transform=ax.transAxes
            )

            ax.set_axis_off()

    fig.tight_layout(pad=1.2)

    output = BytesIO()

    fig.savefig(
        output,
        format="png",
        facecolor=fig.get_facecolor(),
        bbox_inches="tight"
    )

    plt.close(fig)

    output.seek(0)

    return output.getvalue()


# =========================================================
# WORD REPORT GENERATION
# =========================================================

def build_docx_report(
    report: dict,
    chart_images: dict
):

    doc = Document()

    section = doc.sections[0]

    section.top_margin = Inches(.55)
    section.bottom_margin = Inches(.65)
    section.left_margin = Inches(.65)
    section.right_margin = Inches(.65)

    # -----------------------------------------------------
    # DOCUMENT STYLING
    # -----------------------------------------------------

    styles = doc.styles

    styles["Normal"].font.name = "Aptos"
    styles["Normal"].font.size = Pt(9)

    styles["Normal"].font.color.rgb = (
        RGBColor(45, 58, 72)
    )

    # Header

    header = section.header.paragraphs[0]

    header.text = (
        "SMART AI AUTOMATED ROBOT"
        "  |  INDUSTRIAL INSPECTION SYSTEM"
    )

    header.alignment = (
        WD_ALIGN_PARAGRAPH.CENTER
    )

    for run in header.runs:

        run.bold = True
        run.font.size = Pt(8)

        run.font.color.rgb = (
            RGBColor(35, 105, 145)
        )

    # -----------------------------------------------------
    # REPORT TITLE
    # -----------------------------------------------------

    title = doc.add_paragraph()

    title.alignment = (
        WD_ALIGN_PARAGRAPH.CENTER
    )

    title.paragraph_format.space_before = Pt(15)
    title.paragraph_format.space_after = Pt(8)

    run = title.add_run(
        "INSPECTION RESULTS OF AI INSPECTION ROBOT"
    )

    run.bold = True
    run.font.size = Pt(17)

    run.font.color.rgb = (
        RGBColor(20, 65, 105)
    )

    subtitle = doc.add_paragraph(
        "Multimodal AI-based Industrial Condition Assessment"
    )

    subtitle.alignment = (
        WD_ALIGN_PARAGRAPH.CENTER
    )

    subtitle.paragraph_format.space_after = Pt(16)

    # -----------------------------------------------------
    # INSPECTION INFORMATION
    # -----------------------------------------------------

    assessment = report.get(
        "assessment",
        {}
    )

    info = doc.add_table(
        rows=3,
        cols=2
    )

    info.style = "Light Shading Accent 1"

    fields = [

        (
            "Inspection ID",
            report.get("inspection_id", "-")
        ),

        (
            "Generated",
            report.get(
                "generated_at",
                "-"
            ).replace("T", " ")
        ),

        (
            "Final decision",
            assessment.get("decision", "-")
        ),

        (
            "Provisional severity",
            assessment.get(
                "provisional_severity",
                "-"
            )
        ),

        (
            "Suspected fault",
            assessment.get(
                "suspected_fault",
                "-"
            )
        ),

        (
            "Acoustic section",
            str(
                (report.get("acoustic") or {}).get(
                    "section",
                    "Not detected"
                )
            )
        ),
    ]

    for i, (label, value) in enumerate(fields):

        cell = info.cell(
            i // 2,
            i % 2
        )

        cell.text = f"{label}: {value}"

        for paragraph in cell.paragraphs:

            for run in paragraph.runs:

                run.font.size = Pt(8)

                if label in (
                    "Final decision",
                    "Provisional severity"
                ):

                    run.bold = True

    # -----------------------------------------------------
    # EXECUTIVE SUMMARY
    # -----------------------------------------------------

    doc.add_heading(
        "1. Executive Summary",
        level=1
    )

    doc.add_paragraph(
        assessment.get(
            "reason",
            "No assessment summary available."
        )
    )

    doc.add_paragraph(
        "The inspection combines the outputs of "
        "computer vision, acoustic analysis and "
        "vibration-based fault classification. "
        "A rule-based assessment interprets the "
        "individual model predictions."
    )

    # -----------------------------------------------------
    # SENSOR MODEL RESULTS
    # -----------------------------------------------------

    doc.add_heading(
        "2. Sensor Model Results",
        level=1
    )

    table = doc.add_table(
        rows=1,
        cols=4
    )

    table.style = "Light Shading Accent 1"

    headers = [
        "AI module",
        "Prediction",
        "Model / score",
        "Threshold / detail"
    ]

    for i, label in enumerate(headers):

        table.rows[0].cells[i].text = label

    vision = report.get("vision") or {}
    acoustic = report.get("acoustic") or {}
    vibration = report.get("vibration") or {}

    # Acoustic score display

    acoustic_score = acoustic.get("score")

    acoustic_type = acoustic.get(
        "score_type",
        ""
    )

    if acoustic_score is None:

        acoustic_score_text = (
            "Class prediction only"
        )

        acoustic_limit_text = (
            "Not applicable"
        )

    elif acoustic_type == "anomaly_probability":

        acoustic_score_text = (
            f"{acoustic.get('model_name', 'Model')}: "
            f"{float(acoustic_score):.4f}"
        )

        acoustic_limit_text = (
            f"Decision limit: "
            f"{float(acoustic.get('threshold', 0.5)):.2f}"
        )

    else:

        acoustic_score_text = (
            f"{acoustic.get('model_name', 'Model')}: "
            f"{float(acoustic_score):.4f}"
        )

        acoustic_limit_text = (
            "SVM boundary: 0.00"
        )

    # Vibration result

    vibration_classes = (
        vibration.get("class_percentages", {})
        or {}
    )

    if vibration_classes:

        top_vibration = max(
            vibration_classes,
            key=vibration_classes.get
        )

        vibration_detail = (
            f"{top_vibration}: "
            f"{float(vibration_classes[top_vibration]):.1f}%"
        )

    else:

        vibration_detail = (
            "Class distribution unavailable"
        )

    rows = [

        (
            "Vision AI",
            vision.get("prediction", "Unavailable"),

            (
                f"{float(vision.get('ensemble_score')):.4f}"
                if vision.get("ensemble_score") is not None
                else "-"
            ),

            (
                f"{float(vision.get('threshold')):.4f}"
                if vision.get("threshold") is not None
                else "-"
            )
        ),

        (
            "Acoustic AI",
            acoustic.get("status", "Unavailable"),
            acoustic_score_text,
            acoustic_limit_text
        ),

        (
            "Vibration AI",
            vibration.get("predicted_class", "Unavailable"),

            (
                f"{vibration.get('fault_windows', '-')}/"
                f"{vibration.get('total_windows', '-')} windows"
            ),

            vibration_detail
        )
    ]

    for row in rows:

        cells = table.add_row().cells

        for i, value in enumerate(row):

            cells[i].text = str(value)

    # -----------------------------------------------------
    # DIAGNOSTIC VISUALIZATIONS
    # -----------------------------------------------------

    doc.add_heading(
        "3. Diagnostic Visualizations",
        level=1
    )

    charts = doc.add_table(
        rows=1,
        cols=3
    )

    charts.autofit = True

    chart_names = {

        "heatmap": "Vision Heatmap",

        "acoustic": "Acoustic Model Output",

        "vibration": "Vibration Classification"
    }

    for idx, key in enumerate(
        ["heatmap", "acoustic", "vibration"]
    ):

        cell = charts.cell(
            0,
            idx
        )

        paragraph = cell.paragraphs[0]

        paragraph.alignment = (
            WD_ALIGN_PARAGRAPH.CENTER
        )

        paragraph.add_run(
            chart_names[key]
        ).bold = True

        if chart_images.get(key):

            picture_paragraph = cell.add_paragraph()

            picture_paragraph.alignment = (
                WD_ALIGN_PARAGRAPH.CENTER
            )

            picture_paragraph.add_run().add_picture(
                BytesIO(chart_images[key]),
                width=Inches(2.05)
            )

        else:

            cell.add_paragraph(
                "Visualization not available."
            )

    # -----------------------------------------------------
    # RECOMMENDED PRECAUTIONS
    # -----------------------------------------------------

    doc.add_heading(
        "4. Recommended Precautions and Maintenance Actions",
        level=1
    )

    actions = assessment.get(
        "recommended_actions",
        []
    )

    if actions:

        for action in actions:

            doc.add_paragraph(
                str(action),
                style="List Bullet"
            )

    else:

        doc.add_paragraph(
            "No specific precautionary action was generated."
        )

    # -----------------------------------------------------
    # INPUT FILE RECORD
    # -----------------------------------------------------

    doc.add_heading(
        "5. Input File Record",
        level=1
    )

    for label, filename in (
        report.get("inputs") or {}
    ).items():

        doc.add_paragraph(
            f"{label.capitalize()}: {filename}",
            style="List Bullet"
        )

    # -----------------------------------------------------
    # FOOTER WITH SPACING
    # -----------------------------------------------------

    footer = section.footer.paragraphs[0]

    footer.alignment = (
        WD_ALIGN_PARAGRAPH.CENTER
    )

    footer.paragraph_format.space_before = Pt(10)
    footer.paragraph_format.space_after = Pt(6)

    footer_run = footer.add_run(
        "SMART AI AUTOMATED ROBOT"
        "  |  MULTIMODAL INDUSTRIAL INSPECTION"
    )

    footer_run.bold = True
    footer_run.font.size = Pt(8)

    footer_run.font.color.rgb = (
        RGBColor(35, 105, 145)
    )

    footer_note = section.footer.add_paragraph()

    footer_note.alignment = (
        WD_ALIGN_PARAGRAPH.CENTER
    )

    footer_note.paragraph_format.space_before = Pt(7)
    footer_note.paragraph_format.space_after = Pt(0)

    footer_note.add_run(
        "Inspection Results and Recommended Maintenance Actions"
    ).font.size = Pt(8)

    output = BytesIO()

    doc.save(output)

    output.seek(0)

    return output.getvalue()


# =========================================================
# MAIN HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-kicker">
            SMART AUTOMATION
            • MULTIMODAL AI
            • CONDITION MONITORING
        </div>

        <div class="hero-title">
            🏭 AI Industrial Inspection Robot
        </div>

        <div class="hero-sub">
            An intelligent industrial inspection platform
            integrating computer vision, acoustic analysis
            and vibration-based fault classification for
            automated equipment condition assessment.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

if "page_navigation" not in st.session_state:
    st.session_state["page_navigation"] = "01 | Introduction"


def navigate_to_inspection():
    """Switch the sidebar radio to the inspection page safely."""
    st.session_state["page_navigation"] = "02 | Prediction of Given Data"


with st.sidebar:

    st.markdown(
        """
        <div style="
            padding:10px 4px 15px 4px;
            text-align:center;
        ">

            <div style="
                font-size:2.4rem;
                margin-bottom:8px;
            ">
                🏭
            </div>

            <div style="
                font-size:1.15rem;
                font-weight:850;
                color:#173b5e;
            ">
                Inspection Console
            </div>

            <div style="
                font-size:.78rem;
                color:#60758a;
                margin-top:5px;
            ">
                SMART AI AUTOMATED ROBOT
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    selected_page = st.radio(
        "NAVIGATION",

        [
            "01 | Introduction",
            "02 | Prediction of Given Data"
        ],

        key="page_navigation",

        label_visibility="visible"
    )

    st.markdown(
        """
        <div style="height:20px;"></div>

        <div style="
            border-radius:14px;
            background:#eaf4fa;
            padding:13px;
            color:#49677f;
            font-size:.82rem;
            line-height:1.6;
        ">

            Explore the intelligent inspection
            workflow and analyse equipment
            condition using three specialised
            AI models.

        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# INTRODUCTION PAGE
# =========================================================

if selected_page == "01 | Introduction":

    st.markdown(
        """
        <div class="section-heading">
            01 | Introduction to the AI Inspection System
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="notice">

            <b>
                One intelligent inspection system.
                Three specialised AI models.
            </b>

            <br><br>

            The Smart AI Automated Robot is designed
            to analyse industrial equipment condition
            through three complementary sensing
            modalities: visual inspection, acoustic
            monitoring and vibration analysis.

            Each trained model examines its own input
            and contributes its prediction to a
            combined rule-based assessment.

        </div>
        """,
        unsafe_allow_html=True
    )

    # Three model cards

    st.markdown(
        "### The Three AI Intelligence Modules"
    )

    col1, col2, col3 = st.columns(
        3,
        gap="large"
    )

    with col1:

        with st.container(border=True):

            st.markdown("## 📷")

            st.markdown(
                "### Vision Intelligence"
            )

            st.markdown(
                "**PatchCore V6**"
            )

            st.write(
                "Analyses industrial inspection images "
                "to identify visual anomalies and "
                "unusual surface patterns."
            )

            st.caption(
                "Input: Industrial image"
            )

    with col2:

        with st.container(border=True):

            st.markdown("## 🎙️")

            st.markdown(
                "### Acoustic Intelligence"
            )

            st.markdown(
                "**Section-wise ML Models**"
            )

            st.write(
                "Extracts acoustic characteristics "
                "from machine recordings and uses "
                "section-specific trained classifiers "
                "to identify abnormal sound patterns."
            )

            st.caption(
                "Input: WAV recording"
            )

    with col3:

        with st.container(border=True):

            st.markdown("## 📈")

            st.markdown(
                "### Vibration Intelligence"
            )

            st.markdown(
                "**Random Forest**"
            )

            st.write(
                "Analyses vibration signal windows "
                "to classify bearing conditions and "
                "identify possible mechanical faults."
            )

            st.caption(
                "Input: MAT recording"
            )

    st.markdown("---")

    # Actual workflow

    st.markdown(
        "### How the Three Models Work Together"
    )

    st.markdown(
        """
        <div class="notice">

        <b>Stage 1 – Data Acquisition</b>

        Image, acoustic and vibration data are
        provided to the inspection application.

        <br><br>

        <b>Stage 2 – Independent AI Analysis</b>

        Each specialised model processes its
        corresponding input and generates a
        prediction.

        <br><br>

        <b>Stage 3 – Multimodal Result Interpretation</b>

        The three model outputs are interpreted
        through a rule-based assessment process.
        This identifies possible abnormalities,
        suspected faults and inspection priorities.

        <br><br>

        <b>Stage 4 – Inspection Report Generation</b>

        The application presents model predictions,
        diagnostic visualizations and recommended
        precautionary measures.

        </div>
        """,
        unsafe_allow_html=True
    )

    # Fusion explanation

    st.markdown(
        "### Intelligent Fusion and Decision Process"
    )

    st.write(
        "The system uses a rule-based interpretation "
        "of the available model predictions. "
        "It does not average raw scores from "
        "different models. Instead, it evaluates "
        "their individual outputs and applies "
        "predefined decision rules to produce "
        "a consolidated inspection result."
    )

    st.info(
        "The current application performs multimodal "
        "model-output interpretation. It does not "
        "claim synchronized live sensor fusion."
    )

    st.markdown("---")

    st.markdown(
        "### Ready to Perform an Inspection?"
    )

    st.write(
        "Move to the Prediction of Given Data section "
        "to upload an industrial image, acoustic "
        "recording and vibration signal, then run "
        "the AI inspection workflow."
    )

    st.button(
        "Proceed to AI Inspection →",
        type="primary",
        use_container_width=True,
        on_click=navigate_to_inspection
    )


# =========================================================
# PREDICTION OF GIVEN DATA PAGE
# =========================================================

if selected_page == "02 | Prediction of Given Data":

    st.markdown(
        """
        <div class="section-heading">
            02 | Prediction of Given Data
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="notice">

        <b>AI-powered industrial inspection</b>

        Upload the image, acoustic recording and
        vibration signal. The three trained AI models
        will analyse their respective inputs and
        generate a consolidated inspection assessment.

        </div>
        """,
        unsafe_allow_html=True
    )

    # -----------------------------------------------------
    # INPUT UPLOAD SECTION
    # -----------------------------------------------------

    st.markdown(
        "### Upload Inspection Data"
    )

    col1, col2, col3 = st.columns(
        3,
        gap="large"
    )

    with col1:

        with st.container(border=True):

            st.markdown("### 📷 Vision Input")

            st.caption(
                "PatchCore V6 – Image Anomaly Detection"
            )

            image_file = st.file_uploader(
                "Upload industrial image",
                type=["png", "jpg", "jpeg", "bmp"],
                key="vision_upload"
            )

    with col2:

        with st.container(border=True):

            st.markdown("### 🎙️ Acoustic Input")

            st.caption(
                "Section-wise ML – Acoustic Analysis"
            )

            wav_file = st.file_uploader(
                "Upload WAV recording",
                type=["wav"],
                key="audio_upload"
            )

    with col3:

        with st.container(border=True):

            st.markdown("### 📈 Vibration Input")

            st.caption(
                "Random Forest – Bearing Fault Classification"
            )

            mat_file = st.file_uploader(
                "Upload MAT recording",
                type=["mat"],
                key="vibration_upload"
            )

    # -----------------------------------------------------
    # INPUT VALIDATION
    # -----------------------------------------------------

    section_number = None
    section_error = None

    if wav_file is not None:

        try:

            section_number = inspection.parse_section(
                wav_file.name
            )

            st.success(
                f"Acoustic section detected: {section_number}"
            )

        except Exception as exc:

            section_error = str(exc)

            st.error(
                f"Acoustic section identification failed: {exc}"
            )

    model_ready = ACOUSTIC_MODEL_PATH.is_file()

    if not model_ready:

        st.error(
            "Section-wise acoustic model bundle is missing."
        )

    if image_file and wav_file and mat_file:

        st.success(
            "All three inspection inputs have been uploaded."
        )

    else:

        st.info(
            "Upload all three input files to enable AI inspection."
        )

    # -----------------------------------------------------
    # LOAD ACOUSTIC MODEL
    # -----------------------------------------------------

    acoustic_package = None

    if model_ready:

        try:

            acoustic_package = load_acoustic_package(
                str(ACOUSTIC_MODEL_PATH),
                ACOUSTIC_MODEL_PATH.stat().st_mtime
            )

        except Exception as exc:

            st.error(
                f"Acoustic model loading failed: {exc}"
            )

    # -----------------------------------------------------
    # RUN INSPECTION BUTTON
    # -----------------------------------------------------

    st.markdown("---")

    can_run = all([
        image_file,
        wav_file,
        mat_file,
        model_ready,
        acoustic_package is not None,
        section_number is not None
    ])

    run_clicked = st.button(
        "🚀 RUN AI INDUSTRIAL INSPECTION",
        type="primary",
        use_container_width=True,
        disabled=not can_run
    )

    if run_clicked:

        temp_paths = {}

        try:

            # Save uploaded files temporarily

            uploaded_items = {
                "image": image_file,
                "wav": wav_file,
                "mat": mat_file
            }

            for key, uploaded in uploaded_items.items():

                suffix = Path(
                    uploaded.name
                ).suffix

                # Preserve the uploaded WAV filename in the temporary
                # filename so parse_section() can still detect section_00,
                # section_01 or section_02 after the upload is saved.
                temp_prefix = (
                    f"{Path(uploaded.name).stem}_"
                    if key == "wav"
                    else ""
                )

                temp_file = tempfile.NamedTemporaryFile(
                    delete=False,
                    prefix=temp_prefix,
                    suffix=suffix
                )

                temp_file.write(
                    uploaded.getvalue()
                )

                temp_file.close()

                temp_paths[key] = Path(
                    temp_file.name
                )

            # Run all models

            with st.spinner(
                "Running Vision, Acoustic and Vibration AI models..."
            ):

                report = run_inspection(
                    temp_paths["image"],
                    temp_paths["wav"],
                    temp_paths["mat"],
                    acoustic_package
                )

            # Preserve original uploaded filenames

            report["inputs"] = {
                "image": image_file.name,
                "wav": wav_file.name,
                "mat": mat_file.name
            }

            # Save report

            OUTPUT_DIR.mkdir(
                parents=True,
                exist_ok=True
            )

            json_path = (
                OUTPUT_DIR / "latest_inspection.json"
            )

            json_path.write_text(
                json.dumps(
                    report,
                    indent=4,
                    ensure_ascii=False,
                    allow_nan=False
                ),
                encoding="utf-8"
            )

            # Store results in session

            st.session_state["inspection_report"] = report

            st.session_state["inspection_image_bytes"] = (
                image_file.getvalue()
            )

            st.session_state["inspection_acoustic"] = (
                report.get("acoustic") or {}
            )

            st.session_state["inspection_vibration"] = (
                report.get("vibration") or {}
            )

            st.success(
                "AI inspection completed successfully!"
            )

        except Exception as exc:

            st.error(
                f"Inspection execution failed: {exc}"
            )

        finally:

            # Remove temporary uploaded files

            for path in temp_paths.values():

                try:
                    path.unlink(missing_ok=True)

                except Exception:
                    pass

    # -----------------------------------------------------
    # DISPLAY INSPECTION RESULTS
    # -----------------------------------------------------

    report = st.session_state.get(
        "inspection_report"
    )

    if report:

        st.markdown("---")

        st.markdown(
            "## 📊 AI Inspection Results"
        )

        assessment = report.get(
            "assessment",
            {}
        )

        vision = report.get(
            "vision"
        ) or {}

        acoustic = report.get(
            "acoustic"
        ) or {}

        vibration = report.get(
            "vibration"
        ) or {}

        decision = assessment.get(
            "decision",
            "NEEDS_REVIEW"
        )

        severity = assessment.get(
            "provisional_severity",
            "Not available"
        )

        suspected_fault = assessment.get(
            "suspected_fault",
            "Not identified"
        )

        # -------------------------------------------------
        # FINAL DECISION
        # -------------------------------------------------

        st.markdown(
            "### Overall Inspection Assessment"
        )

        result1, result2, result3 = st.columns(3)

        with result1:

            st.metric(
                "Final Decision",
                decision
            )

        with result2:

            st.metric(
                "Severity",
                severity
            )

        with result3:

            st.metric(
                "Suspected Fault",
                str(suspected_fault)
            )

        st.markdown(
            f"""
            <div class="notice">

            <b>Inspection Summary</b>

            <br><br>

            {assessment.get(
                "reason",
                "Assessment summary unavailable."
            )}

            </div>
            """,
            unsafe_allow_html=True
        )

        # -------------------------------------------------
        # INDIVIDUAL MODEL PREDICTIONS
        # -------------------------------------------------

        st.markdown(
            "### Individual AI Model Predictions"
        )

        vcol, acol, vibcol = st.columns(3)

        with vcol:

            st.markdown(
                "#### 📷 Vision AI"
            )

            vision_prediction = vision.get(
                "prediction",
                vision.get("status", "Unavailable")
            )

            st.markdown(
                f"**Result:** {vision_prediction}"
            )

            if vision.get("ensemble_score") is not None:

                st.write(
                    "Anomaly score:",
                    round(
                        float(vision["ensemble_score"]),
                        4
                    )
                )

        with acol:

            st.markdown(
                "#### 🎙️ Acoustic AI"
            )

            st.markdown(
                f"**Result:** {acoustic.get('status', 'Unavailable')}"
            )

            if acoustic:

                st.write(
                    "Model:",
                    acoustic.get("model_name", "-")
                )

                st.write(
                    "Section:",
                    acoustic.get("section", "-")
                )

                if acoustic.get("score") is not None:

                    score_type = acoustic.get(
                        "score_type"
                    )

                    score = float(
                        acoustic["score"]
                    )

                    if score_type == "anomaly_probability":

                        st.write(
                            f"Anomaly probability: {score:.4f}"
                        )

                        st.write(
                            f"Decision limit: "
                            f"{float(acoustic.get('threshold', 0.5)):.2f}"
                        )

                    else:

                        st.write(
                            f"SVM decision margin: {score:.4f}"
                        )

                        st.write(
                            "Decision boundary: 0.00"
                        )

        with vibcol:

            st.markdown(
                "#### 📈 Vibration AI"
            )

            st.markdown(
                "**Result:** "
                + str(
                    vibration.get(
                        "predicted_class",
                        "Unavailable"
                    )
                )
            )

            if vibration:

                st.write(
                    "Fault windows:",
                    vibration.get("fault_windows", "-")
                )

                st.write(
                    "Total windows:",
                    vibration.get("total_windows", "-")
                )

        # -------------------------------------------------
        # DIAGNOSTIC VISUALIZATIONS
        # -------------------------------------------------

        st.markdown("---")

        st.markdown(
            "### Diagnostic Visualizations"
        )

        heatmap_bytes = None

        heatmap_path = (
            vision.get("heatmap_path")
            or vision.get("visualization_path")
        )

        if heatmap_path:

            heatmap_path = Path(
                str(heatmap_path)
            )

            if not heatmap_path.is_absolute():

                heatmap_path = (
                    ROOT / heatmap_path
                )

            if heatmap_path.is_file():

                heatmap_bytes = (
                    heatmap_path.read_bytes()
                )

        if heatmap_bytes is None:

            heatmap_bytes = make_placeholder_bytes(
                "Vision heatmap unavailable"
            )

        acoustic_chart = make_chart_bytes(
            "acoustic",
            acoustic
        )

        vibration_chart = make_chart_bytes(
            "vibration",
            vibration
        )

        chart1, chart2, chart3 = st.columns(
            3,
            gap="medium"
        )

        with chart1:

            st.markdown(
                "**Vision Heatmap**"
            )

            st.image(
                fit_image_bytes(heatmap_bytes),
                use_container_width=True
            )

        with chart2:

            st.markdown(
                "**Acoustic Model Score**"
            )

            st.image(
                fit_image_bytes(acoustic_chart),
                use_container_width=True
            )

        with chart3:

            st.markdown(
                "**Vibration Class Distribution**"
            )

            st.image(
                fit_image_bytes(vibration_chart),
                use_container_width=True
            )

        # -------------------------------------------------
        # PRECAUTIONARY MEASURES ONLY
        # -------------------------------------------------

        st.markdown("---")

        st.markdown(
            "### 🛠️ Recommended Precautionary Measures"
        )

        actions = assessment.get(
            "recommended_actions",
            []
        )

        if actions:

            for index, action in enumerate(
                actions,
                start=1
            ):

                st.markdown(
                    f"""
                    <div class="precaution">

                        <b>{index:02d}.</b>
                        {action}

                    </div>
                    """,
                    unsafe_allow_html=True
                )

        else:

            st.info(
                "No specific precautionary measures "
                "were generated for this inspection."
            )

        # -------------------------------------------------
        # DOWNLOAD REPORTS
        # -------------------------------------------------

        st.markdown("---")

        st.markdown(
            "### 📄 Download Inspection Reports"
        )

        try:

            chart_images = {
                "heatmap": fit_image_bytes(
                    heatmap_bytes
                ),

                "acoustic": fit_image_bytes(
                    acoustic_chart
                ),

                "vibration": fit_image_bytes(
                    vibration_chart
                )
            }

            docx_bytes = build_docx_report(
                report,
                chart_images
            )

            json_bytes = json.dumps(
                report,
                indent=4,
                ensure_ascii=False,
                allow_nan=False
            ).encode("utf-8")

            download1, download2 = st.columns(2)

            with download1:

                st.download_button(
                    label="📘 Download Word Report",
                    data=docx_bytes,
                    file_name=(
                        f"{report['inspection_id']}.docx"
                    ),
                    mime=(
                        "application/vnd.openxmlformats-officedocument."
                        "wordprocessingml.document"
                    ),
                    use_container_width=True
                )

            with download2:

                st.download_button(
                    label="📋 Download JSON Report",
                    data=json_bytes,
                    file_name=(
                        f"{report['inspection_id']}.json"
                    ),
                    mime="application/json",
                    use_container_width=True
                )

        except Exception as exc:

            st.error(
                f"Report generation failed: {exc}"
            )

        st.caption(
            f"Inspection ID: {report.get('inspection_id', '-')}"
        )


# =========================================================
# APPLICATION FOOTER
# =========================================================

st.markdown(
    """
    <div style="
        margin-top:3rem;
        padding-top:1.5rem;
        border-top:1px solid #d9e6ef;
        text-align:center;
        color:#60758a;
        line-height:2;
        font-size:.85rem;
    ">

        <div style="
            font-weight:800;
            color:#2878b8;
            letter-spacing:.08em;
        ">
            SMART AI AUTOMATED ROBOT
        </div>

        <div style="margin-top:9px;">
            Multimodal AI-Based Industrial Inspection System
        </div>

    </div>
    """,
    unsafe_allow_html=True
)
