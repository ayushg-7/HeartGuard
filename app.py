"""Streamlit dashboard for the supplied stroke dataset and trained models."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from heart_monitoring import SCENARIOS, generate_ecg_window, simulated_bpm
from model_predictions import predict_profiles, validate_uploaded_profiles
from patient_history import (
    clear_history,
    delete_history_record,
    list_history_records,
    save_history_record,
)
from synthetic_patients import generate_synthetic_profiles


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "full_data.csv"
MODEL_PATH = PROJECT_DIR / "models" / "stroke_model.joblib"
TARGET_COLUMN = "stroke"
PAGE_NAMES = [
    "Heart Stroke Prediction",
    "Heart Monitoring",
    "Patient Health History",
    "Exploratory Data Analysis",
    "Model Comparison",
]
FEATURE_LABELS = {
    "gender": "Gender",
    "age": "Age (years)",
    "hypertension": "Hypertension",
    "heart_disease": "Heart disease",
    "ever_married": "Ever married",
    "work_type": "Work type",
    "Residence_type": "Residence type",
    "avg_glucose_level": "Average glucose level",
    "bmi": "BMI",
    "smoking_status": "Smoking status",
}

st.set_page_config(
    page_title="HeartGuard | Heart Stroke Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    :root {
        color-scheme: light;
        --ink: #16324f;
        --blue: #2563eb;
        --teal: #0f766e;
        --muted: #526780;
        --surface: #ffffff;
        --canvas: #f5f8fc;
        --border: #d9e3ef;
    }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp, [data-testid="stAppViewContainer"] { background: var(--canvas) !important; color: var(--ink) !important; }
    .stApp, .stApp p, .stApp label, .stApp [data-testid="stCaptionContainer"],
    .stApp [data-testid="stMarkdownContainer"], .stApp li, .stApp span {
        color: var(--ink);
    }
    h1, h2, h3, h4 { font-family: 'Manrope', sans-serif; color: var(--ink) !important; }
    [data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border); }
    [data-testid="stSidebar"] * { color: var(--ink) !important; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label,
    [data-testid="stSidebar"] [data-testid="stRadio"] label p {
        font-size: 1.05rem !important;
        line-height: 1.45 !important;
    }
    [data-testid="stMetric"] {
        background: var(--surface) !important;
        color: var(--ink) !important;
        padding: 18px 20px;
        border: 1px solid var(--border);
        border-radius: 14px;
        box-shadow: 0 3px 14px rgba(26,58,92,.06);
    }
    [data-testid="stMetric"] * { color: var(--ink) !important; opacity: 1 !important; }
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"],
    [data-testid="stMetricDelta"] { color: var(--ink) !important; }
    [data-testid="stCaptionContainer"], .stCaption, small { color: #526780 !important; opacity: 1 !important; }
    .hero { padding: 28px 32px; background: linear-gradient(120deg,#087f8c,#0f766e); border: 1px solid #0b6b70; border-radius: 20px; margin-bottom: 22px; }
    .hero h1 { margin: 0 0 10px 0; font-size: clamp(1.8rem,3vw,2.65rem); }
    .hero h1 { color: #ffffff !important; }
    .hero p { color: #e8fffb !important; max-width: 900px; margin: 0; line-height: 1.65; }
    .eyebrow { color: #d4fff7 !important; font-weight: 700; text-transform: uppercase; letter-spacing: .09em; font-size: .78rem; margin-bottom: 7px; }
    .info-card { background: var(--surface); color: var(--ink); padding: 20px; border: 1px solid var(--border); border-radius: 15px; height: 100%; }
    .info-card h3 { margin-top: 0; font-size: 1.05rem; }
    .info-card p, .info-card li { color: #435a72 !important; line-height: 1.6; }
    div[data-testid="stForm"] { background: var(--surface) !important; color: var(--ink) !important; padding: 22px; border: 1px solid var(--border); border-radius: 16px; }
    .stApp div[data-testid="stWidgetLabel"],
    .stApp div[data-testid="stWidgetLabel"] *,
    .stApp div[data-testid="stSelectbox"] label,
    .stApp div[data-testid="stSlider"] label {
        color: #101828 !important;
        opacity: 1 !important;
        font-size: 1.125rem !important;
        font-weight: 700 !important;
    }
    /* Streamlit's current selectboxes use React Aria rather than BaseWeb. */
    div[data-testid="stSelectbox"] [role="group"][data-rac] {
        background: #173b63 !important;
        border: 1px solid #173b63 !important;
        border-radius: 8px !important;
        min-height: 46px !important;
    }
    div[data-testid="stSelectbox"] input[role="combobox"] {
        background: transparent !important;
        border: 0 !important;
        box-shadow: none !important;
        color: #ffffff !important;
        font-size: 1.125rem !important;
        min-height: 44px !important;
        padding-left: 10px !important;
    }
    div[data-testid="stSelectbox"] input[role="combobox"]::placeholder {
        color: #e5e7eb !important;
        opacity: 1 !important;
    }
    div[data-testid="stSelectbox"] [role="group"][data-rac] button {
        background: #173b63 !important;
        border: 0 !important;
        color: #ffffff !important;
    }
    div[data-testid="stSelectbox"] [role="group"][data-rac] svg {
        color: #ffffff !important;
        fill: currentColor !important;
    }
    [role="listbox"][data-rac] {
        background: #173b63 !important;
        border: 1px solid #333333 !important;
        color: #ffffff !important;
    }
    [role="option"][data-rac] {
        background: #173b63 !important;
        color: #ffffff !important;
        font-size: 1.05rem !important;
    }
    [role="option"][data-rac]:hover,
    [role="option"][data-rac][aria-selected="true"] {
        background: #24517d !important;
        color: #ffffff !important;
    }
    div[data-baseweb="select"] > div {
        background: #173b63 !important;
        border-color: #173b63 !important;
        color: #ffffff !important;
        min-height: 46px;
        font-size: 1.125rem !important;
    }
    div[data-baseweb="select"] *,
    div[data-baseweb="popover"] *,
    div[data-baseweb="menu"] * {
        color: #ffffff !important;
        font-size: 1.125rem !important;
    }
    div[data-baseweb="popover"] [role="listbox"],
    div[data-baseweb="menu"] { background: #173b63 !important; }
    div[data-baseweb="popover"] [role="option"]:hover,
    div[data-baseweb="popover"] [aria-selected="true"],
    div[data-baseweb="menu"] [role="option"]:hover,
    div[data-baseweb="menu"] [aria-selected="true"] { background: #24517d !important; }
    input, textarea {
        background: #ffffff !important;
        color: var(--ink) !important;
        border-color: #9aadc1 !important;
    }
    [data-testid="stSlider"] *, [data-testid="stNumberInput"] * { color: var(--ink) !important; }
    [data-testid="stSlider"] [role="slider"] { background: var(--teal) !important; }
    [data-testid="stFormSubmitButton"] button, [data-testid="stButton"] button {
        background: var(--teal) !important;
        color: #ffffff !important;
        border: 1px solid var(--teal) !important;
        font-weight: 700;
    }
    [data-testid="stFormSubmitButton"] button:hover, [data-testid="stButton"] button:hover {
        background: #115e59 !important;
        border-color: #115e59 !important;
        color: #ffffff !important;
    }
    [data-testid="stDataFrame"], [data-testid="stTable"] { color: var(--ink) !important; }
    [data-testid="stDataFrame"] * { color: var(--ink); }
    [data-testid="stAlert"] { color: #16324f !important; }
    [data-testid="stAlert"] p { color: #16324f !important; }
    .st-key-monitor_range_alert [data-testid="stAlert"],
    .st-key-monitor_live_range_alert [data-testid="stAlert"] {
        background: #fff5f5 !important;
        border: 1px solid #fecaca !important;
        border-left: 4px solid #ef4444 !important;
    }
    .st-key-monitor_range_alert [data-testid="stAlert"] p,
    .st-key-monitor_live_range_alert [data-testid="stAlert"] p {
        color: #b42318 !important;
    }
    .evaluation-metrics {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 14px;
        margin: 10px 0 14px;
    }
    .evaluation-metric {
        min-height: 92px;
        padding: 17px 20px;
        border-radius: 15px;
        color: #ffffff !important;
        background: linear-gradient(135deg, #163b67 0%, #117c83 100%);
        border: 1px solid rgba(255,255,255,.22);
        box-shadow: 0 8px 20px rgba(19,59,92,.18);
    }
    .evaluation-metric:nth-child(3n + 2) {
        background: linear-gradient(135deg, #145b70 0%, #0e8f83 100%);
    }
    .evaluation-metric:nth-child(3n) {
        background: linear-gradient(135deg, #244b83 0%, #187e91 100%);
    }
    .evaluation-metric-label {
        display: block;
        color: #e2f4ff !important;
        font-size: .9rem;
        font-weight: 600;
        letter-spacing: .02em;
        margin-bottom: 8px;
    }
    .evaluation-metric-value {
        display: block;
        color: #ffffff !important;
        font-family: 'Manrope', sans-serif;
        font-size: 1.8rem;
        font-weight: 800;
        line-height: 1.1;
    }
    .evaluation-table-wrap {
        overflow-x: auto;
        border: 1px solid #d2dfed;
        border-radius: 14px;
        box-shadow: 0 6px 18px rgba(25,57,90,.08);
    }
    table.evaluation-table {
        width: 100%;
        border-collapse: collapse;
        background: #ffffff;
        color: #16324f !important;
        font-size: .94rem;
    }
    .evaluation-table th {
        background: #173b63;
        color: #ffffff !important;
        padding: 13px 14px;
        text-align: left;
        white-space: nowrap;
    }
    .evaluation-table td {
        color: #16324f !important;
        padding: 12px 14px;
        border-bottom: 1px solid #e4ebf3;
        white-space: nowrap;
    }
    .evaluation-table tbody tr:nth-child(even) { background: #f0f7fb; }
    .evaluation-table tbody tr:last-child td { border-bottom: 0; }
    .evaluation-table td.metric-value {
        color: #075e59 !important;
        font-weight: 700;
        font-variant-numeric: tabular-nums;
    }
    .prediction-result {
        margin: 18px 0;
        padding: 22px 24px;
        border: 1px solid #b5e4d6;
        border-radius: 18px;
        color: #ffffff !important;
        background: linear-gradient(120deg, #12395e 0%, #0b746e 100%);
        box-shadow: 0 10px 26px rgba(13,74,88,.20);
    }
    .prediction-result.risk-positive {
        background: linear-gradient(120deg, #6c2732 0%, #a6413d 100%);
        border-color: #e8b2a7;
    }
    .prediction-result h3, .prediction-result p,
    .prediction-result strong, .prediction-result span {
        color: #ffffff !important;
    }
    .prediction-result h3 { margin: 0; font-size: 1.5rem; }
    .prediction-score {
        display: inline-block;
        padding: 10px 15px;
        margin-top: 14px;
        border-radius: 12px;
        background: rgba(255,255,255,.16);
        color: #ffffff !important;
        font-weight: 700;
        font-size: 1.1rem;
    }
    .monitor-card {
        padding: 20px;
        min-height: 128px;
        border: 1px solid #cfe0ed;
        border-radius: 16px;
        background: linear-gradient(140deg, #ffffff, #e8f5f6);
        box-shadow: 0 5px 16px rgba(25,57,90,.08);
    }
    .monitor-card-label {
        display: block;
        color: #47637c !important;
        font-size: .95rem;
        font-weight: 600;
    }
    .monitor-card-value {
        display: block;
        margin-top: 8px;
        color: #163b63 !important;
        font-family: 'Manrope', sans-serif;
        font-size: 2rem;
        font-weight: 800;
    }
    .monitor-output {
        padding: 20px 22px;
        border-radius: 16px;
        border: 1px solid #b9d9d7;
        background: linear-gradient(120deg, #e4f3f8, #e1f3ed);
        color: #16324f !important;
    }
    .monitor-output h3, .monitor-output p { color: #16324f !important; }
    @media (max-width: 700px) {
        .evaluation-metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 9px; }
        .evaluation-metric { padding: 14px; min-height: 82px; }
        .evaluation-metric-value { font-size: 1.5rem; }
        .monitor-card-value { font-size: 1.6rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset file is missing: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


@st.cache_resource(show_spinner=False)
def load_bundle() -> dict[str, Any] | None:
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def page_header(title: str, description: str) -> None:
    st.markdown(
        f'<div class="hero"><div class="eyebrow">HeartGuard</div>'
        f"<h1>{title}</h1><p>{description}</p></div>",
        unsafe_allow_html=True,
    )


def card(title: str, body: str) -> None:
    st.markdown(
        f'<div class="info-card"><h3>{title}</h3><p>{body}</p></div>',
        unsafe_allow_html=True,
    )


def require_data() -> pd.DataFrame | None:
    try:
        return load_data()
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
        st.error(f"Unable to read the dataset: {error}")
        return None


def reset_prediction_form(required: list[str]) -> None:
    st.session_state.pop("stroke_prediction", None)
    for name in required:
        st.session_state.pop(f"input_{name}", None)


def render_metrics_table(bundle: dict[str, Any], *, show_all_models: bool) -> None:
    displayed_metrics = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    results = bundle["results"]
    if show_all_models:
        metrics = pd.DataFrame(
            {name: result["metrics"] for name, result in results.items()}
        ).transpose()[displayed_metrics]
    else:
        metrics = pd.DataFrame(
            [
                {
                    name: results["Stacking Classifier"]["metrics"][name]
                    for name in displayed_metrics
                }
            ],
            index=["Stacking Classifier"],
        )
    headers = "".join(f"<th>{name}</th>" for name in metrics.columns)
    rows = []
    for model_name, values in metrics.iterrows():
        cells = "".join(
            f'<td class="metric-value">{value:.3f}</td>' for value in values
        )
        rows.append(f"<tr><td><strong>{model_name}</strong></td>{cells}</tr>")
    st.markdown(
        '<div class="evaluation-table-wrap"><table class="evaluation-table">'
        f"<thead><tr><th>Model</th>{headers}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>",
        unsafe_allow_html=True,
    )


def render_metric_cards(metrics: dict[str, float]) -> None:
    metric_names = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC", "PR-AUC"]
    cards = "".join(
        '<div class="evaluation-metric">'
        f'<span class="evaluation-metric-label">{name}</span>'
        f'<span class="evaluation-metric-value">{metrics[name]:.3f}</span>'
        "</div>"
        for name in metric_names
    )
    st.markdown(f'<div class="evaluation-metrics">{cards}</div>', unsafe_allow_html=True)


def render_home(frame: pd.DataFrame, bundle: dict[str, Any] | None) -> None:
    page_header(
        "HeartGuard",
        "An interactive educational project comparing individual classifiers with a "
        "stacked ensemble, using the supplied dataset and a leakage-aware evaluation workflow.",
    )
    target = frame[TARGET_COLUMN] if TARGET_COLUMN in frame else pd.Series(dtype=int)
    positive = int((target == 1).sum())
    negative = int((target == 0).sum())
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Dataset records", f"{len(frame):,}")
    col2.metric("Predictor columns", str(frame.shape[1] - int(TARGET_COLUMN in frame.columns)))
    col3.metric("Heart stroke cases", f"{positive:,}")
    col4.metric("Missing cells", f"{int(frame.isna().sum().sum()):,}")

    st.markdown("### Project overview")
    left, right = st.columns(2)
    with left:
        card(
            "Objective",
            "Explore how demographic and health indicators relate to the dataset's heart stroke "
            "label, then compare Logistic Regression, Random Forest, Gradient Boosting, "
            "and their Logistic-Regression stacking ensemble.",
        )
    with right:
        card(
            "Workflow",
            "Inspect the real CSV → stratified train/test split → impute, scale, and encode "
            "inside model pipelines → oversample training folds only → evaluate on held-out "
            "records → present results and educational predictions.",
        )
    st.markdown("### Evaluation status")
    if bundle is None:
        st.warning(
            "No trained model artifact was found. From this folder, run "
            "`python train_model.py` to create the model and real evaluation results."
        )
    else:
        st.success(
            f"Trained artifact loaded. Holdout set: "
            f"{bundle['summary']['test_records']:,} stratified records; "
            f"{len(bundle['results'])} models evaluated."
        )
        st.markdown("### Actual model evaluation")
        render_metrics_table(bundle, show_all_models=True)


def render_patient_profile_form(
    frame: pd.DataFrame,
    bundle: dict[str, Any],
    *,
    form_key: str,
    input_key_prefix: str,
    submit_label: str,
    patient_name_key: str | None = None,
) -> pd.DataFrame | None:
    """Render the shared patient-input form and return values only after submission."""
    required = bundle["feature_columns"]
    missing_fields = [name for name in required if name not in frame.columns]
    if missing_fields:
        st.error(f"The dataset is missing model input fields: {', '.join(missing_fields)}")
        return None

    defaults = {"age": 50.0, "avg_glucose_level": 100.0, "bmi": 27.0}
    input_help = {
        "age": "Whole-year age from 1 to 120.",
        "avg_glucose_level": "Average glucose level; bounded by values observed in the dataset.",
        "bmi": "Body mass index; bounded by values observed in the dataset.",
        "hypertension": "Select whether hypertension is recorded for this profile.",
        "heart_disease": "Select whether heart disease is recorded for this profile.",
    }
    form_values: dict[str, Any] = {}
    patient_name = ""
    with st.form(form_key):
        st.subheader("Enter patient profile")
        st.caption("Category options come from the supplied dataset.")
        if patient_name_key is not None:
            patient_name = st.text_input(
                "Patient name",
                key=patient_name_key,
                help="Stored with this history record; not sent to the prediction model.",
            ).strip()
        columns = st.columns(2)
        for index, name in enumerate(required):
            series = frame[name]
            label = FEATURE_LABELS.get(name, name.replace("_", " ").title())
            widget_key = f"{input_key_prefix}{name}"
            with columns[index % 2]:
                if name in bundle["numeric_columns"]:
                    numeric = pd.to_numeric(series, errors="coerce").dropna()
                    minimum = float(numeric.min()) if not numeric.empty else 0.0
                    maximum = float(numeric.max()) if not numeric.empty else 100.0
                    if name in ("hypertension", "heart_disease") and set(
                        numeric.unique()
                    ).issubset({0, 1}):
                        value = st.selectbox(
                            label,
                            options=["No", "Yes"],
                            help=input_help.get(name),
                            key=widget_key,
                        )
                        form_values[name] = int(value == "Yes")
                    else:
                        default = float(
                            defaults.get(
                                name,
                                float(numeric.median()) if not numeric.empty else 0,
                            )
                        )
                        if name == "age":
                            minimum = 1
                            maximum = 120
                            default = int(round(default))
                        default = min(max(default, minimum), maximum)
                        if maximum > minimum:
                            step = (
                                1
                                if name == "age"
                                else (0.1 if name == "bmi" else 0.01)
                            )
                            value = st.slider(
                                label,
                                min_value=minimum,
                                max_value=maximum,
                                value=default,
                                step=min(step, maximum - minimum),
                                help=input_help.get(name),
                                key=widget_key,
                            )
                        else:
                            value = st.number_input(
                                label,
                                min_value=minimum,
                                max_value=maximum,
                                value=minimum,
                                help=input_help.get(name),
                                key=widget_key,
                            )
                        form_values[name] = int(value) if name == "age" else value
                else:
                    choices = sorted(series.dropna().astype(str).unique().tolist())
                    if not choices:
                        st.error(f"No observed category values are available for {name}.")
                        return None
                    form_values[name] = st.selectbox(
                        label,
                        options=choices,
                        key=widget_key,
                    )
        submitted = st.form_submit_button(
            submit_label, type="primary", width="stretch"
        )
    if not submitted:
        return None
    if patient_name_key is not None:
        if not patient_name:
            st.error("Enter the patient's name before predicting.")
            return None
        st.session_state[f"{form_key}_patient_name"] = patient_name
    return pd.DataFrame(
        [{name: form_values[name] for name in required}],
        columns=required,
    )


def predict_profile(
    profile: pd.DataFrame, bundle: dict[str, Any]
) -> dict[str, float | int | None]:
    """Run one profile through the saved stacking estimator and selected threshold."""
    return predict_profiles(profile, bundle)[0]


def render_save_history_button(
    patient_features: dict[str, Any],
    model_output: dict[str, Any],
    *,
    key: str,
    patient_name: str = "",
) -> None:
    if not st.button("Save to History", key=key, type="primary"):
        return
    try:
        save_history_record(
            patient_features,
            model_output,
            patient_name=patient_name,
            simulated_bpm=st.session_state.get("monitor_bpm"),
            simulated_readings=st.session_state.get("monitor_readings", []),
        )
    except (OSError, sqlite3.Error, TypeError, ValueError) as error:
        st.error(f"Could not save this prediction to local history: {error}")
        return
    st.success("Patient details saved to history.")


def render_prediction(frame: pd.DataFrame, bundle: dict[str, Any] | None) -> None:
    page_header(
        "HeartGuard",
        "Enter the patient profile below and select Predict.",
    )
    if bundle is None:
        st.warning("Train the models first by running `python train_model.py`.")
        return

    required = bundle["feature_columns"]
    profile = render_patient_profile_form(
        frame,
        bundle,
        form_key="prediction_form",
        input_key_prefix="input_",
        submit_label="Predict",
        patient_name_key="prediction_patient_name_input",
    )
    reset_col, _ = st.columns([1, 4])
    reset_col.button(
        "Reset form",
        on_click=reset_prediction_form,
        args=(required,),
    )
    if profile is not None:
        try:
            st.session_state["stroke_prediction"] = predict_profile(profile, bundle)
            st.session_state["stroke_prediction_profile"] = profile.iloc[0].to_dict()
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            st.error(f"Prediction failed; check the entered values and trained artifact: {error}")

    result = st.session_state.get("stroke_prediction")
    if result:
        label = (
            "Heart stroke predicted"
            if result["class"] == 1
            else "No heart stroke predicted"
        )
        result_style = "risk-positive" if result["class"] == 1 else ""
        probability_html = (
            f'<span class="prediction-score">Model probability: '
            f'{result["probability"]:.1%}</span>'
            if result["probability"] is not None
            else ""
        )
        st.markdown(
            f'<div class="prediction-result {result_style}">'
            f"<h3>{label}</h3>"
            f"{probability_html}"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Experimental model output based on the supplied dataset; this is not a "
            "medical diagnosis."
        )
        saved_profile = st.session_state.get("stroke_prediction_profile")
        if saved_profile:
            render_save_history_button(
                saved_profile,
                result,
                key="save_prediction_to_history",
                patient_name=st.session_state.get(
                    "prediction_form_patient_name", ""
                ),
            )

    st.markdown("### Model Comparison Results")
    render_metrics_table(bundle, show_all_models=True)


def start_monitoring() -> None:
    if not st.session_state["monitor_running"]:
        st.session_state["monitor_started_at"] = time.monotonic()
        st.session_state["monitor_running"] = True
        st.session_state["monitor_has_started"] = True


def pause_monitoring() -> None:
    if st.session_state["monitor_running"]:
        started_at = st.session_state.get("monitor_started_at")
        if started_at is not None:
            st.session_state["monitor_elapsed"] += time.monotonic() - started_at
        st.session_state["monitor_started_at"] = None
        st.session_state["monitor_running"] = False


def reset_monitoring() -> None:
    st.session_state["monitor_elapsed"] = 0.0
    st.session_state["monitor_started_at"] = (
        time.monotonic() if st.session_state["monitor_running"] else None
    )
    st.session_state["monitor_bpm"] = None
    st.session_state["monitor_readings"] = []
    st.session_state["monitor_last_reading_at"] = None
    st.session_state["monitor_has_started"] = bool(st.session_state["monitor_running"])


def monitoring_elapsed() -> float:
    elapsed = float(st.session_state.get("monitor_elapsed", 0.0))
    started_at = st.session_state.get("monitor_started_at")
    if st.session_state.get("monitor_running") and started_at is not None:
        elapsed += time.monotonic() - started_at
    return elapsed


def render_heart_monitoring(
    frame: pd.DataFrame, bundle: dict[str, Any] | None
) -> None:
    page_header(
        "Live Heart Monitoring",
        "Run a software-only heart-rate and ECG-style waveform demonstration.",
    )
    st.error("SIMULATED DATA — NO PHYSICAL EKG SENSOR CONNECTED.")
    st.info(
        "Every waveform and heart-rate reading is simulated, not a measurement. BPM and "
        "the waveform do not assess stroke risk or diagnose a medical condition. The "
        "saved machine-learning output is an experimental dataset-based prediction, not a "
        "medical diagnosis; these outputs are separate and are not combined into a risk score."
    )
    st.markdown("### How to use this demonstration")
    st.markdown(
        "Choose a sample scenario, or enter a BPM and click **Test Heart Rate** to use "
        "that value as the live demo baseline. Select **Start Monitoring** to animate the "
        "synthetic waveform and BPM, then use **Pause Monitoring** or **Reset** to control "
        "the demonstration. These settings do not describe or assess a patient's actual "
        "heart rhythm."
    )
    with st.container(border=True):
        st.markdown("### If someone may be having a heart stroke")
        st.write(
            "Sudden face drooping, arm weakness, or speech difficulty can be warning signs. "
            "Contact your local emergency services immediately; do not wait for or rely on "
            "this educational dashboard."
        )
    if "monitor_running" not in st.session_state:
        st.session_state.monitor_running = False
        st.session_state.monitor_elapsed = 0.0
        st.session_state.monitor_started_at = None
        st.session_state.monitor_bpm = None
        st.session_state.monitor_readings = []
        st.session_state.monitor_last_reading_at = None
        st.session_state.monitor_has_started = False
        st.session_state.monitor_tested_bpm = None

    scenario_col, alert_col = st.columns([2, 3])
    with scenario_col:
        scenario = st.selectbox(
            "Simulation scenario",
            options=list(SCENARIOS),
            key="monitor_scenario",
        )
    with alert_col:
        st.checkbox(
            "Show informational alert outside selected demo range",
            value=True,
            key="monitor_alert_enabled",
        )
        range_low_col, range_high_col = st.columns(2)
        with range_low_col:
            alert_low = st.slider(
                "Demo range minimum (BPM)",
                min_value=30,
                max_value=150,
                value=60,
                key="monitor_alert_low",
            )
        with range_high_col:
            alert_high = st.slider(
                "Demo range maximum (BPM)",
                min_value=40,
                max_value=180,
                value=100,
                key="monitor_alert_high",
            )
    if alert_low >= alert_high:
        st.error("The demo range minimum must be lower than its maximum.")
        return

    with st.form("manual_bpm_test_form", border=True):
        st.markdown("### Test a simulated heart rate")
        manual_bpm_input = st.number_input(
            "Simulated BPM to test",
            min_value=30,
            max_value=180,
            value=76,
            step=1,
            key="monitor_manual_bpm_input",
        )
        test_submitted = st.form_submit_button(
            "Test Heart Rate",
            type="primary",
            width="stretch",
        )
    if test_submitted:
        if not 30 <= manual_bpm_input <= 180:
            st.error("Enter a simulated BPM from 30 to 180.")
        else:
            st.session_state.monitor_tested_bpm = int(manual_bpm_input)

    tested_bpm = st.session_state.get("monitor_tested_bpm")
    if tested_bpm is not None:
        st.metric("LAST TESTED SIMULATED HEART RATE", f"{tested_bpm} BPM")
        if tested_bpm < alert_low:
            st.write(f"Below selected demo range (minimum: {alert_low} BPM).")
            if st.session_state.monitor_alert_enabled:
                with st.container(key="monitor_range_alert"):
                    st.info(
                        f"Informational demo alert: simulated BPM {tested_bpm} is below "
                        f"the selected minimum of {alert_low} BPM."
                    )
        elif tested_bpm > alert_high:
            st.write(f"Above selected demo range (maximum: {alert_high} BPM).")
            if st.session_state.monitor_alert_enabled:
                with st.container(key="monitor_range_alert"):
                    st.info(
                        f"Informational demo alert: simulated BPM {tested_bpm} is above "
                        f"the selected maximum of {alert_high} BPM."
                    )
        else:
            st.success("Within selected demo range")
        st.caption(
            "This is a simulated demo value, not a physiological measurement or medical assessment."
        )

    start_col, pause_col, reset_col = st.columns(3)
    start_col.button(
        "Start Monitoring",
        on_click=start_monitoring,
        disabled=st.session_state.monitor_running,
        type="primary",
        width="stretch",
    )
    pause_col.button(
        "Pause Monitoring",
        on_click=pause_monitoring,
        disabled=not st.session_state.monitor_running,
        width="stretch",
    )
    reset_col.button(
        "Reset",
        on_click=reset_monitoring,
        width="stretch",
    )
    is_running = bool(st.session_state.monitor_running)

    @st.fragment(run_every=1.5 if is_running else None)
    def live_monitor_panel() -> None:
        elapsed = monitoring_elapsed()
        bpm = simulated_bpm(
            scenario,
            elapsed,
            center_bpm=float(tested_bpm) if tested_bpm is not None else None,
        )
        waveform = generate_ecg_window(scenario, elapsed, bpm=bpm)
        st.session_state.monitor_bpm = bpm
        now = time.monotonic()
        readings = st.session_state.monitor_readings
        last_recorded = st.session_state.monitor_last_reading_at
        if st.session_state.monitor_running and (
            last_recorded is None or now - last_recorded >= 1.0
        ):
            readings.append(
                {
                    "time": time.strftime("%H:%M:%S"),
                    "elapsed_seconds": elapsed,
                    "simulated_bpm": bpm,
                }
            )
            del readings[:-60]
            st.session_state.monitor_last_reading_at = now

        heart_col, status_col = st.columns([1, 2])
        heart_col.metric("SIMULATED HEART RATE", f"{bpm:.0f} BPM")
        status = (
            "Monitoring"
            if st.session_state.monitor_running
            else ("Paused" if st.session_state.monitor_has_started else "Ready")
        )
        status_col.metric(
            "Monitor status",
            status,
            help=f"Scenario: {scenario}",
        )
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=waveform["time_seconds"],
                y=waveform["voltage"],
                mode="lines",
                line={"color": "#39ff8e", "width": 2},
                name="SIMULATED ECG",
            )
        )
        figure.update_layout(
            title="SIMULATED ECG-STYLE WAVEFORM · RELATIVE UNITS",
            xaxis_title="Time (seconds)",
            yaxis_title="Relative voltage (a.u.)",
            height=330,
            margin={"l": 20, "r": 20, "t": 55, "b": 30},
            paper_bgcolor="#06130d",
            plot_bgcolor="#06130d",
            font={"color": "#dcfce7", "family": "monospace"},
            showlegend=False,
        )
        figure.update_xaxes(showgrid=True, gridcolor="#17442b", zerolinecolor="#17442b")
        figure.update_yaxes(showgrid=True, gridcolor="#17442b", zerolinecolor="#17442b")
        st.plotly_chart(figure, width="stretch", key="simulated_ecg_chart")
        st.caption(
            f"SIMULATED WAVEFORM · {scenario} · elapsed demo time: {elapsed:.1f} s"
        )
        if readings:
            readings_frame = pd.DataFrame(readings)
            trend_col, recent_col = st.columns([3, 2])
            trend = go.Figure(
                go.Scatter(
                    x=readings_frame["time"],
                    y=readings_frame["simulated_bpm"],
                    mode="lines+markers",
                    line={"color": "#21b6d7", "width": 2},
                    marker={"size": 6},
                    name="Simulated BPM",
                )
            )
            trend.update_layout(
                title="SIMULATED HEART-RATE TREND",
                xaxis_title="Demo time",
                yaxis_title="BPM (simulated)",
                height=250,
                margin={"l": 20, "r": 20, "t": 50, "b": 30},
                paper_bgcolor="#06130d",
                plot_bgcolor="#06130d",
                font={"color": "#dcfce7", "family": "monospace"},
                showlegend=False,
            )
            trend.update_xaxes(showgrid=True, gridcolor="#17442b")
            trend.update_yaxes(showgrid=True, gridcolor="#17442b")
            trend_col.plotly_chart(trend, width="stretch", key="simulated_bpm_trend")
            recent_col.markdown("#### Recent simulated readings")
            recent_col.dataframe(
                readings_frame[["time", "simulated_bpm"]]
                .tail(10)
                .rename(columns={"time": "Demo time", "simulated_bpm": "BPM (simulated)"})
                .iloc[::-1],
                width="stretch",
                hide_index=True,
            )
        else:
            st.caption("Start monitoring to build a recent simulated-reading history.")
        if (
            st.session_state.monitor_alert_enabled
            and not (alert_low <= bpm <= alert_high)
            and (is_running or tested_bpm is None)
        ):
            with st.container(key="monitor_live_range_alert"):
                st.warning(
                    f"Informational demo alert: simulated BPM {bpm:.0f} is outside the "
                    f"configured demo range ({alert_low}–{alert_high} BPM)."
                )

    live_monitor_panel()

    st.markdown("### Patient monitoring dashboard")
    st.caption(
        "Select a dataset record or enter features to show patient details and the saved "
        "model output alongside the independent simulated heart-rate display."
    )
    source = st.radio(
        "Patient profile source",
        ["Select existing dataset record", "Enter patient features"],
        horizontal=True,
        key="monitor_profile_source",
    )
    if bundle is None:
        st.warning("The saved stacking model is not available. Run `python train_model.py`.")
        return
    profile: pd.DataFrame | None = None
    try:
        if source == "Select existing dataset record":
            record_number = st.number_input(
                "Dataset record number",
                min_value=1,
                max_value=len(frame),
                value=1,
                step=1,
                key="monitor_record_number",
            )
            st.caption(
                f"Try record numbers 1, 2, or 100. Enter a number from 1 to "
                f"{len(frame):,}; numbering follows the dataset row order."
            )
            if st.button(
                "Load record and predict",
                type="primary",
                key="monitor_record_predict",
            ):
                profile = frame.iloc[[int(record_number) - 1]][
                    bundle["feature_columns"]
                ].copy()
        else:
            profile = render_patient_profile_form(
                frame,
                bundle,
                form_key="monitor_patient_form",
                input_key_prefix="monitor_input_",
                submit_label="Predict from profile",
            )
        if profile is not None:
            st.session_state.heart_model_output = predict_profile(profile, bundle)
            st.session_state.heart_model_profile = profile.iloc[0].to_dict()
    except (ValueError, TypeError, KeyError, IndexError, AttributeError) as error:
        st.error(f"Unable to generate the saved-model output: {error}")

    model_output = st.session_state.get("heart_model_output")
    if model_output is not None:
        profile_values = st.session_state.get("heart_model_profile", {})
        model_label = (
            "Heart stroke predicted"
            if model_output["class"] == 1
            else "No heart stroke predicted"
        )
        detail_col, model_col, monitor_col = st.columns([2, 2, 2])
        with detail_col:
            st.markdown("#### Selected patient profile")
            st.dataframe(
                pd.DataFrame(
                    {
                        "Feature": [
                            FEATURE_LABELS.get(name, name) for name in profile_values
                        ],
                        "Value": [str(value) for value in profile_values.values()],
                    }
                ),
                width="stretch",
                hide_index=True,
            )
        with model_col:
            st.markdown("#### Saved stacking model")
            st.metric("Experimental model output", model_label)
            if model_output["probability"] is not None:
                st.metric("Model probability", f"{model_output['probability']:.1%}")
            st.caption(
                "Experimental prediction from dataset features only; not a medical diagnosis."
            )
        with monitor_col:
            st.markdown("#### Separate simulated monitor")
            monitor_bpm = st.session_state.get("monitor_bpm")
            if monitor_bpm is not None:
                st.metric("SIMULATED BPM", f"{monitor_bpm:.0f}")
                if st.session_state.monitor_alert_enabled and not (
                    alert_low <= monitor_bpm <= alert_high
                ):
                    st.warning(
                        f"Simulated BPM is outside the selected demo range "
                        f"({alert_low}–{alert_high} BPM)."
                    )
                else:
                    st.caption("Within the selected demonstration range.")
            else:
                st.caption("Start the simulation to display a BPM reading.")
            st.caption("Simulated heart rate is not an input to the saved model.")
        if profile_values:
            render_save_history_button(
                profile_values,
                model_output,
                key="save_monitor_prediction_to_history",
            )


def history_class_label(predicted_class: int) -> str:
    return (
        "Class 1 — stroke predicted "
        if predicted_class == 1
        else "Class 0 — no stroke predicted "
    )


def history_session_label(record: dict[str, Any]) -> str:
    patient_name = record.get("patient_name") or "Patient session"
    return f"{patient_name} | {record['created_at']}"


def render_patient_health_history(
    frame: pd.DataFrame, bundle: dict[str, Any] | None
) -> None:
    page_header(
        "Patient Health History",
        "Enter patient details, run an experimental prediction, and save it locally.",
    )
    st.info(
        "This is a local educational history, not a clinical medical record. "
        "Predictions are experimental, and any heart-rate data is simulated."
    )
    if bundle is None:
        st.warning("The saved model is unavailable. Restore it or run `python train_model.py`.")
        return

    st.markdown("### Add a patient session")
    profile = render_patient_profile_form(
        frame,
        bundle,
        form_key="history_patient_form",
        input_key_prefix="history_input_",
        submit_label="Predict Patient",
        patient_name_key="history_patient_name_input",
    )
    if profile is not None:
        try:
            st.session_state.history_current_profile = profile.iloc[0].to_dict()
            st.session_state.history_current_output = predict_profile(profile, bundle)
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            st.error(f"Prediction failed; check the entered values and trained artifact: {error}")
            st.session_state.history_current_profile = None
            st.session_state.history_current_output = None

    current_profile = st.session_state.get("history_current_profile")
    current_output = st.session_state.get("history_current_output")
    if current_profile and current_output is not None:
        st.markdown("#### Experimental prediction result")
        st.write(history_class_label(current_output["class"]))
        if current_output["probability"] is not None:
            st.write(f"Model probability: {current_output['probability']:.1%}")
        st.caption(
            "Experimental dataset-based output only; not a medical diagnosis. "
            "No BPM is included unless separately available as a simulated reading."
        )
        if st.button(
            "Save Patient Session to History",
            key="history_save_patient_session",
            type="primary",
        ):
            try:
                save_history_record(
                    current_profile,
                    current_output,
                    patient_name=st.session_state.get(
                        "history_patient_form_patient_name", ""
                    ),
                )
            except (OSError, sqlite3.Error, TypeError, ValueError) as error:
                st.error(f"Could not save this prediction to local history: {error}")
            else:
                st.session_state.history_current_profile = None
                st.session_state.history_current_output = None
                st.session_state.history_notice = "Patient details saved to history."
                st.rerun()

    try:
        records = list_history_records()
    except (OSError, sqlite3.Error, TypeError, ValueError) as error:
        st.error(f"Could not read local patient history: {error}")
        return

    history_notice = st.session_state.pop("history_notice", None)
    if history_notice:
        st.success(history_notice)
    st.markdown("### Search and filter saved sessions")
    search_col, date_col, class_col = st.columns([2, 2, 2])
    with search_col:
        search_text = st.text_input(
            "Search saved inputs",
            placeholder="Search feature names or values",
            key="history_search",
        ).strip().casefold()
    with date_col:
        date_range = st.date_input(
            "Saved date range",
            value=(date(2000, 1, 1), date.today()),
            key="history_date_range",
        )
    with class_col:
        selected_classes = st.multiselect(
            "Predicted class",
            options=[0, 1],
            default=[0, 1],
            format_func=history_class_label,
            key="history_classes",
        )

    if isinstance(date_range, tuple):
        start_date = date_range[0] if date_range else date(2000, 1, 1)
        end_date = date_range[1] if len(date_range) > 1 else date.today()
    else:
        start_date = end_date = date_range
    visible_records = [
        record
        for record in records
        if start_date <= date.fromisoformat(record["created_at"][:10]) <= end_date
        and record["predicted_class"] in selected_classes
        and (
            not search_text
            or search_text in record.get("patient_name", "").casefold()
            or search_text
            in json.dumps(record["patient_features"], ensure_ascii=False).casefold()
        )
    ]
    if visible_records:
        table_rows = [
            {
                "Patient name": record.get("patient_name") or "Not provided",
                "Saved at": record["created_at"],
                "Predicted class": history_class_label(record["predicted_class"]),
                "Model probability": (
                    f"{record['model_probability']:.1%}"
                    if record["model_probability"] is not None
                    else "Not available"
                ),
            }
            for record in visible_records
        ]
        st.dataframe(pd.DataFrame(table_rows), width="stretch", hide_index=True)
    elif records:
        st.caption("No saved sessions match the current search and filters.")
    else:
        st.caption("No sessions saved yet. Use **Save to History** after a prediction.")

    st.markdown("### Export history")
    if not records:
        st.info(
            "There are no saved records to export yet. Run a prediction, click "
            "**Save to History**, then return here."
        )
    else:
        export_all = st.checkbox(
            "Export all saved history (ignore current filters)",
            key="history_export_all",
        )
        selectable_records = records if export_all else visible_records
        if export_all:
            export_records = records
            st.caption(f"All {len(records)} saved record(s) are selected for export.")
        elif not selectable_records:
            export_records = []
            st.info(
                "No records match the current filters. Adjust the filters or choose "
                "**Export all saved history**."
            )
        else:
            selected_export_ids = st.multiselect(
                "Records to include",
                options=[record["id"] for record in selectable_records],
                format_func=lambda record_id: next(
                    history_session_label(record)
                    for record in selectable_records
                    if record["id"] == record_id
                ),
                placeholder="Click here, then choose one or more saved records",
                help=(
                    "Open this dropdown and click each record you want in the CSV. "
                    "The download button becomes active after you select at least one."
                ),
                key="history_export_selection",
            )
            export_records = [
                record
                for record in selectable_records
                if record["id"] in selected_export_ids
            ]
        export_frame = pd.DataFrame(
            [
                {
                    "patient_name": record.get("patient_name", ""),
                    "saved_at": record["created_at"],
                    "patient_inputs_json": json.dumps(
                        record["patient_features"], ensure_ascii=False
                    ),
                    "predicted_class": record["predicted_class"],
                    "prediction_label": history_class_label(record["predicted_class"]),
                    "model_probability": record["model_probability"],
                    "simulated_bpm": record["simulated_bpm"],
                    "simulated_readings_json": json.dumps(
                        record["simulated_readings"], ensure_ascii=False
                    ),
                }
                for record in export_records
            ]
        )
        st.download_button(
            "Download selected records as CSV",
            data=export_frame.to_csv(index=False).encode("utf-8-sig"),
            file_name="patient_health_history.csv",
            mime="text/csv",
            disabled=not export_records,
        )
        st.caption(
            "The CSV contains feature inputs and simulated readings as JSON columns. "
            "Only the selected records are included unless all-history export is enabled."
        )

    if not visible_records:
        if records:
            st.markdown("### Delete history")
            if st.button("Clear all history", key="history_request_clear"):
                st.session_state.history_clear_pending = True
            if st.session_state.get("history_clear_pending"):
                st.warning("Permanently delete every locally saved history record?")
                confirm_col, cancel_col = st.columns(2)
                if confirm_col.button("Confirm clear all", key="history_confirm_clear"):
                    try:
                        removed = clear_history()
                    except (OSError, sqlite3.Error) as error:
                        st.error(f"Could not clear local history: {error}")
                    else:
                        st.session_state.history_clear_pending = False
                        st.session_state.history_notice = (
                            f"Cleared {removed} local history record(s)."
                        )
                        st.rerun()
                if cancel_col.button("Cancel", key="history_cancel_clear"):
                    st.session_state.history_clear_pending = False
                    st.rerun()
        return

    selected_session_id = st.selectbox(
        "Open patient details",
        options=[record["id"] for record in visible_records],
        format_func=lambda record_id: next(
            history_session_label(record)
            for record in visible_records
            if record["id"] == record_id
        ),
        key="history_open_record",
    )
    selected_record = next(
        record for record in visible_records if record["id"] == selected_session_id
    )
    st.markdown("### Complete session details")
    detail_col, output_col = st.columns(2)
    with detail_col:
        st.markdown("#### Patient inputs")
        st.write(
            f"Patient name: {selected_record.get('patient_name') or 'Not provided'}"
        )
        st.dataframe(
            pd.DataFrame(
                {
                    "Feature": [
                        FEATURE_LABELS.get(name, name)
                        for name in selected_record["patient_features"]
                    ],
                    "Value": [
                        "Not provided" if value is None else str(value)
                        for value in selected_record["patient_features"].values()
                    ],
                }
            ),
            width="stretch",
            hide_index=True,
        )
    with output_col:
        st.markdown("#### Experimental model output")
        st.write(history_class_label(selected_record["predicted_class"]))
        probability = selected_record["model_probability"]
        st.write(
            f"Model probability: {probability:.1%}"
            if probability is not None
            else "Model probability: not available"
        )
        bpm = selected_record["simulated_bpm"]
        st.write(
            f"Simulated BPM: {bpm:.1f}"
            if bpm is not None
            else "Simulated BPM: not available"
        )
        st.caption(f"Saved at {selected_record['created_at']}; local record only.")
    st.markdown("#### Associated simulated heart-rate readings")
    if selected_record["simulated_readings"]:
        st.dataframe(
            pd.DataFrame(selected_record["simulated_readings"]),
            width="stretch",
            hide_index=True,
        )
    else:
        st.caption("No simulated heart-rate readings were available when this was saved.")

    st.markdown("### Delete history")
    if st.button("Delete these patient details", key="history_request_delete"):
        st.session_state.history_delete_pending = selected_session_id
    if st.button("Clear all history", key="history_request_clear"):
        st.session_state.history_clear_pending = True

    if st.session_state.get("history_delete_pending") == selected_session_id:
        st.warning(
            f"Delete {history_session_label(selected_record)}? This cannot be undone."
        )
        confirm_col, cancel_col = st.columns(2)
        if confirm_col.button("Confirm delete", key="history_confirm_delete"):
            try:
                deleted = delete_history_record(selected_session_id)
            except (OSError, sqlite3.Error) as error:
                st.error(f"Could not delete the history record: {error}")
            else:
                st.session_state.history_delete_pending = None
                st.session_state.history_notice = (
                    "Patient session deleted."
                    if deleted
                    else "This patient session was already deleted."
                )
                st.rerun()
        if cancel_col.button("Cancel", key="history_cancel_delete"):
            st.session_state.history_delete_pending = None
            st.rerun()

    if st.session_state.get("history_clear_pending"):
        st.warning("Permanently delete every locally saved history record?")
        confirm_col, cancel_col = st.columns(2)
        if confirm_col.button("Confirm clear all", key="history_confirm_clear"):
            try:
                removed = clear_history()
            except (OSError, sqlite3.Error) as error:
                st.error(f"Could not clear local history: {error}")
            else:
                st.session_state.history_clear_pending = False
                st.session_state.history_notice = (
                    f"Cleared {removed} local history record(s)."
                )
                st.rerun()
        if cancel_col.button("Cancel", key="history_cancel_clear"):
            st.session_state.history_clear_pending = False
            st.rerun()


def render_prediction_results(
    profiles: pd.DataFrame, bundle: dict[str, Any]
) -> pd.DataFrame:
    """Build a downloadable result frame without adding synthetic ground truth."""
    predictions = predict_profiles(profiles, bundle)
    results = profiles.copy()
    results["Model-predicted class"] = [item["class"] for item in predictions]
    if all(item["probability"] is not None for item in predictions):
        results["Model probability"] = [item["probability"] for item in predictions]
    return results


def render_synthetic_and_batch(
    frame: pd.DataFrame, bundle: dict[str, Any] | None
) -> None:
    page_header(
        "Synthetic & Batch Predictions",
        "Generate clearly labeled synthetic patient profiles or run validated batch predictions.",
    )
    st.info(
        "Synthetic profiles and model predictions are demonstrations only. They are not "
        "real patient records or observed outcomes, and predictions are not diagnoses."
    )
    if bundle is None:
        st.warning("The saved model is unavailable. Restore it or run `python train_model.py`.")
        return

    feature_columns = bundle["feature_columns"]
    numeric_columns = bundle["numeric_columns"]
    st.markdown("### Synthetic patient demonstration")
    sample_count = st.selectbox(
        "Number of synthetic records",
        options=[100, 500, 1000],
        format_func=lambda value: f"{value:,}",
        key="synthetic_record_count",
    )
    if st.button("Generate synthetic records", type="primary", key="generate_synthetic"):
        try:
            st.session_state.synthetic_profiles = generate_synthetic_profiles(
                frame,
                feature_columns,
                numeric_columns,
                int(sample_count),
            )
            st.session_state.synthetic_count = int(sample_count)
            st.session_state.pop("synthetic_predictions", None)
        except (ValueError, KeyError, TypeError) as error:
            st.error(f"Unable to generate synthetic records: {error}")

    profiles = st.session_state.get("synthetic_profiles")
    if profiles is not None and st.session_state.get("synthetic_count") == sample_count:
        st.caption(
            f"{len(profiles):,} synthetic profiles created from dataset-observed "
            "categories and numeric ranges. No actual outcomes are included."
        )
        st.dataframe(profiles.head(10), width="stretch", hide_index=True)
        st.download_button(
            "Download synthetic profiles CSV",
            data=profiles.to_csv(index=False).encode("utf-8"),
            file_name=f"heartguard_synthetic_{sample_count}.csv",
            mime="text/csv",
            key="download_synthetic_profiles",
        )
        if st.button("Run saved-model predictions", key="predict_synthetic"):
            try:
                st.session_state.synthetic_predictions = render_prediction_results(
                    profiles, bundle
                )
            except (ValueError, TypeError, KeyError, AttributeError) as error:
                st.error(f"Synthetic batch prediction failed: {error}")
        synthetic_results = st.session_state.get("synthetic_predictions")
        if synthetic_results is not None:
            positive = int((synthetic_results["Model-predicted class"] == 1).sum())
            negative = int((synthetic_results["Model-predicted class"] == 0).sum())
            positive_col, negative_col = st.columns(2)
            positive_col.metric("Model-predicted positive", f"{positive:,}")
            negative_col.metric("Model-predicted negative", f"{negative:,}")
            st.dataframe(synthetic_results.head(20), width="stretch", hide_index=True)
            st.download_button(
                "Download synthetic predictions CSV",
                data=synthetic_results.to_csv(index=False).encode("utf-8"),
                file_name=f"heartguard_synthetic_predictions_{sample_count}.csv",
                mime="text/csv",
                key="download_synthetic_predictions",
            )
            st.caption(
                "Positive/negative counts are model-generated classes, not real outcomes."
            )

    st.markdown("### Optional CSV batch prediction")
    st.write(
        "Upload one row per patient. Required columns match the saved model's ten features; "
        "any `stroke` label column is ignored, and extra columns are not used."
    )
    upload = st.file_uploader(
        "Patient records CSV",
        type=["csv"],
        key="batch_prediction_upload",
    )
    if upload is not None:
        upload_bytes = upload.getvalue()
        upload_token = hashlib.sha256(upload_bytes).hexdigest()
        try:
            uploaded = pd.read_csv(upload)
            profiles = validate_uploaded_profiles(uploaded, frame, feature_columns)
            st.success(
                f"Validated {len(profiles):,} records using the saved model's feature order."
            )
            if st.button("Run batch predictions", type="primary", key="run_batch"):
                st.session_state.batch_predictions = render_prediction_results(
                    profiles, bundle
                )
                st.session_state.batch_prediction_token = upload_token
            batch_results = st.session_state.get("batch_predictions")
            if (
                batch_results is not None
                and st.session_state.get("batch_prediction_token") == upload_token
            ):
                positive = int((batch_results["Model-predicted class"] == 1).sum())
                negative = int((batch_results["Model-predicted class"] == 0).sum())
                positive_col, negative_col = st.columns(2)
                positive_col.metric("Model-predicted positive", f"{positive:,}")
                negative_col.metric("Model-predicted negative", f"{negative:,}")
                st.dataframe(batch_results.head(20), width="stretch", hide_index=True)
                st.download_button(
                    "Download batch prediction results CSV",
                    data=batch_results.to_csv(index=False).encode("utf-8"),
                    file_name="heartguard_batch_predictions.csv",
                    mime="text/csv",
                    key="download_batch_predictions",
                )
                st.caption(
                    "Predicted classes are experimental model outputs, not actual patient outcomes."
                )
        except (pd.errors.ParserError, UnicodeDecodeError, ValueError, OSError) as error:
            st.error(f"Unable to use the uploaded CSV: {error}")


def render_explorer(frame: pd.DataFrame) -> None:
    page_header(
        "Dataset Explorer",
        "Review the supplied observations, schema, missingness, and target balance.",
    )
    a, b, c = st.columns(3)
    a.metric("Records", f"{len(frame):,}")
    b.metric("Columns", str(frame.shape[1]))
    b.metric("Predictors", str(frame.shape[1] - int(TARGET_COLUMN in frame.columns)))
    if TARGET_COLUMN in frame:
        counts = frame[TARGET_COLUMN].value_counts().reindex([0, 1], fill_value=0)
        c.metric("Heart stroke prevalence", f"{counts.get(1, 0) / max(len(frame), 1):.1%}")
    st.markdown("### Dataset preview")
    st.dataframe(frame.head(100), width="stretch", hide_index=True)

    left, right = st.columns(2)
    with left:
        st.markdown("### Heart stroke label distribution")
        if TARGET_COLUMN in frame:
            plot_data = frame[TARGET_COLUMN].value_counts().rename_axis("stroke").reset_index(name="records")
            plot_data["label"] = plot_data["stroke"].map(
                {0: "No heart stroke (0)", 1: "Heart stroke (1)"}
            )
            chart = px.bar(
                plot_data,
                x="label",
                y="records",
                color="label",
                color_discrete_map={
                    "No heart stroke (0)": "#2563eb",
                    "Heart stroke (1)": "#0f9d95",
                },
                text="records",
            )
            chart.update_layout(showlegend=False, xaxis_title="", yaxis_title="Records")
            st.plotly_chart(chart, width="stretch")
    with right:
        st.markdown("### Missing-value summary")
        missing = frame.isna().sum().rename_axis("column").reset_index(name="missing")
        missing = missing[missing["missing"] > 0]
        if missing.empty:
            st.success("No missing values in the supplied CSV.")
        else:
            st.dataframe(missing, width="stretch", hide_index=True)
            st.plotly_chart(
                px.bar(missing, x="column", y="missing", color_discrete_sequence=["#0f9d95"]),
                width="stretch",
            )
    st.markdown("### Summary statistics")
    st.dataframe(frame.describe(include="all").transpose(), width="stretch")


def render_eda(frame: pd.DataFrame) -> None:
    page_header(
        "Exploratory Data Analysis",
        "Explore distributions and observed heart-stroke label patterns in demographic and health columns.",
    )
    if TARGET_COLUMN not in frame:
        st.error(f"Required target column '{TARGET_COLUMN}' is missing.")
        return
    age_col, glucose_col = st.columns(2)
    with age_col:
        st.markdown("### Age distribution")
        st.plotly_chart(
            px.histogram(
                frame,
                x="age",
                color=TARGET_COLUMN,
                nbins=35,
                barmode="overlay",
                opacity=0.72,
                color_discrete_map={0: "#2563eb", 1: "#0f9d95"},
            ),
            width="stretch",
        )
    with glucose_col:
        st.markdown("### Average glucose distribution")
        st.plotly_chart(
            px.histogram(
                frame,
                x="avg_glucose_level",
                color=TARGET_COLUMN,
                nbins=35,
                barmode="overlay",
                opacity=0.72,
                color_discrete_map={0: "#2563eb", 1: "#0f9d95"},
            ),
            width="stretch",
        )
    bmi_col, age_risk_col = st.columns(2)
    with bmi_col:
        st.markdown("### BMI distribution")
        st.plotly_chart(
            px.histogram(
                frame,
                x="bmi",
                color=TARGET_COLUMN,
                nbins=35,
                barmode="overlay",
                opacity=0.72,
                color_discrete_map={0: "#2563eb", 1: "#0f9d95"},
            ),
            width="stretch",
        )
    with age_risk_col:
        st.markdown("### Heart stroke rate by age group")
        age_bins = [0, 30, 45, 60, 75, 120]
        age_labels = ["0–29", "30–44", "45–59", "60–74", "75+"]
        grouped = frame.copy()
        grouped["Age group"] = pd.cut(
            grouped["age"], bins=age_bins, labels=age_labels, right=False, include_lowest=True
        )
        rates = (
            grouped.groupby("Age group", observed=False)[TARGET_COLUMN]
            .agg(["mean", "count"])
            .reset_index()
        )
        rates["stroke_rate"] = rates["mean"] * 100
        st.plotly_chart(
            px.bar(
                rates,
                x="Age group",
                y="stroke_rate",
                color="stroke_rate",
                color_continuous_scale=["#bdece6", "#0f9d95"],
                hover_data=["count"],
                labels={"stroke_rate": "Heart-stroke-labeled records (%)"},
            ),
            width="stretch",
        )
    st.markdown("### Heart stroke distribution across health and smoking groups")
    risk_columns = ["hypertension", "heart_disease", "smoking_status"]
    selected = st.selectbox(
        "Choose a group",
        options=risk_columns,
        format_func=lambda item: item.replace("_", " ").title(),
    )
    rates = (
        frame.groupby(selected, dropna=False)[TARGET_COLUMN]
        .agg(stroke_rate="mean", records="size")
        .reset_index()
    )
    rates["stroke_rate"] *= 100
    st.plotly_chart(
        px.bar(
            rates,
            x=selected,
            y="stroke_rate",
            color=selected,
            hover_data=["records"],
            labels={"stroke_rate": "Heart-stroke-labeled records (%)"},
            color_discrete_sequence=["#2563eb", "#0f9d95", "#52b8c2", "#93a4f4"],
        ),
        width="stretch",
    )
    numeric_frame = frame.select_dtypes(include=np.number)
    st.markdown("### Numeric correlation heatmap")
    if numeric_frame.shape[1] > 1:
        correlation = numeric_frame.corr(numeric_only=True)
        st.plotly_chart(
            px.imshow(
                correlation,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                aspect="auto",
                title="Pearson correlation (numeric columns)",
            ),
            width="stretch",
        )
        st.caption("Correlation describes linear association, not causation.")
    else:
        st.info("Not enough numeric columns for a correlation heatmap.")


def render_comparison(bundle: dict[str, Any] | None) -> None:
    page_header(
        "Model Comparison",
        "All metrics, confusion matrices, and curves below come from the saved stratified holdout evaluation.",
    )
    if bundle is None:
        st.warning("No evaluation results yet. Run `python train_model.py` from the project folder.")
        return
    results = bundle["results"]
    metric_frame = pd.DataFrame(
        {name: value["metrics"] for name, value in results.items()}
    ).transpose()
    st.markdown("### Holdout metrics")
    render_metrics_table(bundle, show_all_models=True)
    st.caption(
        f"Train records: {bundle['summary']['train_records']:,} · "
        f"Test records: {bundle['summary']['test_records']:,} · "
        f"Split: stratified 80/20 · random state: {bundle['summary']['random_state']}"
    )
    st.caption(
        "Decision thresholds are selected to maximize F1 on a separate stratified validation "
        "slice drawn only from the training split."
    )
    st.markdown("### Metric comparison")
    metric_choice = st.selectbox(
        "Metric",
        options=list(metric_frame.columns),
        index=list(metric_frame.columns).index("PR-AUC"),
    )
    comparison = metric_frame[[metric_choice]].reset_index().rename(columns={"index": "Model"})
    comparison.columns = ["Model", metric_choice]
    st.plotly_chart(
        px.bar(
            comparison,
            x="Model",
            y=metric_choice,
            color="Model",
            text=comparison[metric_choice].map(lambda value: f"{value:.3f}"),
            color_discrete_sequence=["#2563eb", "#0f9d95", "#56b8bd", "#5b7ee5"],
        ).update_layout(showlegend=False, yaxis_range=[0, 1]),
        width="stretch",
    )
    chosen_model = st.selectbox("Inspect model curves and confusion matrix", list(results))
    detail = results[chosen_model]
    cm = np.asarray(detail["confusion_matrix"])
    left, right = st.columns(2)
    with left:
        st.markdown("### Confusion matrix")
        st.plotly_chart(
            px.imshow(
                cm,
                text_auto=True,
                x=["Predicted no heart stroke", "Predicted heart stroke"],
                y=["Actual no heart stroke", "Actual heart stroke"],
                color_continuous_scale="Blues",
                labels={"x": "", "y": "", "color": "Records"},
            ),
            width="stretch",
        )
    with right:
        st.markdown("### ROC curve")
        curve = detail["roc_curve"]
        roc_chart = go.Figure()
        roc_chart.add_trace(
            go.Scatter(x=curve["fpr"], y=curve["tpr"], mode="lines", name="ROC")
        )
        roc_chart.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line={"dash": "dash"})
        )
        roc_chart.update_layout(
            xaxis_title="False positive rate",
            yaxis_title="True positive rate",
            title=f"ROC-AUC = {detail['metrics']['ROC-AUC']:.3f}",
        )
        st.plotly_chart(roc_chart, width="stretch")
    st.markdown("### Precision-recall curve")
    pr = detail["pr_curve"]
    pr_chart = go.Figure(
        go.Scatter(x=pr["recall"], y=pr["precision"], mode="lines", name="Precision-recall")
    )
    pr_chart.update_layout(
        xaxis_title="Recall",
        yaxis_title="Precision",
        title=f"Average precision (PR-AUC) = {detail['metrics']['PR-AUC']:.3f}",
    )
    st.plotly_chart(pr_chart, width="stretch")
    st.markdown("### How stacking works")
    st.write(
        "Logistic Regression, Random Forest, and Gradient Boosting each produce "
        "out-of-fold class probabilities during five-fold cross-validation. A second "
        "Logistic Regression learns how to combine those base-model outputs. At prediction "
        "time, the fitted base models feed their probabilities to the final estimator. "
        "Oversampling is applied inside each model's training pipeline, never to the held-out "
        "test data."
    )
    st.caption(
        "The heart stroke class is imbalanced, so accuracy should be interpreted alongside "
        "precision, recall, F1-score, PR-AUC, and false-positive/false-negative counts."
    )


def render_about() -> None:
    page_header(
        "About the Project",
        "A college mini project demonstrating an end-to-end, reproducible classification workflow.",
    )
    st.markdown("### Problem statement")
    st.write(
        "Heart stroke is a serious health condition. This project uses a provided, labeled dataset "
        "to demonstrate how machine-learning classifiers can learn patterns associated with "
        "its recorded heart stroke label. It demonstrates classification and evaluation methods "
        "using the dataset supplied with this project."
    )
    left, right = st.columns(2)
    with left:
        card(
            "Objectives",
            "Inspect the supplied data; preprocess mixed numeric/categorical fields; address "
            "class imbalance without leaking test data; compare three classifiers with a "
            "stacking ensemble; and present transparent metrics and interactive exploration.",
        )
        st.markdown("### Algorithms")
        st.write(
            "- Logistic Regression — linear probabilistic classifier.\n"
            "- Random Forest — bagged decision trees.\n"
            "- Gradient Boosting — sequentially improved decision trees.\n"
            "- Stacking — a Logistic Regression meta-model combines cross-validated "
            "probabilities from the three base learners."
        )
    with right:
        card(
            "Methodology",
            "Stratified 80/20 holdout; train-fold preprocessing; random oversampling within "
            "the training pipelines; five-fold stacking; and holdout Accuracy, Precision, "
            "Recall, F1, ROC-AUC, PR-AUC, confusion matrix, ROC, and precision-recall curves.",
        )
        st.markdown("### Advantages and limitations")
        st.write(
            "**Advantages:** compares diverse learners, handles mixed feature types, makes "
            "the rare class visible through resampling and multiple metrics, and provides an "
            "interactive interface.\n\n"
            "**Limitations:** performance depends on this dataset and split; oversampling "
            "changes training prevalence; and data may contain sampling, measurement, and "
            "population biases."
        )
    st.markdown("### Future scope")
    st.write(
        "Future work could examine external validation, probability calibration, decision "
        "threshold selection, fairness analysis, and privacy."
    )


def main() -> None:
    with st.sidebar:
        st.markdown("## HeartGuard")
        st.markdown("### Choose a page")
        page = st.radio(
            "Navigation",
            PAGE_NAMES,
            index=0,
            label_visibility="collapsed",
        )
        st.markdown("---")
      
    frame = require_data()
    if frame is None:
        st.stop()
    try:
        bundle = load_bundle()
    except (OSError, ValueError, KeyError, EOFError) as error:
        st.error(f"The saved model artifact could not be loaded: {error}")
        bundle = None
    if page == "Home Dashboard":
        render_home(frame, bundle)
    elif page == "Heart Stroke Prediction":
        render_prediction(frame, bundle)
    elif page == "Heart Monitoring":
        render_heart_monitoring(frame, bundle)
    elif page == "Patient Health History":
        render_patient_health_history(frame, bundle)
    elif page == "Exploratory Data Analysis":
        render_eda(frame)
    elif page == "Model Comparison":
        render_comparison(bundle)


if __name__ == "__main__":
    main()
