import html
import io
import os
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from google.cloud import storage

try:
    import shap
except Exception:  # pragma: no cover
    shap = None


def render_local_shap(model_obj, explainer_obj, row_df: pd.DataFrame):
    st.markdown('<div class="section-title">Explicabilidad SHAP</div>', unsafe_allow_html=True)

    if shap is None:
        st.info("SHAP no está instalado en este entorno.")
        return

    if explainer_obj is None:
        st.info("No se pudo construir un explainer SHAP para este modelo.")
        return

    try:
        shap_input = row_df.copy()
        shap_input.columns = [
            str(col).strip().replace(" ", "_").replace("<", "lt_").replace(">", "gt_").replace("[", "_").replace("]", "_").replace(",", "_")
            for col in shap_input.columns
        ]

        shap_values = explainer_obj.shap_values(shap_input.to_numpy())
        expected_value = explainer_obj.expected_value

        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

        shap_values = np.asarray(shap_values).reshape(-1)
        feature_names = list(shap_input.columns)
        values = row_df.iloc[0].tolist()

        shap_df = pd.DataFrame(
            {
                "feature": feature_names,
                "value": values,
                "shap_value": shap_values,
                "impact_abs": np.abs(shap_values),
            }
        ).sort_values("impact_abs", ascending=False).head(10)

        c1, c2 = st.columns([1.05, 1])

        with c1:
            fig, ax = plt.subplots(figsize=(8, 4.8))
            colors = ["#d9534f" if v > 0 else "#2ca25f" for v in shap_df["shap_value"]]
            ax.barh(shap_df["feature"][::-1], shap_df["shap_value"][::-1], color=colors[::-1])
            ax.axvline(0, color="#94a3b8", linewidth=1)
            ax.set_title("Top contribuciones SHAP", fontsize=12, fontweight="bold")
            ax.set_xlabel("Impacto sobre la predicción")
            ax.set_ylabel("")
            ax.grid(axis="x", alpha=0.2)
            st.pyplot(fig, clear_figure=True)

        with c2:
            st.dataframe(
                shap_df[["feature", "value", "shap_value"]],
                use_container_width=True,
                hide_index=True,
            )
            st.caption("Valores positivos empujan la predicción hacia la clase de riesgo.")

        st.caption(f"Expected value: {expected_value}")

    except Exception as exc:
        st.warning(f"No se pudo calcular SHAP para esta predicción: {exc}")


def render_label_meaning_card():
    st.markdown(
        """
        <div class="surface">
            <div class="surface-title">Significado de la etiqueta</div>
            <div class="surface-subtitle">
                <strong>0</strong> = bajo riesgo / no moroso
                <br>
                <strong>1</strong> = alto riesgo / moroso
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_label_meaning_card_inline():
    render_label_meaning_card()

import matplotlib.pyplot as plt


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Credit Risk XGB",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# CONFIG
# =========================================================

DEFAULT_BUCKET_NAME = os.getenv("BUCKET_NAME", "am-up-01-credit-risk2")
ARTIFACT_PREFIX = os.getenv("ARTIFACT_PREFIX", "xgb_optuna_v2")
MODEL_BLOB = os.getenv("MODEL_BLOB", f"{ARTIFACT_PREFIX}/xgboost_credit_risk.pkl")
COLUMNS_BLOB = os.getenv("COLUMNS_BLOB", f"{ARTIFACT_PREFIX}/model_columns.pkl")
METADATA_BLOB = os.getenv("METADATA_BLOB", f"{ARTIFACT_PREFIX}/model_metadata.pkl")

DEFAULT_PREDICTION_VALUES = {
    "loan_amnt": 12000.0,
    "int_rate": 12.7400,
    "installment": 375.4300,
    "annual_inc": 65000.0,
    "dti": 17.5700,
    "delinq_2yrs": 0.0,
    "inq_last_6mths": 0.0,
    "open_acc": 11.0,
    "pub_rec": 0.0,
    "revol_bal": 11140.0,
    "revol_util": 52.3000,
    "total_acc": 23.0,
    "mort_acc": 1.0,
    "pub_rec_bankruptcies": 0.0,
    "tax_liens": 0.0,
    "term_ 60 months": 0.0,
    "grade_B": 0.0,
    "grade_C": 0.0,
    "grade_D": 0.0,
    "grade_E": 0.0,
    "grade_F": 0.0,
    "grade_G": 0.0,
    "sub_grade_A2": 0.0,
    "sub_grade_A3": 0.0,
    "sub_grade_A4": 0.0,
    "sub_grade_A5": 0.0,
    "sub_grade_B1": 0.0,
    "sub_grade_B2": 0.0,
    "sub_grade_B3": 0.0,
    "sub_grade_B4": 0.0,
    "sub_grade_B5": 0.0,
    "sub_grade_C1": 0.0,
    "sub_grade_C2": 0.0,
    "sub_grade_C3": 0.0,
    "sub_grade_C4": 0.0,
    "sub_grade_C5": 0.0,
    "sub_grade_D1": 0.0,
    "sub_grade_D2": 0.0,
    "sub_grade_D3": 0.0,
    "sub_grade_D4": 0.0,
    "sub_grade_D5": 0.0,
    "sub_grade_E1": 0.0,
    "sub_grade_E2": 0.0,
    "sub_grade_E3": 0.0,
    "sub_grade_E4": 0.0,
    "sub_grade_E5": 0.0,
    "sub_grade_F1": 0.0,
    "sub_grade_F2": 0.0,
    "sub_grade_F3": 0.0,
    "sub_grade_F4": 0.0,
    "sub_grade_F5": 0.0,
    "sub_grade_G1": 0.0,
    "sub_grade_G2": 0.0,
    "sub_grade_G3": 0.0,
    "sub_grade_G4": 0.0,
    "sub_grade_G5": 0.0,
    "emp_length_10+ years": 1.0,
    "emp_length_2 years": 0.0,
    "emp_length_3 years": 0.0,
    "emp_length_4 years": 0.0,
    "emp_length_5 years": 0.0,
    "emp_length_6 years": 0.0,
    "emp_length_7 years": 0.0,
    "emp_length_8 years": 0.0,
    "emp_length_9 years": 0.0,
    "emp_length__lt_ 1 year": 0.0,
    "emp_length_< 1 year": 0.0,
    "home_ownership_MORTGAGE": 0.0,
    "home_ownership_NONE": 0.0,
    "home_ownership_OTHER": 0.0,
    "home_ownership_OWN": 0.0,
    "home_ownership_RENT": 0.0,
    "verification_status_Source Verified": 0.0,
    "verification_status_Verified": 0.0,
    "application_type_Joint App": 0.0,
    "purpose_credit_card": 0.0,
    "purpose_debt_consolidation": 0.0,
    "purpose_educational": 0.0,
    "purpose_home_improvement": 0.0,
    "purpose_house": 0.0,
    "purpose_major_purchase": 0.0,
    "purpose_medical": 0.0,
    "purpose_moving": 0.0,
    "purpose_other": 0.0,
    "purpose_renewable_energy": 0.0,
    "purpose_small_business": 0.0,
    "purpose_vacation": 0.0,
    "purpose_wedding": 0.0,
}

FEATURE_NAME_ALIASES = {
    "emp_length_< 1 year": ["emp_length__lt_ 1 year"],
    "emp_length__lt_ 1 year": ["emp_length_< 1 year"],
}


# =========================================================
# STYLE
# =========================================================

st.markdown(
    """
    <style>
    :root {
        --bg: #f4f7fb;
        --surface: #ffffff;
        --surface-soft: #f8fbff;
        --primary: #123a63;
        --accent: #2563eb;
        --accent-2: #0f766e;
        --text: #132238;
        --muted: #5e7188;
        --border: #d7e1ee;
        --good: #127b4f;
        --good-bg: #e8f7ef;
        --warn: #b45309;
        --warn-bg: #fff1dc;
        --bad: #b42318;
        --bad-bg: #fdecec;
    }

    .stApp {
        background: linear-gradient(180deg, #f8fbff 0%, var(--bg) 100%);
        color: var(--text);
    }

    body, .stApp, .block-container, p, div, span, label, input, textarea, select {
        color: var(--text);
    }

    .block-container {
        max-width: 1280px;
        padding-top: 1rem;
        padding-bottom: 2rem;
    }

    .hero {
        background: linear-gradient(135deg, #0f3b66 0%, #1756a9 58%, #2563eb 100%);
        color: #ffffff;
        border-radius: 22px;
        padding: 1.5rem 1.6rem;
        box-shadow: 0 18px 40px rgba(15, 59, 102, 0.20);
        margin-bottom: 1rem;
    }

    .hero-badge {
        display: inline-block;
        padding: 0.3rem 0.7rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.14);
        border: 1px solid rgba(255,255,255,0.22);
        font-size: 0.82rem;
        font-weight: 800;
        margin-bottom: 0.75rem;
    }

    .hero h1 {
        margin: 0;
        font-size: 2.1rem;
        font-weight: 900;
        color: #ffffff !important;
    }

    .hero p {
        margin: 0.55rem 0 0;
        color: rgba(255,255,255,0.88) !important;
        line-height: 1.6;
        max-width: 980px;
    }

    .surface {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        box-shadow: 0 10px 26px rgba(16, 42, 67, 0.06);
        margin-bottom: 1rem;
    }

    .surface-title {
        font-size: 1.02rem;
        font-weight: 900;
        color: var(--primary) !important;
        margin-bottom: 0.25rem;
    }

    .surface-subtitle {
        color: #516276 !important;
        line-height: 1.55;
        font-size: 0.94rem;
    }

    .section-title {
        color: var(--text) !important;
        font-size: 1.05rem;
        font-weight: 900;
        margin: 0.9rem 0 0.6rem;
    }

    .info-box {
        background: #eef4fb;
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 0.85rem 0.95rem;
        color: var(--text) !important;
    }

    .artifact {
        background: #f3f7fc;
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 0.8rem 0.85rem;
        font-family: monospace;
        word-break: break-all;
        color: #1e3a5f !important;
    }

    .metric-card {
        background: #ffffff;
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 0.9rem 1rem;
        box-shadow: 0 8px 22px rgba(16, 42, 67, 0.05);
    }

    .metric-label {
        color: #516276 !important;
        font-weight: 800;
        font-size: 0.85rem;
    }

    .metric-value {
        color: var(--text) !important;
        font-weight: 900;
        font-size: 1.45rem;
        margin-top: 0.2rem;
    }

    .pill {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        border-radius: 999px;
        padding: 0.32rem 0.7rem;
        font-size: 0.82rem;
        font-weight: 800;
        border: 1px solid transparent;
    }

    .pill-good {
        background: var(--good-bg);
        color: var(--good) !important;
        border-color: #bfead1;
    }

    .pill-warn {
        background: var(--warn-bg);
        color: var(--warn) !important;
        border-color: #f2ce88;
    }

    .pill-bad {
        background: var(--bad-bg);
        color: var(--bad) !important;
        border-color: #f5b5b0;
    }

    .stButton > button {
        border-radius: 999px;
        font-weight: 800;
        border: 1px solid #8fb0d6;
        background: #ffffff;
        color: var(--primary) !important;
        min-height: 2.75rem;
        box-shadow: none;
    }

    .stButton > button[kind="primary"] {
        background: #1f5fe0;
        color: #ffffff !important;
        border-color: #1f5fe0;
        box-shadow: 0 8px 18px rgba(31, 95, 224, 0.18);
    }

    .stButton > button[kind="secondary"] {
        background: #ffffff;
        color: var(--primary) !important;
        border-color: #7f9cc4;
    }

    .stButton > button[kind="primary"]:hover {
        background: #174cc4;
        color: #ffffff !important;
        border-color: #174cc4;
    }

    .stButton > button[kind="secondary"]:hover {
        background: #eef4fb;
        color: var(--primary) !important;
        border-color: #6d89b0;
    }

    .stButton > button:disabled {
        background: #edf2f7 !important;
        color: #8aa0b8 !important;
        border-color: #d2dcea !important;
        opacity: 1;
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid var(--border);
        border-radius: 16px;
        box-shadow: 0 8px 22px rgba(16, 42, 67, 0.05);
    }

    h1, h2, h3, h4, h5, h6 {
        color: var(--text) !important;
    }

    div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] * {
        color: var(--text) !important;
    }

    div[data-testid="stMetricLabel"], div[data-testid="stMetricLabel"] * {
        color: #516276 !important;
    }

    input, textarea, select {
        background: #ffffff !important;
        border: 1px solid #b9c8da !important;
        color: var(--text) !important;
    }

    input::placeholder, textarea::placeholder {
        color: #6f8198 !important;
        opacity: 1;
    }

    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid var(--border);
    }

    section[data-testid="stSidebar"] * {
        color: var(--text) !important;
    }

    div[data-testid="stRadio"] label,
    div[data-testid="stRadio"] label * {
        color: var(--text) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HELPERS
# =========================================================


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def sanitize_shap_name(name: str) -> str:
    return (
        str(name)
        .strip()
        .replace(" ", "_")
        .replace("<", "lt_")
        .replace(">", "gt_")
        .replace("[", "_")
        .replace("]", "_")
        .replace(",", "_")
    )


@st.cache_resource
def get_storage_client():
    return storage.Client()


def get_bucket(bucket_name: str):
    return get_storage_client().bucket(bucket_name)


def load_bytes_from_gcs(bucket_name: str, blob_name: str) -> bytes | None:
    bucket = get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    if not blob.exists():
        return None
    return blob.download_as_bytes()


def load_pickle_from_gcs(bucket_name: str, blob_name: str):
    data = load_bytes_from_gcs(bucket_name, blob_name)
    if data is None:
        return None
    return joblib.load(io.BytesIO(data))


def normalize_columns(columns_obj, model=None) -> list[str]:
    if columns_obj is None:
        if model is not None and hasattr(model, "feature_names_in_"):
            return list(model.feature_names_in_)
        return []

    if isinstance(columns_obj, pd.DataFrame):
        for candidate in ("feature", "column", "columns", "name"):
            if candidate in columns_obj.columns:
                return columns_obj[candidate].dropna().astype(str).tolist()
        return columns_obj.iloc[:, 0].dropna().astype(str).tolist()

    if isinstance(columns_obj, pd.Series):
        return columns_obj.dropna().astype(str).tolist()

    if isinstance(columns_obj, dict):
        for candidate in ("columns", "feature_columns", "features", "names"):
            if candidate in columns_obj:
                value = columns_obj[candidate]
                if isinstance(value, (list, tuple, set, np.ndarray, pd.Index)):
                    return [str(v) for v in value]
        return [str(v) for v in columns_obj.keys()]

    if isinstance(columns_obj, (list, tuple, set, np.ndarray, pd.Index)):
        return [str(v) for v in columns_obj]

    return [str(columns_obj)]


def extract_metadata(metadata_obj) -> dict[str, Any]:
    if metadata_obj is None:
        return {}
    if isinstance(metadata_obj, dict):
        return metadata_obj
    if isinstance(metadata_obj, pd.DataFrame) and not metadata_obj.empty:
        return metadata_obj.iloc[0].to_dict()
    return {"raw_metadata": metadata_obj}


def infer_feature_kind(name: str, metadata: dict[str, Any]) -> str:
    schema = metadata.get("schema") or metadata.get("feature_schema") or {}
    if isinstance(schema, dict) and name in schema:
        spec = schema[name]
        if isinstance(spec, dict):
            return str(spec.get("type", "string")).lower()

    feature_types = metadata.get("feature_types") or metadata.get("types") or {}
    if isinstance(feature_types, dict) and name in feature_types:
        return str(feature_types[name]).lower()

    lowered = name.lower()
    bool_tokens = ("is_", "has_", "flag", "binary", "default", "late", "active", "opened", "insured")
    cat_tokens = (
        "grade", "purpose", "state", "country", "gender", "marital", "education",
        "job", "occupation", "segment", "channel", "product", "status", "type",
        "ownership", "home", "region", "category", "risk_band", "bucket", "term"
    )
    if lowered.startswith(bool_tokens) or any(token in lowered for token in bool_tokens):
        return "bool"
    if any(token in lowered for token in cat_tokens):
        return "categorical"
    return "numeric"


def feature_default(name: str, metadata: dict[str, Any], kind: str):
    schema = metadata.get("schema") or metadata.get("feature_schema") or {}
    if isinstance(schema, dict) and name in schema and isinstance(schema[name], dict):
        spec = schema[name]
        if "default" in spec:
            return spec["default"]
        if kind == "bool":
            return bool(spec.get("default", 0))
        if kind == "numeric":
            return float(spec.get("default", 0.0))
        if "categories" in spec and spec["categories"]:
            return spec["categories"][0]

    if kind == "bool":
        return 0
    if kind == "numeric":
        return 0.0
    return "Unknown"


def feature_categories(name: str, metadata: dict[str, Any]) -> list[str]:
    schema = metadata.get("schema") or metadata.get("feature_schema") or {}
    if isinstance(schema, dict) and name in schema and isinstance(schema[name], dict):
        cats = schema[name].get("categories") or schema[name].get("values")
        if isinstance(cats, (list, tuple, set, np.ndarray, pd.Index)):
            return [str(v) for v in cats]
    cats_map = metadata.get("categories") or metadata.get("feature_categories") or {}
    if isinstance(cats_map, dict) and name in cats_map:
        cats = cats_map[name]
        if isinstance(cats, (list, tuple, set, np.ndarray, pd.Index)):
            return [str(v) for v in cats]
    return []


def feature_bounds(name: str, metadata: dict[str, Any]):
    schema = metadata.get("schema") or metadata.get("feature_schema") or {}
    if isinstance(schema, dict) and name in schema and isinstance(schema[name], dict):
        spec = schema[name]
        return spec.get("min"), spec.get("max"), spec.get("step")
    return None, None, None


def build_input_widgets(columns: list[str], metadata: dict[str, Any], defaults: dict[str, Any] | None = None) -> dict[str, Any]:
    inputs: dict[str, Any] = {}
    defaults = defaults or {}
    cols = st.columns(3)

    for idx, feature in enumerate(columns):
        kind = infer_feature_kind(feature, metadata)
        default = defaults.get(feature, feature_default(feature, metadata, kind))
        category_values = feature_categories(feature, metadata)
        min_value, max_value, step = feature_bounds(feature, metadata)
        target_col = cols[idx % 3]

        with target_col:
            if category_values:
                inputs[feature] = st.selectbox(
                    feature,
                    category_values,
                    index=0 if str(default) not in category_values else category_values.index(str(default)),
                    key=f"feat_{feature}",
                )
            elif feature in defaults and isinstance(default, (int, float, np.integer, np.floating)):
                num_default = float(default)
                inputs[feature] = st.number_input(
                    feature,
                    value=num_default,
                    min_value=0.0 if num_default in (0.0, 1.0) else (float(min_value) if min_value is not None else None),
                    max_value=1.0 if num_default in (0.0, 1.0) else (float(max_value) if max_value is not None else None),
                    step=1.0 if num_default in (0.0, 1.0) else (float(step) if step is not None else 1.0),
                    key=f"feat_{feature}",
                )
            elif kind == "bool":
                inputs[feature] = st.selectbox(
                    feature,
                    [0, 1],
                    index=int(bool(default)),
                    key=f"feat_{feature}",
                )
            elif kind == "numeric":
                kwargs = {"value": float(default) if default not in ("", None) else 0.0}
                if min_value is not None:
                    kwargs["min_value"] = float(min_value)
                if max_value is not None:
                    kwargs["max_value"] = float(max_value)
                if step is not None:
                    kwargs["step"] = float(step)
                inputs[feature] = st.number_input(feature, key=f"feat_{feature}", **kwargs)
            else:
                inputs[feature] = st.text_input(feature, value="Unknown" if default in ("", None) else str(default), key=f"feat_{feature}")

    return inputs


def build_default_row(columns: list[str], metadata: dict[str, Any]) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for feature in columns:
        resolved_feature = resolve_column_name(feature, columns)

        if feature in DEFAULT_PREDICTION_VALUES:
            defaults[resolved_feature] = DEFAULT_PREDICTION_VALUES[feature]
            continue
        if resolved_feature in DEFAULT_PREDICTION_VALUES:
            defaults[resolved_feature] = DEFAULT_PREDICTION_VALUES[resolved_feature]
            continue

        kind = infer_feature_kind(resolved_feature, metadata)
        category_values = feature_categories(resolved_feature, metadata)
        default = feature_default(resolved_feature, metadata, kind)

        if category_values:
            defaults[resolved_feature] = default if default in category_values else category_values[0]
        elif kind == "bool":
            defaults[resolved_feature] = int(bool(default))
        elif kind == "numeric":
            defaults[resolved_feature] = float(default) if default not in ("", None) else 0.0
        else:
            defaults[resolved_feature] = "Unknown" if default in ("", None) else str(default)
    return defaults


def summarize_column_kinds(columns: list[str], metadata: dict[str, Any]) -> dict[str, int]:
    summary = {"numeric": 0, "categorical": 0, "bool": 0, "text": 0}
    for feature in columns:
        kind = infer_feature_kind(feature, metadata)
        if kind in summary:
            summary[kind] += 1
        else:
            summary["text"] += 1
    return summary


def resolve_column_name(name: str, columns: list[str]) -> str:
    if name in columns:
        return name
    for alias in FEATURE_NAME_ALIASES.get(name, []):
        if alias in columns:
            return alias
    return name


def get_model_feature_names(model_obj) -> list[str]:
    if model_obj is None:
        return []

    if hasattr(model_obj, "feature_names_in_"):
        try:
            return [str(v) for v in model_obj.feature_names_in_]
        except Exception:
            pass

    if hasattr(model_obj, "get_booster"):
        try:
            booster = model_obj.get_booster()
            if hasattr(booster, "feature_names") and booster.feature_names:
                return [str(v) for v in booster.feature_names]
        except Exception:
            pass

    if hasattr(model_obj, "get_xgb_params"):
        try:
            booster = model_obj.get_booster()
            if hasattr(booster, "feature_names") and booster.feature_names:
                return [str(v) for v in booster.feature_names]
        except Exception:
            pass

    return []


def standardize_to_model_schema(df: pd.DataFrame, model_obj, defaults: dict[str, Any]) -> pd.DataFrame:
    expected = get_model_feature_names(model_obj)
    if not expected:
        expected = list(df.columns)

    aligned = df.copy()
    rename_map = {}
    available = set(aligned.columns)

    for target in expected:
        if target in available:
            continue
        for alias in FEATURE_NAME_ALIASES.get(target, []):
            if alias in available:
                rename_map[alias] = target
                available.remove(alias)
                available.add(target)
                break

    if rename_map:
        aligned = aligned.rename(columns=rename_map)

    for target in expected:
        if target not in aligned.columns:
            aligned[target] = defaults.get(target, 0)

    aligned = aligned[expected]
    return aligned


def seed_prediction_widget_state(columns: list[str], defaults: dict[str, Any]):
    version_key = "_prediction_defaults_version"
    current_version = 2
    if st.session_state.get(version_key) != current_version:
        for feature in columns:
            widget_key = f"feat_{feature}"
            if feature in defaults:
                st.session_state[widget_key] = defaults[feature]
        st.session_state[version_key] = current_version


def build_dataframe_from_inputs(inputs: dict[str, Any], columns: list[str]) -> pd.DataFrame:
    row = {}
    for feature in columns:
        value = inputs.get(feature)
        if isinstance(value, (np.integer, np.floating)):
            value = value.item()
        if isinstance(value, bool):
            value = int(value)
        row[feature] = value
    return pd.DataFrame([row], columns=columns)


def align_batch_dataframe(df: pd.DataFrame, columns: list[str], defaults: dict[str, Any]) -> pd.DataFrame:
    aligned = df.copy()

    for feature in columns:
        if feature not in aligned.columns:
            aligned[feature] = defaults.get(feature, 0)

    aligned = aligned[columns]

    for feature in columns:
        default_value = defaults.get(feature, 0)
        if isinstance(default_value, str):
            aligned[feature] = aligned[feature].fillna(default_value).astype(str)
        else:
            aligned[feature] = pd.to_numeric(aligned[feature], errors="coerce")
            aligned[feature] = aligned[feature].fillna(float(default_value) if default_value not in ("", None) else 0.0)

    return aligned


def predict_batch(model_obj, df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_to_model_schema(df, model_obj, default_row)
    output = df.copy()

    if hasattr(model_obj, "predict_proba"):
        proba = model_obj.predict_proba(df)
        classes = list(getattr(model_obj, "classes_", [0, 1]))
        pos_index = classes.index(1) if 1 in classes else 1 if proba.shape[1] > 1 else 0
        preds = model_obj.predict(df)
        output["prediction"] = preds
        output["probability"] = proba[:, pos_index] if proba.ndim == 2 else proba
        output["risk_label"] = np.where(
            output["probability"] >= 0.75,
            "Riesgo alto",
            np.where(output["probability"] >= 0.45, "Riesgo medio", "Riesgo bajo"),
        )
        return output

    output["prediction"] = model_obj.predict(df)
    return output


def predict(model, df: pd.DataFrame):
    df = standardize_to_model_schema(df, model, default_row)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df)[0]
        classes = list(getattr(model, "classes_", [0, 1]))
        if len(proba) > 1:
            pos_index = classes.index(1) if 1 in classes else int(np.argmax(proba))
            probability = float(proba[pos_index])
        else:
            probability = float(proba[0])
        y_pred = int(model.predict(df)[0])
        return y_pred, probability, proba

    y_pred = int(model.predict(df)[0])
    return y_pred, None, None


def risk_label(probability: float | None, y_pred: int) -> tuple[str, str]:
    if probability is None:
        return ("Resultado", "pill-warn")
    if probability >= 0.75 or y_pred == 1:
        return ("Riesgo alto", "pill-bad")
    if probability >= 0.45:
        return ("Riesgo medio", "pill-warn")
    return ("Riesgo bajo", "pill-good")


def extract_metrics(metadata: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "metrics",
        "evaluation_metrics",
        "model_metrics",
        "train_metrics",
        "test_metrics",
        "validation_metrics",
    ]
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, dict) and value:
            return value
    flat = {}
    for key in ("auc", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "threshold"):
        if key in metadata:
            flat[key] = metadata[key]
    return flat


def extract_probability_cutoff(metadata: dict[str, Any]) -> float:
    for key in ("threshold", "cutoff", "decision_threshold", "best_threshold"):
        if key in metadata:
            try:
                return float(metadata[key])
            except Exception:
                pass
    return 0.5


def load_shap_explainer(model):
    if shap is None:
        return None
    try:
        return shap.TreeExplainer(model)
    except Exception:
        return None


# =========================================================
# LOAD ARTIFACTS
# =========================================================


@st.cache_resource
def load_artifacts(bucket_name: str, model_blob: str, columns_blob: str, metadata_blob: str):
    model = load_pickle_from_gcs(bucket_name, model_blob)
    columns_obj = load_pickle_from_gcs(bucket_name, columns_blob)
    metadata_obj = load_pickle_from_gcs(bucket_name, metadata_blob)
    columns = normalize_columns(columns_obj, model)
    metadata = extract_metadata(metadata_obj)
    return {
        "model": model,
        "columns": columns,
        "metadata": metadata,
        "explainer": load_shap_explainer(model) if model is not None else None,
    }


# =========================================================
# SESSION STATE
# =========================================================

if "bucket_name" not in st.session_state:
    st.session_state.bucket_name = DEFAULT_BUCKET_NAME

if "last_input" not in st.session_state:
    st.session_state.last_input = None

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None


artifacts = load_artifacts(
    st.session_state.bucket_name,
    MODEL_BLOB,
    COLUMNS_BLOB,
    METADATA_BLOB,
)

model = artifacts["model"]
columns = artifacts["columns"]
metadata = artifacts["metadata"]
explainer = artifacts["explainer"]
metrics = extract_metrics(metadata)
threshold = extract_probability_cutoff(metadata)
column_summary = summarize_column_kinds(columns, metadata) if columns else {"numeric": 0, "categorical": 0, "bool": 0, "text": 0}
default_row = build_default_row(columns, metadata) if columns else {}
if columns:
    seed_prediction_widget_state(columns, default_row)


# =========================================================
# HIGH-CONTRAST OVERRIDES
# =========================================================

st.markdown(
    """
    <style>
    .stJson, div[data-testid="stJson"], pre, pre code, code {
        background: #0f172a !important;
        color: #e5eefc !important;
    }
    .stJson, div[data-testid="stJson"] {
        border: 1px solid #24364f !important;
        border-radius: 12px !important;
    }
    .stJson *, div[data-testid="stJson"] * {
        color: #e5eefc !important;
    }
    .stDownloadButton > button, div[data-testid="stDownloadButton"] button {
        background: #0f172a !important;
        border: 1px solid #334155 !important;
        color: #ffffff !important;
    }
    .stDownloadButton > button *, div[data-testid="stDownloadButton"] button * {
        color: #ffffff !important;
    }
    .stDownloadButton > button:hover, div[data-testid="stDownloadButton"] button:hover {
        background: #1e293b !important;
        border-color: #475569 !important;
    }
    .stButton > button, .stButton > button * {
        color: inherit !important;
    }
    .stCaption, .stCaption *, caption, small {
        color: #516276 !important;
    }
    div[data-testid="stAlert"], div[data-testid="stAlert"] * {
        color: #0f172a !important;
    }
    .stDataFrame, div[data-testid="stDataFrame"], div[data-testid="stTable"] {
        color: var(--text) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# UI
# =========================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">Streamlit · Google Cloud Run · GCS</div>
        <h1>Credit Risk XGB</h1>
        <h2>Hecho por: Emanuel Landero, Daniel Sandoval y José María Sánchez</h2>
        <p>
            Panel para desplegar un modelo de riesgo crediticio con predicción, métricas del bundle
            y explicabilidad local con SHAP.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if model is None:
    st.error(
        f"No se pudo cargar el modelo desde gs://{st.session_state.bucket_name}/{MODEL_BLOB}. "
        "Revisa el nombre del PKL y la ruta del bucket."
    )

st.markdown(
    """
    <div class="info-box">
        La app lee tres artefactos desde GCS: el modelo, la lista de columnas esperadas y la metadata
        con métricas/configuración.
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.radio(
    "Navegación",
    ["Inicio", "Predicción", "Procesamiento por lotes", "Explicabilidad", "Configuración técnica"],
    horizontal=True,
    label_visibility="collapsed",
)


def predict_batch_ordered(model_obj, df: pd.DataFrame, defaults: dict[str, Any]) -> pd.DataFrame:
    df = standardize_to_model_schema(df, model_obj, defaults)
    out = df.copy()
    out["label"] = model_obj.predict(df)
    if hasattr(model_obj, "predict_proba"):
        proba = model_obj.predict_proba(df)
        classes = list(getattr(model_obj, "classes_", [0, 1]))
        pos_index = classes.index(1) if 1 in classes and proba.ndim == 2 and len(classes) > 1 else 1 if proba.ndim == 2 and proba.shape[1] > 1 else 0
        out["probability"] = proba[:, pos_index] if proba.ndim == 2 else proba
    first_cols = ["label"]
    if "probability" in out.columns:
        first_cols.append("probability")
    remaining_cols = [c for c in out.columns if c not in first_cols]
    return out[first_cols + remaining_cols]

if page == "Procesamiento por lotes":
    st.markdown('<div class="section-title">Procesamiento por lotes</div>', unsafe_allow_html=True)
    st.caption("Carga un CSV con uno o más registros. Si entra un solo registro, además se muestra SHAP.")

    if not columns:
        st.warning("No se encontraron columnas en model_columns.pkl. Revisa el artefacto.")
    else:
        uploaded = st.file_uploader("Sube un CSV para clasificar", type=["csv"])
        if uploaded is not None:
            try:
                raw_df = pd.read_csv(uploaded, sep=None, engine="python")
                batch_df = raw_df.copy()
                for col in columns:
                    if col not in batch_df.columns:
                        batch_df[col] = default_row.get(col, 0)
                batch_df = batch_df[columns]
                preds_df = predict_batch_ordered(model, batch_df, default_row)
                st.success(f"Se procesaron {len(preds_df)} registros.")
                st.dataframe(preds_df, use_container_width=True)
                st.download_button(
                    "Descargar predicciones CSV",
                    data=preds_df.to_csv(index=False).encode("utf-8"),
                    file_name="batch_predictions.csv",
                    mime="text/csv",
                )
                if len(preds_df) == 1:
                    render_local_shap(model, explainer, batch_df.iloc[[0]])
                else:
                    st.info("Como el archivo contiene más de un registro, solo se muestran las predicciones.")
            except Exception as exc:
                st.error(f"No se pudo procesar el archivo CSV: {exc}")
    st.stop()


def render_metrics(metrics_dict: dict[str, Any]):
    if not metrics_dict:
        st.info("No se encontraron métricas en model_metadata.pkl.")
        return

    cols = st.columns(min(4, max(1, len(metrics_dict))))
    for idx, (name, value) in enumerate(metrics_dict.items()):
        with cols[idx % len(cols)]:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-label">{esc(name)}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{esc(value)}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)


def render_local_shap(model_obj, explainer_obj, row_df: pd.DataFrame):
    st.markdown('<div class="section-title">Explicabilidad SHAP</div>', unsafe_allow_html=True)

    if shap is None:
        st.info("SHAP no está instalado en este entorno.")
        return
    if explainer_obj is None:
        st.info("No se pudo construir un explainer SHAP para este modelo.")
        return

    try:
        shap_input = row_df.copy()
        shap_input.columns = [sanitize_shap_name(col) for col in shap_input.columns]

        # SHAP/XGBoost falla con nombres que contienen '<', '[', ']' o ','.
        # Para explicabilidad usamos una versión interna sanitizada y mantenemos
        # los nombres originales solo para la vista.
        shap_values = explainer_obj.shap_values(shap_input.to_numpy())
        expected = explainer_obj.expected_value

        if isinstance(shap_values, list):
            if len(shap_values) > 1:
                shap_values = shap_values[1]
            else:
                shap_values = shap_values[0]
        shap_values = np.asarray(shap_values).reshape(-1)
        feature_names = [sanitize_shap_name(col) for col in row_df.columns]
        values = row_df.iloc[0].tolist()

        shap_df = pd.DataFrame(
            {
                "feature": feature_names,
                "value": values,
                "shap_value": shap_values,
                "impact_abs": np.abs(shap_values),
            }
        ).sort_values("impact_abs", ascending=False).head(10)

        c1, c2 = st.columns([1.1, 1])

        with c1:
            fig, ax = plt.subplots(figsize=(8, 4.8))
            colors = ["#d9534f" if v > 0 else "#2ca25f" for v in shap_df["shap_value"]]
            ax.barh(
                shap_df["feature"][::-1],
                shap_df["shap_value"][::-1],
                color=colors[::-1],
            )
            ax.axvline(0, color="#94a3b8", linewidth=1)
            ax.set_title("Top contribuciones SHAP", fontsize=12, fontweight="bold")
            ax.set_xlabel("Impacto sobre la predicción")
            ax.set_ylabel("")
            ax.grid(axis="x", alpha=0.2)
            st.pyplot(fig, clear_figure=True)

        with c2:
            st.dataframe(
                shap_df[["feature", "value", "shap_value"]],
                use_container_width=True,
                hide_index=True,
            )
            st.caption("Valores positivos empujan la predicción hacia la clase de riesgo; negativos la reducen.")

        st.caption(f"Expected value: {expected}")

    except Exception as exc:
        st.warning(f"No se pudo calcular SHAP para esta predicción: {exc}")


if page == "Inicio":
    st.markdown('<div class="section-title">Resumen</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Modelo cargado", "Sí" if model is not None else "No")
    c2.metric("Columnas esperadas", len(columns) if columns else 0)
    c3.metric("Métricas detectadas", len(metrics))
    c4.metric("Threshold", f"{threshold:.2f}")

    left, right = st.columns([1.15, 1])

    with left:
        st.markdown('<div class="surface">', unsafe_allow_html=True)
        st.markdown('<div class="surface-title">Estado operativo</div>', unsafe_allow_html=True)
        if model is not None:
            st.markdown('<div class="pill pill-good">Modelo disponible</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="pill pill-bad">Modelo no cargado</div>', unsafe_allow_html=True)
        st.write("Bucket:", st.session_state.bucket_name)
        st.write("Modelo:", MODEL_BLOB)
        st.write("Columnas:", COLUMNS_BLOB)
        st.write("Metadata:", METADATA_BLOB)
        st.write("Distribución de columnas:", column_summary)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="surface">', unsafe_allow_html=True)
        st.markdown('<div class="surface-title">Métricas del bundle</div>', unsafe_allow_html=True)
        render_metrics(metrics)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Columnas esperadas</div>', unsafe_allow_html=True)
    st.caption("La lista completa sale de `model_columns.pkl`; aquí ves el total y una vista previa para validar el esquema.")
    if columns:
        preview_df = pd.DataFrame(
            {
                "feature": columns[:20],
                "default": [default_row.get(col) for col in columns[:20]],
                "kind": [infer_feature_kind(col, metadata) for col in columns[:20]],
            }
        )
        st.dataframe(preview_df, use_container_width=True, hide_index=True)
        with st.expander(f"Ver todas las columnas ({len(columns)})"):
            st.write(columns)
    else:
        st.info("No se pudieron leer columnas esperadas desde el artefacto.")

elif page == "Predicción":
    st.markdown('<div class="section-title">Predicción</div>', unsafe_allow_html=True)
    st.caption("Puedes ejecutar una predicción rápida con el perfil base o ajustar solo las variables que necesites.")

    if not columns:
        st.warning("No se encontraron columnas en model_columns.pkl. Revisa el artefacto.")
    else:
        top_left, top_right = st.columns([1, 1])
        with top_left:
            run_default = st.button("Ejecutar con perfil base", type="primary")
        with top_right:
            st.caption(f"El bundle espera {len(columns)} columnas. La mayoría ya viene con valores por defecto.")

        with st.form("prediction_form"):
            inputs = build_input_widgets(columns, metadata, default_row)
            submitted = st.form_submit_button("Ejecutar con los valores del formulario", type="primary")

        if run_default:
            inputs = default_row.copy()
            row_df = build_dataframe_from_inputs(inputs, columns)
            st.session_state.last_input = row_df.copy()

            try:
                y_pred, probability, raw_proba = predict(model, row_df)
                label, pill_class = risk_label(probability, y_pred)
                st.session_state.last_prediction = {
                    "y_pred": y_pred,
                    "probability": probability,
                    "raw_proba": raw_proba.tolist() if isinstance(raw_proba, np.ndarray) else raw_proba,
                }

                p1, p2 = st.columns([0.7, 1.3])
                with p1:
                    st.markdown(f'<div class="pill {pill_class}">{label}</div>', unsafe_allow_html=True)
                    if probability is not None:
                        st.metric("Probabilidad de riesgo", f"{probability:.3f}")
                    st.metric("Clase predicha", y_pred)
                    st.metric("Threshold", f"{threshold:.2f}")

                with p2:
                    st.markdown('<div class="surface">', unsafe_allow_html=True)
                    st.markdown('<div class="surface-title">Entrada enviada</div>', unsafe_allow_html=True)
                    st.dataframe(row_df, use_container_width=True, hide_index=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                render_local_shap(model, explainer, row_df)

            except Exception as exc:
                st.error(f"No se pudo ejecutar la predicción con el perfil base: {exc}")

        if submitted:
            row_df = build_dataframe_from_inputs(inputs, columns)
            st.session_state.last_input = row_df.copy()

            try:
                y_pred, probability, raw_proba = predict(model, row_df)
                label, pill_class = risk_label(probability, y_pred)
                st.session_state.last_prediction = {
                    "y_pred": y_pred,
                    "probability": probability,
                    "raw_proba": raw_proba.tolist() if isinstance(raw_proba, np.ndarray) else raw_proba,
                }

                p1, p2 = st.columns([0.7, 1.3])
                with p1:
                    st.markdown(f'<div class="pill {pill_class}">{label}</div>', unsafe_allow_html=True)
                    if probability is not None:
                        st.metric("Probabilidad de riesgo", f"{probability:.3f}")
                    st.metric("Clase predicha", y_pred)
                    st.metric("Threshold", f"{threshold:.2f}")

                with p2:
                    st.markdown('<div class="surface">', unsafe_allow_html=True)
                    st.markdown('<div class="surface-title">Entrada enviada</div>', unsafe_allow_html=True)
                    st.dataframe(row_df, use_container_width=True, hide_index=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                render_local_shap(model, explainer, row_df)

            except Exception as exc:
                st.error(f"No se pudo ejecutar la predicción: {exc}")

elif page == "Explicabilidad":
    st.markdown('<div class="section-title">Explicabilidad</div>', unsafe_allow_html=True)
    st.caption("Esta vista usa la última predicción para mostrar contribuciones SHAP.")

    if st.session_state.last_input is None:
        st.info("Primero ejecuta una predicción para generar una explicación local.")
    else:
        render_local_shap(model, explainer, st.session_state.last_input)

elif page == "Procesamiento por lotes":
    st.markdown('<div class="section-title">Procesamiento por lotes</div>', unsafe_allow_html=True)
    st.caption("Carga un CSV con uno o más registros. Si entra un solo registro, además se muestra SHAP.")

    if not columns:
        st.warning("No se encontraron columnas en model_columns.pkl. Revisa el artefacto.")
    else:
        uploaded = st.file_uploader("Sube un CSV para clasificar", type=["csv"])

        st.markdown("#### Perfil base usado para completar columnas faltantes")
        st.dataframe(
            pd.DataFrame(
                {
                    "feature": list(default_row.keys())[:20],
                    "default": list(default_row.values())[:20],
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        if uploaded is not None:
            try:
                raw_df = pd.read_csv(uploaded)
                batch_df = align_batch_dataframe(raw_df, columns, default_row)
                preds_df = predict_batch(model, batch_df)

                st.success(f"Se procesaron {len(preds_df)} registros.")
                st.dataframe(preds_df, use_container_width=True)

                csv_bytes = preds_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Descargar predicciones CSV",
                    data=csv_bytes,
                    file_name="batch_predictions.csv",
                    mime="text/csv",
                )

                if len(preds_df) == 1:
                    st.markdown('<div class="section-title">Explicación SHAP del único registro</div>', unsafe_allow_html=True)
                    render_local_shap(model, explainer, batch_df.iloc[[0]])
                else:
                    st.info("Como el archivo contiene más de un registro, solo se muestran las predicciones.")

            except Exception as exc:
                st.error(f"No se pudo procesar el archivo CSV: {exc}")

elif page == "Configuración técnica":
    st.markdown('<div class="section-title">Configuración técnica</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="surface">', unsafe_allow_html=True)
        st.markdown('<div class="surface-title">Artefactos GCS</div>', unsafe_allow_html=True)
        st.write("Bucket fijo")
        render_artifact = lambda x: st.markdown(
            f'<div class="artifact">{esc(x)}</div>', unsafe_allow_html=True
        )
        render_artifact(st.session_state.bucket_name)
        st.write("Modelo")
        render_artifact(f"gs://{st.session_state.bucket_name}/{MODEL_BLOB}")
        st.write("Columnas")
        render_artifact(f"gs://{st.session_state.bucket_name}/{COLUMNS_BLOB}")
        st.write("Metadata")
        render_artifact(f"gs://{st.session_state.bucket_name}/{METADATA_BLOB}")
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="surface">', unsafe_allow_html=True)
        st.markdown('<div class="surface-title">Resumen del bundle</div>', unsafe_allow_html=True)
        st.write("Cantidad de columnas:", len(columns))
        st.write("Tipo de modelo:", type(model).__name__ if model is not None else "No cargado")
        st.write("SHAP disponible:", "Sí" if shap is not None else "No")
        st.write("Threshold:", f"{threshold:.2f}")
        st.write("Metadata keys:", ", ".join(sorted(metadata.keys())) if metadata else "Sin metadata")
        st.markdown('</div>', unsafe_allow_html=True)


st.markdown(
    """
    <div style="margin-top: 1.8rem; padding-top: 1rem; border-top: 1px solid #d7e1ee; color: #5e7188;">
        Credit Risk XGB · Streamlit · Google Cloud Run · Google Cloud Storage
    </div>
    """,
    unsafe_allow_html=True,
)

if page in ("Predicción", "Procesamiento por lotes"):
    render_label_meaning_card_inline()
if "page" in globals() and page in ("Predicción", "Procesamiento por lotes"):
    render_label_meaning_card_inline()
