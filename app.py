"""
Customer Churn Prediction Dashboard
-----------------------------------
Read-only dashboard for the results already stored in Snowflake:
    ML_PROJECT.ML_SCHEMA.CUSTOMER_CHURN_PREDICTIONS

It does NOT train a model, make new predictions, or write to Snowflake.
All numbers shown are calculated from the real rows in that table.
"""

import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import snowflake.connector
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# --------------------------------------------------------------------------
# Page setup & styling
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📉",
    layout="wide",
)

NAVY = "#0B1F4B"
BLUE = "#2563EB"
LIGHT_BLUE = "#93C5FD"
PALE = "#EFF6FF"
GREY = "#64748B"

st.markdown(
    f"""
    <style>
        .stApp {{ background-color: #FFFFFF; }}
        h1, h2, h3, h4 {{ color: {NAVY} !important; font-weight: 700; }}
        .hero {{
            background: linear-gradient(135deg, {NAVY} 0%, {BLUE} 100%);
            padding: 28px 32px; border-radius: 18px; margin-bottom: 24px;
        }}
        .hero h1 {{ color: #FFFFFF !important; margin: 0; font-size: 2rem; }}
        .hero p {{ color: #DBEAFE; margin: 6px 0 0 0; font-size: 1rem; }}
        .kpi {{
            background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 16px;
            padding: 20px 22px; box-shadow: 0 4px 14px rgba(11, 31, 75, 0.07);
        }}
        .kpi .label {{ color: {GREY}; font-size: 0.85rem; font-weight: 600;
                       text-transform: uppercase; letter-spacing: .04em; }}
        .kpi .value {{ color: {NAVY}; font-size: 2.2rem; font-weight: 800; line-height: 1.2; }}
        .kpi .sub {{ color: {GREY}; font-size: 0.8rem; }}
        .metric-card {{
            background: {PALE}; border-radius: 14px; padding: 16px 18px;
            border-left: 5px solid {BLUE};
        }}
        .metric-card .label {{ color: {GREY}; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; }}
        .metric-card .value {{ color: {NAVY}; font-size: 1.7rem; font-weight: 800; }}
        .stTabs [data-baseweb="tab"] {{ font-weight: 600; color: {NAVY}; }}
        section[data-testid="stSidebar"] {{ background-color: {PALE}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

LABELS = {0: "Stay", 1: "Churn"}
COLOR_MAP = {"Churn": NAVY, "Stay": LIGHT_BLUE}


def kpi_card(label: str, value: str, sub: str = "", accent: str = BLUE) -> None:
    st.markdown(
        f"""<div class="kpi" style="border-top:5px solid {accent}">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="sub">{sub}</div></div>""",
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str) -> None:
    st.markdown(
        f"""<div class="metric-card"><div class="label">{label}</div>
            <div class="value">{value}</div></div>""",
        unsafe_allow_html=True,
    )


def style_fig(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color=NAVY, family="Inter, Segoe UI, sans-serif"),
        margin=dict(l=20, r=20, t=50, b=20),
        title_font=dict(size=18, color=NAVY),
    )
    return fig


# --------------------------------------------------------------------------
# Secure Snowflake configuration (Streamlit secrets or environment variables)
# --------------------------------------------------------------------------
def load_config() -> dict:
    """Read settings from st.secrets['snowflake'] first, then SNOWFLAKE_* env vars."""
    cfg = {}
    try:
        if "snowflake" in st.secrets:
            cfg = {k.lower(): v for k, v in dict(st.secrets["snowflake"]).items()}
    except Exception:
        cfg = {}

    for key in [
        "account", "user", "password", "private_key", "private_key_passphrase",
        "role", "warehouse", "database", "schema", "table",
    ]:
        env_val = os.getenv(f"SNOWFLAKE_{key.upper()}")
        if key not in cfg and env_val:
            cfg[key] = env_val

    cfg.setdefault("warehouse", "ML_WH")
    cfg.setdefault("database", "ML_PROJECT")
    cfg.setdefault("schema", "ML_SCHEMA")
    cfg.setdefault("table", "CUSTOMER_CHURN_PREDICTIONS")
    return cfg


def safe_identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]*", str(value)):
        raise ValueError(f"Invalid Snowflake identifier: {value!r}")
    return str(value)


def private_key_bytes(pem: str, passphrase: str | None) -> bytes:
    from cryptography.hazmat.primitives import serialization

    key = serialization.load_pem_private_key(
        pem.encode(), password=passphrase.encode() if passphrase else None
    )
    return key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


@st.cache_data(show_spinner="Loading results from Snowflake…")
def load_predictions() -> tuple[pd.DataFrame, str]:
    """Read the existing results table. Cached until the Refresh button is clicked."""
    cfg = load_config()
    missing = [k for k in ("account", "user") if not cfg.get(k)]
    if missing or not (cfg.get("password") or cfg.get("private_key")):
        raise RuntimeError(
            "Snowflake credentials are missing. Add them to .streamlit/secrets.toml "
            "(or SNOWFLAKE_* environment variables). See README.md."
        )

    params = dict(
        account=cfg["account"],
        user=cfg["user"],
        warehouse=safe_identifier(cfg["warehouse"]),
        database=safe_identifier(cfg["database"]),
        schema=safe_identifier(cfg["schema"]),
    )
    if cfg.get("role"):
        params["role"] = cfg["role"]
    if cfg.get("private_key"):
        params["private_key"] = private_key_bytes(
            cfg["private_key"], cfg.get("private_key_passphrase")
        )
    else:
        params["password"] = cfg["password"]

    table = ".".join(
        safe_identifier(cfg[k]) for k in ("database", "schema", "table")
    )
    conn = snowflake.connector.connect(**params)
    try:
        cur = conn.cursor()
        cur.execute(f"SELECT * FROM {table}")
        df = cur.fetch_pandas_all()
    finally:
        conn.close()

    df.columns = [str(c).upper() for c in df.columns]
    return df, datetime.now().strftime("%d %b %Y, %H:%M:%S")


# --------------------------------------------------------------------------
# Helpers: column detection & label cleaning
# --------------------------------------------------------------------------
ALIASES = {
    "actual": ["ACTUAL_CHURN", "ACTUAL", "TRUE_CHURN", "Y_TRUE", "CHURN"],
    "pred": ["PREDICTED_CHURN", "PREDICTION", "PREDICTED", "Y_PRED", "PRED_CHURN"],
    "prob": ["CHURN_PROBABILITY", "CHURN_PROB", "PROBABILITY", "PROB", "PREDICTED_PROBABILITY"],
}


def detect(cols: list[str], kind: str) -> str | None:
    for name in ALIASES[kind]:
        if name in cols:
            return name
    return None


def to_binary(series: pd.Series) -> pd.Series:
    """Convert 0/1, True/False or Yes/No style values into 0/1 (NaN if unrecognised)."""
    if series.dtype == bool:
        return series.astype(int)
    if pd.api.types.is_numeric_dtype(series):
        out = pd.to_numeric(series, errors="coerce")
        return out.where(out.isin([0, 1]))
    mapping = {
        "1": 1, "yes": 1, "true": 1, "churn": 1, "y": 1,
        "0": 0, "no": 0, "false": 0, "stay": 0, "n": 0,
    }
    return series.astype(str).str.strip().str.lower().map(mapping)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    if st.button("🔄 Refresh Results", use_container_width=True, type="primary"):
        st.cache_data.clear()
        st.rerun()

# --------------------------------------------------------------------------
# Load data
# --------------------------------------------------------------------------
try:
    raw_df, refreshed_at = load_predictions()
except Exception as exc:  # show a friendly error instead of a stack trace
    st.markdown(
        '<div class="hero"><h1>📉 Customer Churn Dashboard</h1>'
        "<p>Could not load results from Snowflake.</p></div>",
        unsafe_allow_html=True,
    )
    st.error(f"**Connection or query problem:** {exc}")
    st.info(
        "Checklist: (1) secrets are filled in, (2) the account identifier is "
        "correct (e.g. `myorg-myaccount`), (3) your role has USAGE on the warehouse, "
        "database and schema and SELECT on the table, (4) no network policy is "
        "blocking this app. Details are in README.md."
    )
    st.stop()

if raw_df.empty:
    st.warning("The predictions table exists but contains no rows.")
    st.stop()

cfg_view = load_config()
cols = list(raw_df.columns)

with st.sidebar:
    st.markdown("---")
    st.caption(
        f"**Source:** `{cfg_view['database']}.{cfg_view['schema']}.{cfg_view['table']}`"
    )
    st.caption(f"**Warehouse:** `{cfg_view['warehouse']}`")
    st.caption(f"**Last refreshed:** {refreshed_at}")
    st.caption(f"**Rows loaded:** {len(raw_df):,}")

    with st.expander("Column mapping (auto-detected)"):
        options = ["— none —"] + cols

        def pick(label: str, kind: str) -> str | None:
            found = detect(cols, kind)
            idx = options.index(found) if found else 0
            choice = st.selectbox(label, options, index=idx)
            return None if choice == "— none —" else choice

        actual_col = pick("Actual churn column", "actual")
        pred_col = pick("Predicted churn column", "pred")
        prob_col = pick("Churn probability column", "prob")

if pred_col is None:
    st.error(
        "No predicted-churn column was found. Columns in the table: "
        + ", ".join(cols)
        + ". Pick the right one in the sidebar under *Column mapping*."
    )
    st.stop()

# --------------------------------------------------------------------------
# Clean working copy
# --------------------------------------------------------------------------
df = raw_df.copy()
df["_PRED"] = to_binary(df[pred_col])
if df["_PRED"].isna().any():
    st.error(
        f"Column `{pred_col}` contains values that are not 0/1 or Yes/No. "
        "Please check the column mapping in the sidebar."
    )
    st.stop()
df["_PRED"] = df["_PRED"].astype(int)

has_actual = False
if actual_col:
    df["_ACTUAL"] = to_binary(df[actual_col])
    has_actual = df["_ACTUAL"].notna().all()
    if has_actual:
        df["_ACTUAL"] = df["_ACTUAL"].astype(int)

has_prob = False
if prob_col:
    df["_PROB"] = pd.to_numeric(df[prob_col], errors="coerce")
    has_prob = df["_PROB"].notna().all() and df["_PROB"].between(0, 1).all()

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown(
    """<div class="hero"><h1>📉 Customer Churn Prediction Dashboard</h1>
    <p>Random Forest results stored in Snowflake — explore predictions, charts and model performance.</p></div>""",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# 1. Summary cards
# --------------------------------------------------------------------------
total = len(df)
n_churn = int((df["_PRED"] == 1).sum())
n_stay = int((df["_PRED"] == 0).sum())
churn_pct = n_churn / total * 100

c1, c2, c3, c4 = st.columns(4)
with c1:
    kpi_card("Customers analyzed", f"{total:,}", "Rows in the results table", NAVY)
with c2:
    kpi_card("Predicted to churn", f"{n_churn:,}", "Model predicts they will leave", BLUE)
with c3:
    kpi_card("Predicted to stay", f"{n_stay:,}", "Model predicts they will remain", LIGHT_BLUE)
with c4:
    kpi_card("Predicted churn rate", f"{churn_pct:.1f}%", "Share of customers predicted to churn", NAVY)

st.write("")

tab_charts, tab_table, tab_perf = st.tabs(
    ["📊 Visualizations", "🔎 Prediction results", "🎯 Model performance"]
)

# --------------------------------------------------------------------------
# 3. Visualizations
# --------------------------------------------------------------------------
with tab_charts:
    left, right = st.columns(2)

    with left:
        pie_df = (
            df["_PRED"].map(LABELS).value_counts().rename_axis("Prediction").reset_index(name="Customers")
        )
        fig = px.pie(
            pie_df, names="Prediction", values="Customers", hole=0.55,
            color="Prediction", color_discrete_map=COLOR_MAP,
            title="Predicted churn vs. predicted stay",
        )
        fig.update_traces(textinfo="percent+value", textfont_size=14)
        st.plotly_chart(style_fig(fig), use_container_width=True)

    with right:
        if has_actual:
            comp = pd.DataFrame({
                "Outcome": ["Stay", "Churn"],
                "Actual": [int((df["_ACTUAL"] == 0).sum()), int((df["_ACTUAL"] == 1).sum())],
                "Predicted": [n_stay, n_churn],
            }).melt(id_vars="Outcome", var_name="Source", value_name="Customers")
            fig = px.bar(
                comp, x="Outcome", y="Customers", color="Source", barmode="group",
                color_discrete_map={"Actual": NAVY, "Predicted": BLUE}, text="Customers",
                title="Actual vs. predicted churn",
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(style_fig(fig), use_container_width=True)
        else:
            st.info("Actual vs. predicted chart unavailable: no valid actual-churn column was found.")

    left, right = st.columns(2)
    with left:
        if has_actual:
            cm = confusion_matrix(df["_ACTUAL"], df["_PRED"], labels=[0, 1])
            fig = px.imshow(
                cm, text_auto=True, color_continuous_scale="Blues", aspect="auto",
                x=["Predicted Stay", "Predicted Churn"],
                y=["Actual Stay", "Actual Churn"],
                title="Confusion matrix",
            )
            fig.update_traces(textfont_size=20)
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(style_fig(fig), use_container_width=True)
        else:
            st.info("Confusion matrix unavailable: no valid actual-churn column was found.")

    with right:
        if has_prob:
            hist_df = pd.DataFrame({
                "Churn probability": df["_PROB"],
                "Prediction": df["_PRED"].map(LABELS),
            })
            fig = px.histogram(
                hist_df, x="Churn probability", color="Prediction", nbins=30,
                barmode="overlay", opacity=0.75, color_discrete_map=COLOR_MAP,
                title="Distribution of churn probability",
            )
            st.plotly_chart(style_fig(fig), use_container_width=True)
        else:
            st.info("Probability distribution unavailable: no valid churn-probability column was found.")

    st.markdown("#### Feature importance")
    st.info(
        "Feature importance is not shown. It needs the trained model (or a saved table of "
        "importance values), and neither is stored in the predictions table. "
        "Showing a chart without that data would mean inventing results."
    )

# --------------------------------------------------------------------------
# 2. Prediction results table
# --------------------------------------------------------------------------
with tab_table:
    f1_, f2_, f3_ = st.columns([2, 2, 3])
    with f1_:
        view = st.radio(
            "Show customers", ["All", "Predicted churn", "Predicted stay"], horizontal=True
        )
    with f2_:
        if has_prob:
            lo, hi = st.slider("Churn probability range", 0.0, 1.0, (0.0, 1.0), step=0.01)
        else:
            lo, hi = 0.0, 1.0
    with f3_:
        query = st.text_input("🔎 Search any column", placeholder="e.g. Month-to-month, Fiber optic, 0.85 …")

    shown = df.copy()
    if view == "Predicted churn":
        shown = shown[shown["_PRED"] == 1]
    elif view == "Predicted stay":
        shown = shown[shown["_PRED"] == 0]
    if has_prob:
        shown = shown[shown["_PROB"].between(lo, hi)]
    if query:
        mask = (
            raw_df.loc[shown.index]
            .astype(str)
            .apply(lambda c: c.str.contains(query, case=False, regex=False))
            .any(axis=1)
        )
        shown = shown[mask]

    # Build a readable display table (original columns, friendly labels)
    display = raw_df.loc[shown.index].copy()
    front = []
    if prob_col and has_prob:
        display[prob_col] = shown["_PROB"]
        front.append(prob_col)
    display[pred_col] = shown["_PRED"].map(LABELS)
    front.append(pred_col)
    if actual_col and has_actual:
        display[actual_col] = shown["_ACTUAL"].map(LABELS)
        front.append(actual_col)
    display = display[front + [c for c in display.columns if c not in front]]

    st.caption(f"Showing **{len(display):,}** of **{total:,}** customers")

    col_cfg = {}
    if prob_col and has_prob:
        col_cfg[prob_col] = st.column_config.ProgressColumn(
            "Churn probability", min_value=0.0, max_value=1.0, format="%.2f"
        )
    st.dataframe(
        display, use_container_width=True, hide_index=True, height=520, column_config=col_cfg
    )

    st.download_button(
        "⬇️ Download filtered results (CSV)",
        data=display.to_csv(index=False).encode("utf-8"),
        file_name="customer_churn_predictions.csv",
        mime="text/csv",
    )

# --------------------------------------------------------------------------
# 4. Model performance (only what can be correctly calculated)
# --------------------------------------------------------------------------
with tab_perf:
    st.markdown("#### Metrics calculated from the stored labels")
    if not has_actual:
        st.warning(
            "Model performance cannot be calculated because no valid actual-churn "
            "column (0/1 or Yes/No) was found in the table."
        )
    else:
        y_true, y_pred = df["_ACTUAL"], df["_PRED"]
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            metric_card("Accuracy", f"{accuracy_score(y_true, y_pred):.3f}")
        with m2:
            metric_card("Precision", f"{precision_score(y_true, y_pred, zero_division=0):.3f}")
        with m3:
            metric_card("Recall", f"{recall_score(y_true, y_pred, zero_division=0):.3f}")
        with m4:
            metric_card("F1-score", f"{f1_score(y_true, y_pred, zero_division=0):.3f}")
        with m5:
            if has_prob and y_true.nunique() == 2:
                metric_card("ROC-AUC", f"{roc_auc_score(y_true, df['_PROB']):.3f}")
            else:
                metric_card("ROC-AUC", "N/A")

        if not has_prob:
            st.warning(
                "ROC-AUC cannot be calculated: it requires a valid churn-probability "
                "column (values between 0 and 1)."
            )
        elif y_true.nunique() < 2:
            st.warning("ROC-AUC cannot be calculated: the actual labels contain only one class.")

        st.caption(
            "Churn is treated as the positive class. These metrics are computed on the rows "
            "in the results table (the held-out test set saved by your notebook), so they "
            "describe performance on data the model did not train on."
        )

# --------------------------------------------------------------------------
st.markdown("---")
st.caption("Customer Churn Prediction · Python · Snowflake · Scikit-learn · Streamlit · Plotly")
