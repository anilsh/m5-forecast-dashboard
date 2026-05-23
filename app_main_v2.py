"""
M5 Forecasting Dashboard — Streamlit App
=========================================
Sections:
  1. Data Overview (EDA)
  2. Feature Engineering Insights
  3. Model Evaluation (Metrics)
  4. Product-Level Forecast Explorer

Run:
    streamlit run app.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import plotly.io as pio

pio.templates.default = "plotly_dark"
#pio.templates.default = "plotly_white"

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="M5 Forecast Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Dark sidebar */
[data-testid="stSidebar"] {
    background: #0f1117;
    border-right: 1px solid #1e2130;
}
[data-testid="stSidebar"] * {
    color: #e0e4ef !important;
}
[data-testid="stSidebar"] .stRadio label {
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
    letter-spacing: 0.04em;
}

/* Main background */
.main { background: #f7f8fc; }

/* KPI card */
.kpi-card {
    background: white;
    border-radius: 12px;
    padding: 22px 28px;
    border-left: 4px solid #4f6ef7;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    margin-bottom: 8px;
}
.kpi-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: #8b92a5;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.kpi-value {
    font-size: 2rem;
    font-weight: 600;
    color: #1a1d2e;
    line-height: 1.1;
}
.kpi-sub {
    font-size: 0.78rem;
    color: #8b92a5;
    margin-top: 4px;
}

/* Section headers */
.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.1rem;
    font-weight: 700;
    color: #1a1d2e;
    letter-spacing: -0.01em;
    margin-bottom: 4px;
    padding-bottom: 8px;
    border-bottom: 2px solid #4f6ef7;
    display: inline-block;
}
.section-sub {
    font-size: 0.87rem;
    color: #6b7280;
    margin-bottom: 20px;
}

/* Metric badge */
.badge {
    display: inline-block;
    background: #eef0ff;
    color: #4f6ef7;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    padding: 3px 10px;
    border-radius: 20px;
    margin-right: 6px;
    letter-spacing: 0.04em;
}
.badge-green { background: #e6f7ee; color: #1e7e4a; }
.badge-red   { background: #fde8e8; color: #c0392b; }
.badge-amber { background: #fef3e2; color: #b45309; }

/* Divider */
.divider {
    border: none;
    border-top: 1px solid #e5e7eb;
    margin: 24px 0;
}

/* Page header */
.page-header {
    background: linear-gradient(135deg, #1a1d2e 0%, #2d3561 100%);
    color: white;
    padding: 32px 40px;
    border-radius: 16px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.page-header::after {
    content: '';
    position: absolute;
    right: -60px; top: -60px;
    width: 240px; height: 240px;
    border-radius: 50%;
    background: rgba(79,110,247,0.18);
}
.page-header h1 {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    margin: 0 0 6px 0;
    color: white;
}
.page-header p {
    font-size: 0.9rem;
    color: #a5aed4;
    margin: 0;
}

/* Table styling */
.styled-table {
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA LOADING  (cached)
# ─────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

@st.cache_data(show_spinner="Loading data…")
def load_data():
    """
    Load pre-exported CSVs produced by data_prep.py.
    Returns a dict of DataFrames.
    """
    def fp(name):
        return os.path.join(DATA_DIR, name)

    val_df        = pd.read_csv(fp("val_df.csv"), parse_dates=["date"])
    train_df      = pd.read_csv(fp("train_df_small_1.csv"), parse_dates=["date"])
    #train_df = pd.read_parquet('train_df_small.parquet' )
    agg_sales     = pd.read_csv(fp("agg_sales_val.csv"))
    cat_sales     = pd.read_csv(fp("cat_sales_val.csv"))
    importance_df = pd.read_csv(fp("feature_importance.csv"))
    zero_ratio    = pd.read_csv(fp("zero_ratio.csv"))

    return {
        "val_df":        val_df,
        "train_df":      train_df,
        "agg_sales":     agg_sales,
        "cat_sales":     cat_sales,
        "importance_df": importance_df,
        "zero_ratio":    zero_ratio,
    }


@st.cache_data(show_spinner="Computing metrics…")
def compute_metrics(_val_df, _train_df):
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    val_df   = _val_df
    train_df = _train_df

    rmse = float(np.sqrt(mean_squared_error(val_df["sales"], val_df["pred"])))
    mae  = float(mean_absolute_error(val_df["sales"], val_df["pred"]))
    wmape = float(
        np.sum(np.abs(val_df["sales"] - val_df["pred"])) /
        np.sum(np.abs(val_df["sales"]))
    )
    bias = float(np.mean(val_df["pred"] - val_df["sales"]))

    # RMSSE per item
    def rmsse_fn(train_s, actual, forecast):
        train_s  = np.asarray(train_s, dtype=float)
        actual   = np.asarray(actual,  dtype=float)
        forecast = np.asarray(forecast, dtype=float)
        denom = np.mean(np.diff(train_s) ** 2)
        if denom == 0:
            return np.nan
        return float(np.sqrt(np.mean((actual - forecast) ** 2) / denom))

    train_lookup = train_df.groupby("id")["sales"].apply(list).to_dict()
    item_scores  = []
    for item_id, grp in val_df.groupby("id"):
        ts = train_lookup.get(item_id, [])
        if len(ts) < 2:
            continue
        s = rmsse_fn(ts, grp["sales"].values, grp["pred"].values)
        if not np.isnan(s):
            item_scores.append(s)

    item_scores_s = pd.Series(item_scores)
    overall_rmsse = float(item_scores_s.mean())

    # Hierarchical breakdowns
    def hier_eval(df, col):
        g = df.groupby(col).agg(sales=("sales","sum"), pred=("pred","sum")).reset_index()
        g["error"]     = g["pred"] - g["sales"]
        g["abs_error"] = g["error"].abs()
        g["wmape"]     = (g["abs_error"] / g["sales"].replace(0, np.nan)) * 100
        g["bias_pct"]  = (g["error"] / g["sales"].replace(0, np.nan)) * 100
        return g.round(2)

    state_eval    = hier_eval(val_df, "state_id")
    store_eval    = hier_eval(val_df, "store_id")
    cat_eval      = hier_eval(val_df, "cat_id")

    # Zero demand
    zero_df = val_df[val_df["sales"] == 0]
    fp_rate = float(np.mean(zero_df["pred"] > 0.5)) * 100
    fp_avg  = float(zero_df["pred"].mean())

    # Event analysis
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    event_rows = []
    for flag, grp in val_df.groupby("has_event"):
        er = float(np.sqrt(mean_squared_error(grp["sales"], grp["pred"])))
        em = float(mean_absolute_error(grp["sales"], grp["pred"]))
        eb = float(np.mean(grp["pred"] - grp["sales"]))
        # rmsse for event
        rs = []
        for iid, ig in grp.groupby("id"):
            ts = train_lookup.get(iid, [])
            if len(ts) < 2: continue
            sc = rmsse_fn(ts, ig["sales"].values, ig["pred"].values)
            if not np.isnan(sc): rs.append(sc)
        event_rows.append({
            "Event Day": "Yes" if flag == 1 else "No",
            "RMSE": round(er, 4),
            "MAE":  round(em, 4),
            "Bias": round(eb, 4),
            "RMSSE": round(float(np.mean(rs)), 4) if rs else np.nan,
        })
    event_eval = pd.DataFrame(event_rows)

    # Error distribution
    errors = (val_df["pred"] - val_df["sales"]).values

    # SKU classification
    sku_stats = (
        val_df.groupby("id")
        .agg(avg_sales=("sales","mean"), zero_ratio=("sales", lambda x: (x==0).mean()))
        .reset_index()
    )
    high_vol_id = sku_stats.sort_values("avg_sales", ascending=False).iloc[0]["id"]
    sparse_id   = sku_stats.sort_values("zero_ratio", ascending=False).iloc[0]["id"]

    return {
        "rmse": rmse, "mae": mae, "wmape": wmape,
        "bias": bias, "overall_rmsse": overall_rmsse,
        "item_scores": item_scores_s,
        "state_eval": state_eval, "store_eval": store_eval, "cat_eval": cat_eval,
        "fp_rate": fp_rate, "fp_avg": fp_avg,
        "event_eval": event_eval,
        "errors": errors,
        "sku_stats": sku_stats,
        "high_vol_id": high_vol_id,
        "sparse_id": sparse_id,
    }


# ─────────────────────────────────────────────
# PLOTLY THEME HELPER
# ─────────────────────────────────────────────
#PALETTE = ["#4f6ef7", "#f7914f", "#4fc9a4", "#f74f6e", "#a44ff7", "#f7d44f"]
PALETTE = [
    "#1f77b4",   # blue
    "#ff7f0e",   # orange
    "#2ca02c",   # green
    "#d62728",   # red
    "#9467bd",   # purple
    "#8c564b"    # brown
]
def apply_theme(fig, title="", height=380):
    fig.update_layout(
        title=dict(text=title, font=dict(family="Space Mono", size=13, color="#111827")),
        paper_bgcolor="white",
        #plot_bgcolor="#f7f8fc",
        plot_bgcolor="#eef2f7",
        #font=dict(family="DM Sans", size=12, color="#3d4459"),
        font=dict(family="DM Sans", size=13, color="#111827"),
        height=height,
        margin=dict(l=20, r=20, t=50 if title else 20, b=20),
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#e5e7eb",
            borderwidth=1,
            font=dict(size=11),
        ),
        #xaxis=dict(gridcolor="#eef0f5", zeroline=False),
        #yaxis=dict(gridcolor="#eef0f5", zeroline=False),
        xaxis=dict(gridcolor="#d6dbe6", zeroline=False),
        yaxis=dict(gridcolor="#d6dbe6", zeroline=False),
    )
    return fig


# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:18px 0 8px 0;'>
        <div style='font-family:Space Mono,monospace;font-size:1.05rem;font-weight:700;
                    color:#4f6ef7;letter-spacing:-0.01em;'>M5 Forecast</div>
        <div style='font-size:0.75rem;color:#6b7aaa;margin-top:2px;'>Dashboard v1.0</div>
    </div>
    <hr style='border-color:#1e2130;margin:10px 0 18px 0;'/>
    """, unsafe_allow_html=True)

    section = st.radio(
        "Navigate",
        [
            "🏠  Overview",
            "🔬  EDA",
            "⚙️  Features",
            "📊  Model Evaluation",
            "🔍  Forecast Explorer",
        ],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:#1e2130;margin:24px 0 14px 0;'/>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.72rem;color:#4a5070;line-height:1.7;'>
        <b style='color:#6b7aaa;'>Model</b><br>LightGBM · Tweedie<br><br>
        <b style='color:#6b7aaa;'>Horizon</b><br>28 days<br><br>
        <b style='color:#6b7aaa;'>Dataset</b><br>M5 Walmart (evaluation)<br>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
try:
    data = load_data()
except FileNotFoundError as e:
    st.error(
        f"**Data files not found.**\n\n"
        f"Run `python data_prep.py` from your Colab notebook first to export the required CSVs into the `data/` folder.\n\n"
        f"Missing: `{e.filename}`"
    )
    st.stop()

val_df        = data["val_df"]
train_df      = data["train_df"]
agg_sales     = data["agg_sales"]
cat_sales     = data["cat_sales"]
importance_df = data["importance_df"]
zero_ratio    = data["zero_ratio"]

metrics = compute_metrics(val_df, train_df)


# ═══════════════════════════════════════════════════════════════
# SECTION: OVERVIEW
# ═══════════════════════════════════════════════════════════════
if section == "🏠  Overview":
    st.markdown("""
    <div class='page-header'>
        <h1>📦 M5 Forecasting Dashboard</h1>
        <p>Walmart item-level demand forecasting · LightGBM + Tweedie · 28-day horizon</p>
    </div>
    """, unsafe_allow_html=True)

    # Top KPI row
    k1, k2, k3, k4 = st.columns(4)

    def kpi(col, label, value, sub, accent="#4f6ef7"):
        col.markdown(f"""
        <div class='kpi-card' style='border-left-color:{accent};'>
            <div class='kpi-label'>{label}</div>
            <div class='kpi-value'>{value}</div>
            <div class='kpi-sub'>{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    kpi(k1, "RMSSE", f"{metrics['overall_rmsse']:.4f}", "< 1.0 beats naïve baseline ✓", "#4f6ef7")
    kpi(k2, "WMAPE", f"{metrics['wmape']*100:.1f}%",   "Weighted abs % error", "#f7914f")
    kpi(k3, "MAE",   f"{metrics['mae']:.4f}",          "Mean absolute error (units)", "#4fc9a4")
    kpi(k4, "RMSE",  f"{metrics['rmse']:.4f}",         "Root mean squared error", "#f74f6e")

    st.markdown("<hr class='divider'/>", unsafe_allow_html=True)

    # Dataset stats
    st.markdown("<div class='section-title'>Dataset at a Glance</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>M5 competition — Walmart sales evaluation split</div>", unsafe_allow_html=True)

    d1, d2, d3, d4, d5 = st.columns(5)
    stats = [
        (d1, "Stores",      val_df["store_id"].nunique()  if "store_id" in val_df else "—"),
        (d2, "States",      val_df["state_id"].nunique()  if "state_id" in val_df else "—"),
        (d3, "Categories",  val_df["cat_id"].nunique()    if "cat_id"   in val_df else "—"),
        (d4, "Unique SKUs", val_df["id"].nunique()),
        (d5, "Val Rows",    f"{len(val_df):,}"),
    ]
    for col, lbl, val in stats:
        col.metric(lbl, val)

    st.markdown("<hr class='divider'/>", unsafe_allow_html=True)

    # Quick chart — aggregate sales
    st.markdown("<div class='section-title'>Total Sales Over Time</div>", unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=agg_sales["day_index"], y=agg_sales["sales"],
        mode="lines", line=dict(color="#4f6ef7", width=1.5),
        fill="tozeroy", fillcolor="rgba(79,110,247,0.08)",
        name="Total Sales",
    ))
    apply_theme(fig, height=300)
    fig.update_xaxes(title_text="Day Index")
    fig.update_yaxes(title_text="Units Sold")
    st.plotly_chart(fig, use_container_width=True)

    # Metric explanation
    st.markdown("<hr class='divider'/>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Metric Glossary</div>", unsafe_allow_html=True)
    mc1, mc2 = st.columns(2)
    mc1.markdown("""
    | Metric | Meaning |
    |--------|---------|
    | **RMSSE** | M5 official metric — scale-free, compares vs naïve baseline |
    | **RMSE**  | Penalises large errors; same unit as sales |
    """)
    mc2.markdown("""
    | Metric | Meaning |
    |--------|---------|
    | **MAE**   | Average absolute error per day/item |
    | **WMAPE** | Revenue-weighted % error; business-friendly |
    """)


# ═══════════════════════════════════════════════════════════════
# SECTION: EDA
# ═══════════════════════════════════════════════════════════════
elif section == "🔬  EDA":
    st.markdown("<div class='page-header'><h1>🔬 Exploratory Data Analysis</h1><p>Understanding demand structure before modelling</p></div>", unsafe_allow_html=True)

    # Zero demand distribution
    st.markdown("<div class='section-title'>Zero-Demand Ratio Distribution</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Fraction of days with zero sales per SKU — reveals intermittency</div>", unsafe_allow_html=True)

    fig = px.histogram(
        zero_ratio, x="zero_ratio", nbins=60,
        color_discrete_sequence=["#4f6ef7"],
    )
    fig.update_traces(marker_line_width=0)
    apply_theme(fig, height=320)
    fig.update_xaxes(title_text="Fraction of Zero-Sales Days")
    fig.update_yaxes(title_text="Number of SKUs")

    avg_zr = float(zero_ratio["zero_ratio"].mean())
    fig.add_vline(x=avg_zr, line_dash="dash", line_color="#f74f6e",
                  annotation_text=f"Mean: {avg_zr:.2f}", annotation_position="top right")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    col1.metric("Average Zero-Demand Ratio", f"{avg_zr:.1%}")
    col2.metric("Highly Sparse SKUs (>70% zeros)",
                f"{(zero_ratio['zero_ratio']>0.7).sum():,}")

    st.markdown("<hr class='divider'/>", unsafe_allow_html=True)

    # Category level sales
    st.markdown("<div class='section-title'>Category-Level Sales Trends</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Aggregate daily sales per product category</div>", unsafe_allow_html=True)

    fig2 = go.Figure()
    for i, col in enumerate([c for c in cat_sales.columns if c != "day_index"]):
        fig2.add_trace(go.Scatter(
            x=cat_sales["day_index"], y=cat_sales[col],
            mode="lines", name=col,
            line=dict(color=PALETTE[i % len(PALETTE)], width=1.8),
        ))
    apply_theme(fig2, height=360)
    fig2.update_xaxes(title_text="Day Index")
    fig2.update_yaxes(title_text="Units Sold")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<hr class='divider'/>", unsafe_allow_html=True)

    # Demand by store
    if "store_id" in val_df.columns:
        st.markdown("<div class='section-title'>Sales Distribution by Store</div>", unsafe_allow_html=True)
        store_totals = val_df.groupby("store_id")["sales"].sum().reset_index().sort_values("sales", ascending=False)
        fig3 = px.bar(store_totals, x="store_id", y="sales",
                      color="sales", color_continuous_scale="Blues",
                      text_auto=".2s")
        fig3.update_traces(marker_line_width=0)
        apply_theme(fig3, height=320)
        fig3.update_layout(coloraxis_showscale=False)
        fig3.update_xaxes(title_text="Store")
        fig3.update_yaxes(title_text="Total Sales (validation period)")
        st.plotly_chart(fig3, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# SECTION: FEATURES
# ═══════════════════════════════════════════════════════════════
elif section == "⚙️  Features":
    st.markdown("<div class='page-header'><h1>⚙️ Feature Engineering</h1><p>What the model learned — importance, price signals, event flags</p></div>", unsafe_allow_html=True)

    # Feature importance
    st.markdown("<div class='section-title'>Top 25 Feature Importances</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>LightGBM split-based importance — higher = more splits used</div>", unsafe_allow_html=True)

    top_n = importance_df.head(25).sort_values("importance")
    fig = px.bar(
        top_n, x="importance", y="feature", orientation="h",
        color="importance", color_continuous_scale="Blues",
    )
    fig.update_traces(marker_line_width=0)
    apply_theme(fig, height=520)
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(title_text="Importance (splits)")
    fig.update_yaxes(title_text="")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<hr class='divider'/>", unsafe_allow_html=True)

    # Feature group breakdown
    st.markdown("<div class='section-title'>Feature Groups</div>", unsafe_allow_html=True)

    groups = {
        "Lag Features":     [f for f in importance_df["feature"] if "lag_" in f],
        "Rolling Stats":    [f for f in importance_df["feature"] if "rolling_" in f],
        "Date/Calendar":    [f for f in importance_df["feature"] if f in ["day","week","month","quarter","year","wday","dayofweek"]],
        "Price Signals":    [f for f in importance_df["feature"] if "price" in f or "sell_price" in f],
        "Event/SNAP":       [f for f in importance_df["feature"] if "event" in f or "snap" in f or "has_event" in f],
        "Hierarchy":        [f for f in importance_df["feature"] if "_avg_" in f or "dept_" in f or "store_avg" in f],
    }

    imp_lookup = importance_df.set_index("feature")["importance"].to_dict()

    gc1, gc2, gc3 = st.columns(3)
    cols = [gc1, gc2, gc3]
    for i, (grp, feats) in enumerate(groups.items()):
        total_imp = sum(imp_lookup.get(f, 0) for f in feats)
        cols[i % 3].markdown(f"""
        <div class='kpi-card' style='border-left-color:{PALETTE[i]};'>
            <div class='kpi-label'>{grp}</div>
            <div class='kpi-value' style='font-size:1.4rem;'>{total_imp:,}</div>
            <div class='kpi-sub'>{len(feats)} feature(s)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='divider'/>", unsafe_allow_html=True)

    # Event vs SNAP day counts in validation
    if "has_event" in val_df.columns and "is_snap" in val_df.columns:
        st.markdown("<div class='section-title'>Event & SNAP Days in Validation</div>", unsafe_allow_html=True)
        ev_c1, ev_c2 = st.columns(2)
        ev_c1.metric("Event Days (rows)", f"{int(val_df['has_event'].sum()):,}")
        ev_c2.metric("SNAP Days (rows)",  f"{int(val_df['is_snap'].sum()):,}")


# ═══════════════════════════════════════════════════════════════
# SECTION: MODEL EVALUATION
# ═══════════════════════════════════════════════════════════════
elif section == "📊  Model Evaluation":
    st.markdown("<div class='page-header'><h1>📊 Model Evaluation</h1><p>Quantitative accuracy across hierarchies, events, and error distributions</p></div>", unsafe_allow_html=True)

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    def kpi(col, label, value, sub, accent="#4f6ef7"):
        col.markdown(f"""
        <div class='kpi-card' style='border-left-color:{accent};'>
            <div class='kpi-label'>{label}</div>
            <div class='kpi-value'>{value}</div>
            <div class='kpi-sub'>{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    kpi(k1, "RMSSE", f"{metrics['overall_rmsse']:.4f}", "Avg item-level", "#4f6ef7")
    kpi(k2, "WMAPE", f"{metrics['wmape']*100:.2f}%",    "Weighted abs % error", "#f7914f")
    kpi(k3, "MAE",   f"{metrics['mae']:.4f}",           "Mean absolute error", "#4fc9a4")
    kpi(k4, "RMSE",  f"{metrics['rmse']:.4f}",          "Root mean squared error", "#f74f6e")

    st.markdown("<hr class='divider'/>", unsafe_allow_html=True)

    # ── Row 1: RMSSE distribution + Error distribution
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        st.markdown("<div class='section-title'>RMSSE Distribution (per SKU)</div>", unsafe_allow_html=True)
        fig = px.histogram(
            x=metrics["item_scores"].clip(upper=5), nbins=60,
            color_discrete_sequence=["#4f6ef7"],
        )
        fig.update_traces(marker_line_width=0)
        apply_theme(fig, height=300)
        fig.add_vline(x=1.0, line_dash="dash", line_color="#f74f6e",
                      annotation_text="Naïve baseline = 1.0")
        fig.add_vline(x=metrics["overall_rmsse"], line_dash="dot", line_color="#4fc9a4",
                      annotation_text=f"Mean = {metrics['overall_rmsse']:.2f}",
                      annotation_position="top left")
        fig.update_xaxes(title_text="RMSSE (clipped at 5)")
        fig.update_yaxes(title_text="Count")
        st.plotly_chart(fig, use_container_width=True)

    with r1c2:
        st.markdown("<div class='section-title'>Forecast Error Distribution</div>", unsafe_allow_html=True)
        fig2 = px.histogram(
            x=np.clip(metrics["errors"], -10, 10), nbins=80,
            color_discrete_sequence=["#f7914f"],
        )
        fig2.update_traces(marker_line_width=0)
        apply_theme(fig2, height=300)
        fig2.add_vline(x=0, line_dash="dash", line_color="#111827",
                       annotation_text="Zero bias")
        fig2.add_vline(x=metrics["bias"], line_dash="dot", line_color="#4f6ef7",
                       annotation_text=f"Bias={metrics['bias']:.3f}",
                       annotation_position="top left")
        fig2.update_xaxes(title_text="Prediction − Actual (clipped ±10)")
        fig2.update_yaxes(title_text="Count")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<hr class='divider'/>", unsafe_allow_html=True)

    # ── Row 2: Hierarchical tables
    st.markdown("<div class='section-title'>Hierarchical Error Breakdown</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>WMAPE % and directional bias across dimensions</div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["By State", "By Store", "By Category"])

    def render_hier_table(df, id_col):
        display = df[[id_col, "sales", "pred", "wmape", "bias_pct"]].copy()
        display.columns = [id_col.replace("_id","").title(), "Actual Sales", "Predicted", "WMAPE %", "Bias %"]
        st.dataframe(
            display.style
                .background_gradient(subset=["WMAPE %"], cmap="YlOrRd")
                .background_gradient(subset=["Bias %"],  cmap="coolwarm")
                .format({"Actual Sales": "{:,.0f}", "Predicted": "{:,.0f}",
                         "WMAPE %": "{:.1f}%", "Bias %": "{:.1f}%"}),
            use_container_width=True, hide_index=True,
        )

    with tab1: render_hier_table(metrics["state_eval"], "state_id")
    with tab2: render_hier_table(metrics["store_eval"], "store_id")
    with tab3: render_hier_table(metrics["cat_eval"],   "cat_id")

    st.markdown("<hr class='divider'/>", unsafe_allow_html=True)

    # ── Row 3: Event analysis + Zero demand
    ec1, ec2 = st.columns([3, 2])

    with ec1:
        st.markdown("<div class='section-title'>Event vs Non-Event Performance</div>", unsafe_allow_html=True)
        st.dataframe(
            metrics["event_eval"].style
                .background_gradient(subset=["RMSE","MAE","RMSSE"], cmap="YlOrRd"),
            use_container_width=True, hide_index=True,
        )

    with ec2:
        st.markdown("<div class='section-title'>Sparse Demand Quality</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class='kpi-card' style='border-left-color:#f74f6e;'>
            <div class='kpi-label'>False Positive Rate</div>
            <div class='kpi-value' style='font-size:1.5rem;'>{metrics['fp_rate']:.1f}%</div>
            <div class='kpi-sub'>% of zero-sales days where model predicted &gt; 0.5</div>
        </div>
        <div class='kpi-card' style='border-left-color:#f7914f;margin-top:12px;'>
            <div class='kpi-label'>Avg Prediction on Zero Days</div>
            <div class='kpi-value' style='font-size:1.5rem;'>{metrics['fp_avg']:.3f}</div>
            <div class='kpi-sub'>Units — ideally close to 0</div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# SECTION: FORECAST EXPLORER
# ═══════════════════════════════════════════════════════════════
elif section == "🔍  Forecast Explorer":
    st.markdown("<div class='page-header'><h1>🔍 Forecast Explorer</h1><p>Anecdotal validation — actual vs predicted for any product</p></div>", unsafe_allow_html=True)

    # ── Controls
    ctrl1, ctrl2 = st.columns([2, 3])

    with ctrl1:
        st.markdown("<div class='section-title'>Select Product</div>", unsafe_allow_html=True)
        mode = st.selectbox(
            "Selection mode",
            ["High-Volume SKU", "Sparse SKU", "Random SKU", "Custom SKU"],
        )

    sku_stats  = metrics["sku_stats"]
    all_ids    = val_df["id"].unique().tolist()

    if mode == "High-Volume SKU":
        selected_id = metrics["high_vol_id"]
    elif mode == "Sparse SKU":
        selected_id = metrics["sparse_id"]
    elif mode == "Random SKU":
        if "rand_sku" not in st.session_state:
            st.session_state["rand_sku"] = np.random.choice(all_ids)
        if st.button("🎲 Pick another random SKU"):
            st.session_state["rand_sku"] = np.random.choice(all_ids)
        selected_id = st.session_state["rand_sku"]
    else:  # Custom
        with ctrl2:
            st.markdown("<br>", unsafe_allow_html=True)
            cats = sorted(val_df["cat_id"].unique()) if "cat_id" in val_df.columns else []
            sel_cat = st.selectbox("Category", ["All"] + list(cats))
            id_pool = val_df if sel_cat == "All" else val_df[val_df["cat_id"] == sel_cat]

            if "store_id" in val_df.columns:
                stores = sorted(id_pool["store_id"].unique())
                sel_store = st.selectbox("Store", ["All"] + list(stores))
                if sel_store != "All":
                    id_pool = id_pool[id_pool["store_id"] == sel_store]

            selected_id = st.selectbox("SKU ID", sorted(id_pool["id"].unique()))

    # ── SKU metadata badges
    sku_row = sku_stats[sku_stats["id"] == selected_id]
    if not sku_row.empty:
        avg_s = sku_row["avg_sales"].values[0]
        zr    = sku_row["zero_ratio"].values[0]
        sku_type = (
            "🔥 High-Volume" if avg_s >= sku_stats["avg_sales"].quantile(0.9) else
            "❄️ Sparse"      if zr    >= 0.7 else
            "📦 Regular"
        )
        st.markdown(f"""
        <div style='margin:12px 0 20px 0;'>
            <span class='badge'>{selected_id}</span>
            <span class='badge badge-{"green" if "High" in sku_type else "red" if "Sparse" in sku_type else "amber"}'>{sku_type}</span>
            <span class='badge'>Avg Sales: {avg_s:.2f}</span>
            <span class='badge'>Zero Ratio: {zr:.1%}</span>
        </div>
        """, unsafe_allow_html=True)

    # ── Pull data for selected SKU
    sample = val_df[val_df["id"] == selected_id].sort_values("date")

    if sample.empty:
        st.warning("No validation data found for this SKU.")
    else:
        sku_mae   = float(np.mean(np.abs(sample["sales"] - sample["pred"])))
        sku_rmse  = float(np.sqrt(np.mean((sample["sales"] - sample["pred"]) ** 2)))
        sku_wmape = float(
            np.sum(np.abs(sample["sales"] - sample["pred"])) /
            max(np.sum(np.abs(sample["sales"])), 1e-6)
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("SKU MAE",   f"{sku_mae:.3f}")
        m2.metric("SKU RMSE",  f"{sku_rmse:.3f}")
        m3.metric("SKU WMAPE", f"{sku_wmape*100:.1f}%")

        # ── Main forecast chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sample["date"], y=sample["sales"],
            mode="lines+markers", name="Actual",
            line=dict(color="#111827", width=2),
            marker=dict(size=5, color="#111827"),
        ))
        fig.add_trace(go.Scatter(
            x=sample["date"], y=sample["pred"],
            mode="lines+markers", name="Predicted",
            line=dict(color="#4f6ef7", width=2, dash="dot"),
            marker=dict(size=5, color="#4f6ef7", symbol="x"),
        ))
        apply_theme(fig, title=f"Actual vs Predicted — {selected_id}", height=380)
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Units Sold")
        st.plotly_chart(fig, use_container_width=True)

        # ── Error subplot
        err_series = sample["pred"] - sample["sales"]
        fig_err = go.Figure()
        fig_err.add_bar(
            x=sample["date"], y=err_series,
            marker_color=["#f74f6e" if e < 0 else "#4f6ef7" for e in err_series],
            name="Daily Error",
        )
        fig_err.add_hline(y=0, line_color="#111827", line_width=1)
        apply_theme(fig_err, title="Daily Forecast Error (Predicted − Actual)", height=240)
        fig_err.update_xaxes(title_text="Date")
        fig_err.update_yaxes(title_text="Error")
        st.plotly_chart(fig_err, use_container_width=True)

        st.markdown("<hr class='divider'/>", unsafe_allow_html=True)

        # ── 3-SKU side-by-side comparison
        st.markdown("<div class='section-title'>3-Type SKU Comparison</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-sub'>High-volume · Sparse · Random — in one view</div>", unsafe_allow_html=True)

        rand_sku = st.session_state.get("rand_sku", np.random.choice(all_ids))
        three_ids = [
            ("High-Volume", metrics["high_vol_id"], "#4f6ef7"),
            ("Sparse",      metrics["sparse_id"],   "#f7914f"),
            ("Random",      rand_sku,               "#4fc9a4"),
        ]

        fig3 = make_subplots(rows=3, cols=1, shared_xaxes=False,
                             subplot_titles=[t for t, _, _ in three_ids],
                             vertical_spacing=0.1)

        for row, (label, sid, color) in enumerate(three_ids, start=1):
            s = val_df[val_df["id"] == sid].sort_values("date")
            fig3.add_trace(go.Scatter(
                x=s["date"], y=s["sales"],
                mode="lines", name=f"{label} Actual",
                line=dict(color="#111827", width=1.5),
                showlegend=(row == 1),
            ), row=row, col=1)
            fig3.add_trace(go.Scatter(
                x=s["date"], y=s["pred"],
                mode="lines", name=f"{label} Predicted",
                line=dict(color=color, width=1.8, dash="dot"),
                showlegend=(row == 1),
            ), row=row, col=1)

        fig3.update_layout(
            height=700,
            paper_bgcolor="white",
            plot_bgcolor="#f7f8fc",
            font=dict(family="DM Sans", size=11, color="#3d4459"),
            margin=dict(l=20, r=20, t=40, b=20),
        )
        #fig3.update_xaxes(gridcolor="#eef0f5")
        #fig3.update_yaxes(gridcolor="#eef0f5")
        fig3.update_xaxes(gridcolor="#d6dbe6")
        fig3.update_yaxes(gridcolor="#d6dbe6")
        
        st.plotly_chart(fig3, use_container_width=True)