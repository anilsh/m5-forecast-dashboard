# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb

from sklearn.metrics import mean_squared_error, mean_absolute_error

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="M5 Forecasting Dashboard",
    layout="wide"
)

st.title("M5 Forecasting Analysis Dashboard")
st.markdown("Interactive analysis for M5 retail demand forecasting")

# ============================================================
# CONFIG
# ============================================================

DATA_PATH = './data/'
MODEL_PATH = './m5_lightgbm_model_v4.txt'

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data

def load_data():

    val_df = pd.read_csv(f'{DATA_PATH}/validation_predictions.csv')

    feature_importance = pd.read_csv(
        f'{DATA_PATH}/feature_importance.csv'
    )

    item_scores = pd.read_csv(
        f'{DATA_PATH}/item_rmsse_scores.csv'
    )

    return val_df, feature_importance, item_scores


val_df, feature_importance, item_scores = load_data()

# ============================================================
# METRICS
# ============================================================

rmse = np.sqrt(
    mean_squared_error(val_df['sales'], val_df['pred'])
)

mae = mean_absolute_error(
    val_df['sales'],
    val_df['pred']
)

wmape = (
    np.sum(np.abs(val_df['sales'] - val_df['pred'])) /
    np.sum(np.abs(val_df['sales']))
)

overall_rmsse = item_scores['rmsse'].mean()

# ============================================================
# SIDEBAR
# ============================================================

section = st.sidebar.radio(
    'Select Section',
    [
        'Overview',
        'EDA',
        'Forecast Performance',
        'Product Explorer',
        'Hierarchical Analysis',
        'Sparse Demand Analysis',
        'Feature Importance',
        'Model Insights'
    ]
)

# ============================================================
# OVERVIEW
# ============================================================

if section == 'Overview':

    st.header('Model Overview')

    col1, col2, col3, col4 = st.columns(4)

    col1.metric('RMSSE', f'{overall_rmsse:.4f}')
    col2.metric('RMSE', f'{rmse:.4f}')
    col3.metric('MAE', f'{mae:.4f}')
    col4.metric('WMAPE', f'{wmape:.2%}')

    st.markdown('---')

    c1, c2, c3 = st.columns(3)

    c1.metric('Products', val_df['id'].nunique())
    c2.metric('Stores', val_df['store_id'].nunique())
    c3.metric('Categories', val_df['cat_id'].nunique())

    st.markdown('---')

    st.markdown(
        '''
        ### Summary

        - LightGBM-based retail forecasting model
        - Hierarchical M5 forecasting analysis
        - Includes sparse demand evaluation
        - Event-based forecasting diagnostics
        - Product-level actual vs predicted analysis
        '''
    )

# ============================================================
# EDA
# ============================================================

elif section == 'EDA':

    st.header('Exploratory Data Analysis')

    st.subheader('Aggregate Sales Trend')

    agg_sales = (
        val_df.groupby('date')['sales']
        .sum()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(agg_sales['date'], agg_sales['sales'])

    ax.set_title('Aggregate Sales Trend')

    st.pyplot(fig)

    st.subheader('Category-Level Sales')

    selected_cat = st.selectbox(
        'Select Category',
        sorted(val_df['cat_id'].unique())
    )

    cat_df = val_df[val_df['cat_id'] == selected_cat]

    cat_sales = (
        cat_df.groupby('date')['sales']
        .sum()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(cat_sales['date'], cat_sales['sales'])

    ax.set_title(f'Sales Trend - {selected_cat}')

    st.pyplot(fig)

# ============================================================
# FORECAST PERFORMANCE
# ============================================================

elif section == 'Forecast Performance':

    st.header('Forecast Performance')

    st.subheader('Metrics Summary')

    metrics_df = pd.DataFrame({
        'Metric': ['RMSSE', 'RMSE', 'MAE', 'WMAPE'],
        'Value': [
            round(overall_rmsse, 4),
            round(rmse, 4),
            round(mae, 4),
            round(wmape, 4)
        ]
    })

    st.dataframe(metrics_df)

    st.subheader('RMSSE Distribution')

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(item_scores['rmsse'], bins=50)

    ax.set_title('Item-Level RMSSE Distribution')

    st.pyplot(fig)

    st.subheader('Forecast Error Distribution')

    val_df['error'] = val_df['pred'] - val_df['sales']

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(val_df['error'], bins=100)

    ax.set_title('Forecast Error Distribution')

    st.pyplot(fig)

# ============================================================
# PRODUCT EXPLORER
# ============================================================

elif section == 'Product Explorer':

    st.header('Product-Level Forecast Explorer')

    product_ids = sorted(val_df['id'].unique())

    selected_product = st.selectbox(
        'Select Product',
        product_ids
    )

    product_df = val_df[
        val_df['id'] == selected_product
    ]

    col1, col2, col3 = st.columns(3)

    avg_sales = product_df['sales'].mean()
    zero_ratio = (product_df['sales'] == 0).mean()
    product_mae = np.mean(
        np.abs(product_df['sales'] - product_df['pred'])
    )

    col1.metric('Average Sales', f'{avg_sales:.2f}')
    col2.metric('Zero Ratio', f'{zero_ratio:.2%}')
    col3.metric('SKU MAE', f'{product_mae:.2f}')

    fig, ax = plt.subplots(figsize=(16, 5))

    ax.plot(
        product_df['date'],
        product_df['sales'],
        label='Actual',
        marker='o'
    )

    ax.plot(
        product_df['date'],
        product_df['pred'],
        label='Predicted',
        marker='x'
    )

    ax.set_title(f'Forecast vs Actual - {selected_product}')

    ax.legend()

    st.pyplot(fig)

    st.subheader('Raw Prediction Data')

    st.dataframe(
        product_df[
            ['date', 'sales', 'pred']
        ]
    )

# ============================================================
# HIERARCHICAL ANALYSIS
# ============================================================

elif section == 'Hierarchical Analysis':

    st.header('Hierarchical Forecast Analysis')

    analysis_level = st.selectbox(
        'Select Level',
        ['state_id', 'store_id', 'cat_id']
    )

    grouped = (
        val_df.groupby(analysis_level)
        .agg({
            'sales': 'sum',
            'pred': 'sum'
        })
        .reset_index()
    )

    grouped['error'] = grouped['pred'] - grouped['sales']

    grouped['wmape'] = (
        np.abs(grouped['error']) /
        grouped['sales']
    )

    st.dataframe(grouped)

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(
        data=grouped,
        x=analysis_level,
        y='wmape',
        ax=ax
    )

    ax.set_title(f'WMAPE by {analysis_level}')

    st.pyplot(fig)

# ============================================================
# SPARSE DEMAND ANALYSIS
# ============================================================

elif section == 'Sparse Demand Analysis':

    st.header('Sparse Demand Analysis')

    zero_actual = val_df[
        val_df['sales'] == 0
    ]

    false_positive_rate = np.mean(
        zero_actual['pred'] > 0.5
    )

    avg_false_positive = zero_actual['pred'].mean()

    col1, col2 = st.columns(2)

    col1.metric(
        'False Positive Rate',
        f'{false_positive_rate:.2%}'
    )

    col2.metric(
        'Average Predicted Sales',
        f'{avg_false_positive:.4f}'
    )

    st.markdown(
        '''
        Sparse-demand products are difficult because many days have zero sales.
        This section evaluates whether the model overpredicts demand.
        '''
    )

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

elif section == 'Feature Importance':

    st.header('Feature Importance')

    top_n = st.slider(
        'Number of Features',
        5,
        30,
        15
    )

    top_features = (
        feature_importance
        .sort_values('importance', ascending=False)
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.barh(
        top_features['feature'][::-1],
        top_features['importance'][::-1]
    )

    ax.set_title('Top Feature Importances')

    st.pyplot(fig)

# ============================================================
# MODEL INSIGHTS
# ============================================================

elif section == 'Model Insights':

    st.header('Model Insights & Limitations')

    st.markdown(
        '''
        ### Current Strengths

        - Captures retail seasonality reasonably well
        - Handles large-scale hierarchical forecasting
        - Supports sparse/intermittent demand
        - Efficient LightGBM-based implementation

        ### Current Limitations

        - Baseline implementation
        - No recursive forecasting
        - No probabilistic forecasting
        - No hierarchical reconciliation

        ### Future Improvements

        - Temporal Fusion Transformer (TFT)
        - DeepAR / Transformer models
        - Better rolling-window validation
        - Quantile forecasting
        - Forecast reconciliation
        '''
    )
