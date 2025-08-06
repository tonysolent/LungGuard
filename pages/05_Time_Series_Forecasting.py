import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Time Series & Forecasting", layout="wide")
st.title("📈 Time Series Analysis & Forecasting")

st.markdown("""
Upload a time-series CSV, visualise its trends and seasonal patterns, and generate ARIMA forecasts.  
*Only columns appropriate for time series are shown in the dropdowns.*
""")

uploaded_file = st.file_uploader(
    "Upload your time series CSV file",
    type=["csv"],
    help="Upload a CSV with a datetime column (e.g., date, timestamp) and a numeric value column for time series analysis."
)

def best_datetime_cols(df):
    date_like = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
    date_like += [
        c for c in df.columns
        if pd.api.types.is_datetime64_any_dtype(df[c]) or
        (pd.to_datetime(df[c], errors="coerce").notna().sum() > 0.8 * len(df))
    ]
    seen = set()
    date_cols = []
    for c in date_like:
        if c not in seen and "patientbirth" not in c.lower():
            date_cols.append(c)
            seen.add(c)
    return date_cols if date_cols else [df.columns[0]]

def best_numeric_cols(df):
    cols = []
    for c in df.select_dtypes(include=np.number).columns:
        cname = c.lower()
        nunique = df[c].nunique()
        if (
            nunique < 5 or
            "id" in cname or
            "study" in cname or
            "year" in cname or
            cname.startswith("is_") or
            cname.endswith("_flag") or
            "patientage" in cname or
            df[c].std() < 1e-3 or
            df[c].isnull().all()
        ):
            continue
        cols.append(c)
    return cols

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns.")
    st.dataframe(df.head(10), use_container_width=True)

    date_cols = best_datetime_cols(df)
    num_cols = best_numeric_cols(df)

    st.subheader("Step 1: Select Columns")
    date_col = st.selectbox(
        "Date/Time column", 
        options=date_cols, 
        help="Pick the column with timestamps or dates (except PatientBirth). This is used for the time axis."
    )
    value_col = st.selectbox(
        "Value column (numeric, continuous)", 
        options=num_cols, 
        help="Select a numeric column to forecast (e.g., cases, counts, measurements; PatientAge excluded)."
    )

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    ts_df = df[[date_col, value_col]].dropna().sort_values(by=date_col)
    ts_df = ts_df.set_index(date_col)
    st.write(ts_df.head(10))
    st.caption("Table: Shows the first rows of your selected time series for review.")

    st.subheader("Step 2: Visualise Time Series")
    st.caption("Line plot shows how your selected value changes over time. Helps spot trends and outliers.")
    fig, ax = plt.subplots()
    ax.plot(ts_df.index, ts_df[value_col], marker='o', linestyle='-')
    ax.set_title("Time Series Plot")
    ax.set_xlabel(date_col)
    ax.set_ylabel(value_col)
    st.pyplot(fig)

    # --- Decomposition ---
    st.subheader("Step 3: Seasonal Decomposition")
    period = st.number_input(
        "Decomposition period (e.g., 12 for monthly, 7 for weekly)",
        value=12, min_value=2, max_value=365,
        help="Set the period that matches your seasonality (12=monthly, 7=weekly, etc.)."
    )
    st.caption("Decomposes the series into trend, seasonal, and residual components for deeper insight.")
    try:
        decomposition = seasonal_decompose(ts_df[value_col], period=int(period), model='additive')
        fig2, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8, 7))
        decomposition.observed.plot(ax=ax1, title="Observed")
        decomposition.trend.plot(ax=ax2, title="Trend")
        decomposition.seasonal.plot(ax=ax3, title="Seasonal")
        decomposition.resid.plot(ax=ax4, title="Residual")
        plt.tight_layout()
        st.pyplot(fig2)
    except Exception as e:
        st.warning(f"Could not perform decomposition: {e}")

    # --- ARIMA Forecasting ---
    st.subheader("Step 4: ARIMA Forecasting")
    st.caption("Fit an ARIMA model to predict future values. Adjust p, d, q for best results; forecast horizon sets how far to predict.")
    p = st.slider(
        "AR term (p)", 0, 5, 1, 
        help="Number of autoregressive (AR) terms. Controls how much past values influence prediction."
    )
    d = st.slider(
        "Differencing (d)", 0, 2, 1, 
        help="How many times to difference the data to make it stationary (remove trend)."
    )
    q = st.slider(
        "MA term (q)", 0, 5, 0, 
        help="Number of moving average (MA) terms. Controls how much past errors influence prediction."
    )
    forecast_horizon = st.number_input(
        "Forecast steps into future", min_value=1, max_value=100, value=12,
        help="Number of future time points to predict beyond your dataset."
    )

    with st.spinner("Fitting ARIMA and forecasting..."):
        try:
            model = ARIMA(ts_df[value_col], order=(p, d, q))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=forecast_horizon)
            freq = pd.infer_freq(ts_df.index)
            if not freq:
                freq = 'D'
            pred_index = pd.date_range(ts_df.index[-1], periods=forecast_horizon+1, freq=freq)[1:]

            fig3, ax = plt.subplots()
            ts_df[value_col].plot(ax=ax, label="Observed", color="blue")
            ax.plot(pred_index, forecast, label="Forecast", color="red", linestyle="--", marker='o')
            ax.set_title("ARIMA Forecast")
            ax.legend()
            st.pyplot(fig3)
            st.caption("Forecast plot: Blue is observed, red dashed is the predicted future.")

            st.write("Model summary:")
            st.text(model_fit.summary())
        except Exception as e:
            st.warning(f"ARIMA could not be fit: {e}")

else:
    st.info("Upload a time-series CSV file to get started.")

st.markdown("---")
st.caption("© 2025 Time Series & Forecasting | MSc Applied AI and Data Science, Solent University.")
