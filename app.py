import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import fetch_crypto_data
from src.garch_model import fit_garch_model
from src.lstm_model import train_lstm_volatility
from src.risk_engine import calculate_position_size
from src.transformer import train_transformer_volatility

st.set_page_config(
    page_title="Crypto Volatility Forecasting Engine", layout="wide"
)

st.title("📈 Crypto Volatility Forecasting: GARCH vs LSTM vs Transformer")
st.markdown(
    "A quantitative risk framework comparing classical econometrics with deep learning architectures."
)

# Sidebar controls
st.sidebar.header("Configuration")
symbol = st.sidebar.selectbox(
    "Select Asset", ["BTC-USD", "ETH-USD", "SOL-USD"]
)
capital = st.sidebar.number_input(
    "Trading Capital ($USD)", value=10000.0, step=1000.0
)

if st.sidebar.button("Run Volatility Engine"):
    with st.spinner("Fetching market data and running models..."):
        df = fetch_crypto_data(symbol=symbol)

        # 1. Fit GARCH
        _, garch_vol = fit_garch_model(df["Log_Return"])

        # 2. Fit LSTM
        _, lstm_vol = train_lstm_volatility(df["Log_Return"], epochs=20)

        # 3. Fit Transformer
        _, trans_vol = train_transformer_volatility(
            df["Log_Return"], epochs=20
        )

        # Align lengths for plotting
        min_len = min(len(garch_vol), len(lstm_vol), len(trans_vol))
        plot_dates = df.index[-min_len:]

        # Create Plotly Chart
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=plot_dates,
                y=garch_vol[-min_len:] * 100,
                name="GARCH(1,1)",
                line=dict(color="blue"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=plot_dates,
                y=lstm_vol[-min_len:] * 100,
                name="LSTM Network",
                line=dict(color="orange"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=plot_dates,
                y=trans_vol[-min_len:] * 100,
                name="Transformer (Attention)",
                line=dict(color="green"),
            )
        )

        fig.update_layout(
            title=f"Predicted Daily Volatility (%) - {symbol}",
            xaxis_title="Date",
            yaxis_title="Volatility (%)",
            template="plotly_white",
        )

        st.plotly_chart(fig, use_container_width=True)

        # Risk Management Output based on Latest Forecast
        latest_garch_vol = float(garch_vol.iloc[-1])
        risk_summary = calculate_position_size(
            account_balance=capital, predicted_volatility=latest_garch_vol
        )

        st.subheader("🛡️ Real-Time Risk Management Engine")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Predicted Volatility", risk_summary["predicted_daily_volatility"])
        col2.metric("1-Day 95% VaR", risk_summary["one_day_95pct_VaR"])
        col3.metric("Max Dollar Risk", risk_summary["max_allowed_risk_usd"])
        col4.metric(
            "Recommended Max Position", risk_summary["recommended_position_usd"]
        )
