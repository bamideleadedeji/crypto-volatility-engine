import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from src.data_loader import fetch_crypto_data
from src.garch_model import fit_garch_model
from src.risk_engine import calculate_position_size

# Safe imports for Deep Learning models
try:
    from src.lstm_model import train_lstm_volatility
    HAS_LSTM = True
except Exception:
    HAS_LSTM = False

try:
    from src.transformer import train_transformer_volatility
    HAS_TRANSFORMER = True
except Exception:
    HAS_TRANSFORMER = False

# Page Configuration - Executive Theme
st.set_page_config(
    page_title="Crypto Volatility & Quantitative Risk Suite",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Header & Metadata
st.title("📈 Crypto Volatility Forecasting & Quantitative Risk Suite")
st.caption("Institutional Multi-Model Benchmarking Engine (GARCH(1,1) vs. LSTM vs. Transformer) with Parametric VaR Risk Targeting")
st.divider()

# Sidebar controls
st.sidebar.header("🕹️ Institutional Controls")
symbol = st.sidebar.selectbox(
    "Asset Identifier", ["BTC-USD", "ETH-USD", "SOL-USD"], index=0
)
capital = st.sidebar.number_input(
    "Portfolio Capital ($USD)", value=10000.0, step=1000.0, format="%.2f"
)
confidence_level = st.sidebar.slider(
    "VaR Confidence Level (%)", min_value=90.0, max_value=99.0, value=95.0, step=1.0
) / 100.0
epochs = st.sidebar.number_input(
    "Neural Network Epochs", value=20, min_value=5, max_value=100, step=5
)

st.sidebar.divider()
st.sidebar.info("💡 **Consultant Note:** Benchmarks econometric time-series against deep learning architectures to optimize risk-weighted position sizing.")

if st.sidebar.button("🚀 Execute Quantitative Analytics", use_container_width=True):
    with st.spinner("Retrieving market data feed, parameterizing models, and performing econometric audit..."):
        df = fetch_crypto_data(symbol=symbol)

        if df is None or df.empty or len(df) < 30:
            st.error("Error: Insufficient market data returned from data feeds.")
            st.stop()

        # 1. Fit GARCH Baseline
        _, garch_vol = fit_garch_model(df["Log_Return"])

        # 2. Fit LSTM
        if HAS_LSTM:
            try:
                _, lstm_vol = train_lstm_volatility(df["Log_Return"], epochs=int(epochs))
            except Exception as e:
                st.warning(f"LSTM Training Note: {e}. Utilizing GARCH baseline.")
                lstm_vol = garch_vol
        else:
            lstm_vol = garch_vol

        # 3. Fit Transformer
        if HAS_TRANSFORMER:
            try:
                _, trans_vol = train_transformer_volatility(df["Log_Return"], epochs=int(epochs))
            except Exception as e:
                st.warning(f"Transformer Training Note: {e}. Utilizing GARCH baseline.")
                trans_vol = garch_vol
        else:
            trans_vol = garch_vol

        # Data Alignment & Formatting
        min_len = min(len(garch_vol), len(lstm_vol), len(trans_vol))
        plot_dates = df.index[-min_len:]
        
        garch_series = garch_vol[-min_len:] * 100
        lstm_series = lstm_vol[-min_len:] * 100
        trans_series = trans_vol[-min_len:] * 100

        # Latest Forecast Metrics
        latest_price = df["Close"].iloc[-1]
        latest_garch_vol = float(garch_vol.iloc[-1])
        latest_lstm_vol = float(lstm_vol.iloc[-1]) if HAS_LSTM else latest_garch_vol
        latest_trans_vol = float(trans_vol.iloc[-1]) if HAS_TRANSFORMER else latest_garch_vol

        risk_summary = calculate_position_size(
            account_balance=capital, predicted_volatility=latest_garch_vol
        )

        # -------------------------------------------------------------
        # Section 1: Executive Key Metrics Summary
        # -------------------------------------------------------------
        st.subheader("📌 Executive Risk & Volatility Dashboard")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Spot Price", f"${latest_price:,.2f}")
        m2.metric("GARCH(1,1) Vol", f"{latest_garch_vol*100:.2f}%")
        m3.metric("LSTM Vol", f"{latest_lstm_vol*100:.2f}%")
        m4.metric("1-Day Parametric VaR", risk_summary.get("one_day_95pct_VaR", "N/A"))
        m5.metric("Target Max Allocation", risk_summary.get("recommended_position_usd", "N/A"))

        st.divider()

        # -------------------------------------------------------------
        # Section 2: Multi-Model Volatility Benchmark & Analytics
        # -------------------------------------------------------------
        tab_chart, tab_table, tab_stats = st.tabs([
            "📈 Forecast Comparison Plot", 
            "📋 Model Volatility Diagnostics", 
            "📊 Econometric Statistical Audit"
        ])

        with tab_chart:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=plot_dates,
                    y=garch_series,
                    name="GARCH(1,1) Econometric",
                    line=dict(color="#1f77b4", width=2.5),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=plot_dates,
                    y=lstm_series,
                    name="LSTM Recurrent Net",
                    line=dict(color="#ff7f0e", width=2, dash="dash"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=plot_dates,
                    y=trans_series,
                    name="Transformer Self-Attention",
                    line=dict(color="#2ca02c", width=2, dash="dot"),
                )
            )

            fig.update_layout(
                title=f"Multi-Model Daily Volatility Forecast Horizon (%) — {symbol}",
                xaxis_title="Timeline",
                yaxis_title="Annualized/Daily Conditional Volatility (%)",
                template="plotly_white",
                height=520,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab_table:
            st.markdown("##### Model Forecast Matrix (Last 15 Trading Sessions)")
            diag_df = pd.DataFrame({
                "Date": plot_dates[-15:].strftime("%Y-%m-%d"),
                "Asset Spot ($)": df["Close"].iloc[-15:].values,
                "Log Return (%)": df["Log_Return"].iloc[-15:].values * 100,
                "GARCH Vol (%)": garch_series[-15:].values,
                "LSTM Vol (%)": lstm_series[-15:].values,
                "Transformer Vol (%)": trans_series[-15:].values
            })
            st.dataframe(diag_df.style.format({
                "Asset Spot ($)": "${:,.2f}",
                "Log Return (%)": "{:+.2f}%",
                "GARCH Vol (%)": "{:+.2f}%",
                "GARCH Vol (%)": "{:.2f}%",
                "LSTM Vol (%)": "{:.2f}%",
                "Transformer Vol (%)": "{:.2f}%"
            }), use_container_width=True)

        with tab_stats:
            st.markdown("##### Econometric Distribution & Dispersion Statistics")
            s1, s2, s3 = st.columns(3)
            with s1:
                st.markdown("**GARCH Volatility Distribution**")
                st.write(pd.Series(garch_series).describe().to_frame("GARCH Metrics"))
            with s2:
                st.markdown("**LSTM Volatility Distribution**")
                st.write(pd.Series(lstm_series).describe().to_frame("LSTM Metrics"))
            with s3:
                st.markdown("**Transformer Volatility Distribution**")
                st.write(pd.Series(trans_series).describe().to_frame("Transformer Metrics"))

        st.divider()

        # -------------------------------------------------------------
        # Section 3: Professional Risk Management Engine
        # -------------------------------------------------------------
        st.subheader("🛡️ Executive Risk-Targeted Position Sizing Framework")
        r1, r2 = st.columns([1, 1])

        with r1:
            st.markdown("#### Quantitative Risk Metrics")
            st.write(f"• **Portfolio Capital:** `${capital:,.2f}`")
            st.write(f"• **Predicted Daily Volatility:** `{risk_summary.get('predicted_daily_volatility', 'N/A')}`")
            st.write(f"• **1-Day 95% Value-at-Risk (VaR):** `{risk_summary.get('one_day_95pct_VaR', 'N/A')}`")
            st.write(f"• **Maximum Allowable Risk Budget:** `{risk_summary.get('max_allowed_risk_usd', 'N/A')}`")
            st.write(f"• **Recommended Allocation Cap:** `{risk_summary.get('recommended_position_usd', 'N/A')}`")

        with r2:
            st.markdown("#### Capital Allocation Breakdown")
            pos_val_str = str(risk_summary.get("recommended_position_usd", "$0")).replace("$", "").replace(",", "")
            try:
                rec_pos_val = float(pos_val_str)
            except Exception:
                rec_pos_val = capital * 0.2
            
            cash_val = max(0.0, capital - rec_pos_val)
            alloc_df = pd.DataFrame({
                "Category": ["Target Position Allocation", "Unallocated Risk Buffer"],
                "Amount ($)": [rec_pos_val, cash_val]
            })
            fig_pie = px.pie(
                alloc_df, values="Amount ($)", names="Category", 
                hole=0.4, color_discrete_sequence=["#1f77b4", "#e377c2"]
            )
            fig_pie.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_pie, use_container_width=True)
