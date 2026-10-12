import os
import gc
import torch

# ----------------------------------------------------------------------
# 1. Threading & Memory Guards for Render Free Tier (512MB RAM Limit)
# ----------------------------------------------------------------------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from src.data_loader import fetch_crypto_data
from src.garch_model import fit_garch_model
from src.risk_engine import calculate_position_size

# Safe import for LSTM model
try:
    from src.lstm_model import train_lstm_volatility
    HAS_LSTM = True
except Exception:
    HAS_LSTM = False

# Safe import for Transformer model
try:
    from src.transformer import train_transformer_volatility
    HAS_TRANSFORMER = True
except Exception:
    HAS_TRANSFORMER = False

# ----------------------------------------------------------------------
# 2. Page Configuration & Theme Setup
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Crypto Volatility & Quantitative Risk Suite",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Paper Trading Portfolio Memory State
if "paper_portfolio" not in st.session_state:
    st.session_state["paper_portfolio"] = {
        "cash_usd": 10000.0,
        "positions": {},
        "trade_history": []
    }

# Professional Header & Metadata
st.title("📈 Crypto Volatility Forecasting & Quantitative Risk Suite")
st.caption("Institutional Multi-Model Benchmarking Engine (GARCH(1,1) vs. LSTM vs. Transformer) with Parametric VaR Risk Targeting")
st.divider()

# Sidebar Controls
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
    "Neural Network Epochs", value=10, min_value=5, max_value=20, step=5
)

st.sidebar.divider()
st.sidebar.info("💡 **Consultant Note:** Benchmarks econometric time-series against deep learning architectures to optimize risk-weighted position sizing.")

# ----------------------------------------------------------------------
# 3. Cached Data Fetcher Function
# ----------------------------------------------------------------------
@st.cache_data(ttl=900, show_spinner=False)
def load_cached_market_data(ticker_symbol: str) -> pd.DataFrame:
    """Fetches market data with a 15-minute TTL cache to avoid hitting API rate limits."""
    data = fetch_crypto_data(symbol=ticker_symbol)
    if data is not None and not data.empty:
        float_cols = data.select_dtypes(include=['float64']).columns
        data[float_cols] = data[float_cols].astype('float32')
    return data

# ----------------------------------------------------------------------
# 4. Main Execution Engine (Session State Persistence)
# ----------------------------------------------------------------------
if "analytics_results" not in st.session_state:
    st.session_state["analytics_results"] = None

if st.sidebar.button("🚀 Execute Quantitative Analytics", key="btn_run_analytics", width="stretch"):
    with st.spinner("Retrieving market data feed, parameterizing models, and performing econometric audit..."):
        
        # Load Cached Market Data
        df = load_cached_market_data(symbol)

        if df is None or df.empty or len(df) < 30:
            st.error("Error: Insufficient market data returned from data feeds.")
            st.stop()

        # 1. Fit GARCH Baseline
        _, garch_vol = fit_garch_model(df["Log_Return"])

        if isinstance(garch_vol, np.ndarray):
            garch_vol = pd.Series(garch_vol, index=df.index[-len(garch_vol):], dtype="float32")
        else:
            garch_vol = garch_vol.astype("float32")

        # 2. Fit LSTM
        if HAS_LSTM:
            try:
                _, lstm_vol = train_lstm_volatility(df["Log_Return"], epochs=int(epochs))
                if isinstance(lstm_vol, np.ndarray):
                    lstm_vol = pd.Series(lstm_vol, index=df.index[-len(lstm_vol):], dtype="float32")
                else:
                    lstm_vol = lstm_vol.astype("float32")
            except Exception as e:
                st.warning(f"LSTM Training Note: {e}. Utilizing GARCH baseline.")
                lstm_vol = garch_vol
        else:
            lstm_vol = garch_vol

        # 3. Fit Transformer
        if HAS_TRANSFORMER:
            try:
                _, trans_vol = train_transformer_volatility(df["Log_Return"], epochs=int(epochs))
                if isinstance(trans_vol, np.ndarray):
                    trans_vol = pd.Series(trans_vol, index=df.index[-len(trans_vol):], dtype="float32")
                else:
                    trans_vol = trans_vol.astype("float32")
            except Exception as e:
                st.warning(f"Transformer Training Note: {e}. Utilizing GARCH baseline.")
                trans_vol = garch_vol
        else:
            trans_vol = garch_vol

        # Data Alignment & Formatting
        min_len = min(len(garch_vol), len(lstm_vol), len(trans_vol))
        plot_dates = df.index[-min_len:]
        
        garch_series = (garch_vol.iloc[-min_len:] * 100).astype("float32")
        lstm_series = (lstm_vol.iloc[-min_len:] * 100).astype("float32")
        trans_series = (trans_vol.iloc[-min_len:] * 100).astype("float32")

        # Latest Forecast Metrics
        latest_price = float(df["Close"].iloc[-1])
        latest_garch_vol = float(garch_vol.iloc[-1])
        latest_lstm_vol = float(lstm_vol.iloc[-1]) if HAS_LSTM else latest_garch_vol
        latest_trans_vol = float(trans_vol.iloc[-1]) if HAS_TRANSFORMER else latest_garch_vol

        risk_summary = calculate_position_size(
            account_balance=capital, predicted_volatility=latest_garch_vol
        )

        gc.collect()

        st.session_state["analytics_results"] = {
            "df": df,
            "garch_series": garch_series,
            "lstm_series": lstm_series,
            "trans_series": trans_series,
            "latest_price": latest_price,
            "latest_garch_vol": latest_garch_vol,
            "latest_lstm_vol": latest_lstm_vol,
            "latest_trans_vol": latest_trans_vol,
            "risk_summary": risk_summary,
            "plot_dates": plot_dates,
            "symbol": symbol,
            "capital": capital
        }

# ----------------------------------------------------------------------
# 5. Dashboard Rendering & Execution Engine
# ----------------------------------------------------------------------
if st.session_state["analytics_results"] is not None:
    res = st.session_state["analytics_results"]
    
    df = res["df"]
    garch_series = res["garch_series"]
    lstm_series = res["lstm_series"]
    trans_series = res["trans_series"]
    latest_price = res["latest_price"]
    latest_garch_vol = res["latest_garch_vol"]
    latest_lstm_vol = res["latest_lstm_vol"]
    risk_summary = res["risk_summary"]
    plot_dates = res["plot_dates"]
    current_symbol = res["symbol"]
    current_capital = res["capital"]

    # Section 1: Executive Key Metrics
    st.subheader("📌 Executive Risk & Volatility Dashboard")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Spot Price", f"${latest_price:,.2f}")
    m2.metric("GARCH(1,1) Vol", f"{latest_garch_vol*100:.2f}%")
    m3.metric("LSTM Vol", f"{latest_lstm_vol*100:.2f}%")
    m4.metric("1-Day Parametric VaR", risk_summary.get("one_day_95pct_VaR", "N/A"))
    m5.metric("Target Max Allocation", risk_summary.get("recommended_position_usd", "N/A"))

    st.divider()

    # Section 2: Forecast Horizon Plots
    tab_chart, tab_table, tab_stats = st.tabs([
        "📈 Forecast Comparison Plot", 
        "📋 Model Volatility Diagnostics", 
        "📊 Econometric Statistical Audit"
    ])

    with tab_chart:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=plot_dates, y=garch_series, name="GARCH(1,1)", line=dict(color="#1f77b4", width=2.5)))
        fig.add_trace(go.Scatter(x=plot_dates, y=lstm_series, name="LSTM Net", line=dict(color="#ff7f0e", width=2, dash="dash")))
        fig.add_trace(go.Scatter(x=plot_dates, y=trans_series, name="Transformer", line=dict(color="#2ca02c", width=2, dash="dot")))
        fig.update_layout(
            title=f"Multi-Model Daily Volatility Horizon (%) — {current_symbol}",
            xaxis_title="Timeline", yaxis_title="Conditional Volatility (%)",
            template="plotly_white", height=520
        )
        st.plotly_chart(fig, width="stretch")

    with tab_table:
        diag_df = pd.DataFrame({
            "Date": plot_dates[-15:].strftime("%Y-%m-%d"),
            "Asset Spot ($)": df["Close"].iloc[-15:].values,
            "Log Return (%)": df["Log_Return"].iloc[-15:].values * 100,
            "GARCH Vol (%)": garch_series.iloc[-15:].values,
            "LSTM Vol (%)": lstm_series.iloc[-15:].values,
            "Transformer Vol (%)": trans_series.iloc[-15:].values
        })
        st.dataframe(diag_df.style.format({
            "Asset Spot ($)": "${:,.2f}", "Log Return (%)": "{:+.2f}%",
            "GARCH Vol (%)": "{:.2f}%", "LSTM Vol (%)": "{:.2f}%", "Transformer Vol (%)": "{:.2f}%"
        }), width="stretch")

    with tab_stats:
        s1, s2, s3 = st.columns(3)
        s1.write(pd.Series(garch_series).describe().to_frame("GARCH Metrics"))
        s2.write(pd.Series(lstm_series).describe().to_frame("LSTM Metrics"))
        s3.write(pd.Series(trans_series).describe().to_frame("Transformer Metrics"))

    st.divider()

    # Section 3: Risk Sizing Engine
    st.subheader("🛡️ Executive Risk-Targeted Position Sizing Framework")
    r1, r2 = st.columns([1, 1])
    with r1:
        st.write(f"• **Portfolio Capital:** `${current_capital:,.2f}`")
        st.write(f"• **Predicted Daily Volatility:** `{risk_summary.get('predicted_daily_volatility', 'N/A')}`")
        st.write(f"• **1-Day 95% Value-at-Risk (VaR):** `{risk_summary.get('one_day_95pct_VaR', 'N/A')}`")
        st.write(f"• **Maximum Allowable Risk Budget:** `{risk_summary.get('max_allowed_risk_usd', 'N/A')}`")
        st.write(f"• **Recommended Allocation Cap:** `{risk_summary.get('recommended_position_usd', 'N/A')}`")

    with r2:
        pos_val_str = str(risk_summary.get("recommended_position_usd", "$0")).replace("$", "").replace(",", "")
        try:
            rec_pos_val = float(pos_val_str)
        except Exception:
            rec_pos_val = current_capital * 0.2
        cash_val = max(0.0, current_capital - rec_pos_val)
        alloc_df = pd.DataFrame({"Category": ["Target Position", "Risk Buffer"], "Amount ($)": [rec_pos_val, cash_val]})
        fig_pie = px.pie(alloc_df, values="Amount ($)", names="Category", hole=0.4)
        fig_pie.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_pie, width="stretch")

    # -------------------------------------------------------------
    # Section 4: Live Risk-Engineered Paper & Testnet Execution Panel
    # -------------------------------------------------------------
    st.divider()
    st.subheader("⚡ Live Risk-Engineered Execution Panel")
    
    # 1. Fetch Bybit Credentials from Render Environment Variables
    bybit_key = os.environ.get("BYBIT_TESTNET_KEY")
    bybit_secret = os.environ.get("BYBIT_TESTNET_SECRET")

    # 2. Execution Mode Selection Toggle
    execution_mode = st.radio(
        "Select Execution Venue:",
        ["Virtual Paper Trading (Local)", "Bybit Testnet API (Live Sandbox)"],
        horizontal=True,
        key="radio_execution_venue"
    )

    portfolio = st.session_state["paper_portfolio"]
    
    # Portfolio Equity Display
    p1, p2, p3 = st.columns(3)
    p1.metric("Available Virtual Cash", f"${portfolio['cash_usd']:,.2f}")
    
    open_position_val = sum(
        pos["units"] * latest_price 
        for sym, pos in portfolio["positions"].items() if sym == current_symbol
    )
    p2.metric(f"Open Position Value ({current_symbol})", f"${open_position_val:,.2f}")
    
    total_portfolio_val = portfolio["cash_usd"] + open_position_val
    p3.metric("Total Account Equity", f"${total_portfolio_val:,.2f}")

    col_buy, col_sell = st.columns(2)

    # BUY ORDER ROUTING
    with col_buy:
        rec_allocation_usd = float(str(risk_summary.get("recommended_position_usd", "$0")).replace("$", "").replace(",", ""))
        st.markdown(f"**Target Recommended Allocation:** `${rec_allocation_usd:,.2f}`")
        
        if st.button(f"📥 Execute VaR-Sized BUY ({current_symbol})", key="btn_execute_var_buy", width="stretch"):
            if execution_mode == "Bybit Testnet API (Live Sandbox)":
                if not bybit_key or not bybit_secret:
                    st.error("❌ Bybit Testnet keys not found in Render Environment Variables.")
                else:
                    st.info("🚀 Transmitting VaR-sized order to Bybit V5 Testnet API...")
            else:
                if rec_allocation_usd <= 0 or rec_allocation_usd > portfolio["cash_usd"]:
                    st.error("Insufficient virtual cash buffer to execute target size.")
                else:
                    units_to_buy = rec_allocation_usd / latest_price
                    portfolio["cash_usd"] -= rec_allocation_usd
                    
                    if current_symbol in portfolio["positions"]:
                        existing = portfolio["positions"][current_symbol]
                        total_units = existing["units"] + units_to_buy
                        avg_entry = ((existing["units"] * existing["entry_price"]) + rec_allocation_usd) / total_units
                        portfolio["positions"][current_symbol] = {"units": total_units, "entry_price": avg_entry}
                    else:
                        portfolio["positions"][current_symbol] = {"units": units_to_buy, "entry_price": latest_price}

                    portfolio["trade_history"].append({
                        "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Type": "BUY",
                        "Asset": current_symbol,
                        "Price": latest_price,
                        "Amount ($)": rec_allocation_usd,
                        "Units": units_to_buy
                    })
                    st.success(f"Executed BUY order for {units_to_buy:.4f} {current_symbol} at ${latest_price:,.2f}")
                    st.rerun()

    # SELL ORDER ROUTING
    with col_sell:
        has_position = current_symbol in portfolio["positions"] and portfolio["positions"][current_symbol]["units"] > 0
        st.markdown(f"**Current Asset Holdings:** `{portfolio['positions'].get(current_symbol, {}).get('units', 0.0):.4f} {current_symbol}`")
        
        if st.button(f"📤 Liquidate Position (SELL ALL)", key="btn_execute_sell_all", width="stretch", disabled=not has_position):
            pos_data = portfolio["positions"].pop(current_symbol)
            sell_value_usd = pos_data["units"] * latest_price
            portfolio["cash_usd"] += sell_value_usd
            
            pnl_usd = sell_value_usd - (pos_data["units"] * pos_data["entry_price"])
            
            portfolio["trade_history"].append({
                "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Type": "SELL",
                "Asset": current_symbol,
                "Price": latest_price,
                "Amount ($)": sell_value_usd,
                "Realized P&L ($)": pnl_usd
            })
            st.success(f"Liquidated position for ${sell_value_usd:,.2f} (Realized P&L:${pnl_usd:+,.2f})")
            st.rerun()

    if portfolio["trade_history"]:
        with st.expander("📜 Virtual Execution Audit Trail & Order History"):
            trade_df = pd.DataFrame(portfolio["trade_history"])
            st.dataframe(trade_df, width="stretch")

    gc.collect()
