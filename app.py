import sys
import os
import requests

# ----------------------------------------------------------------------
# 0. Global Path & Environment Configuration (Must execute first)
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Disable Matplotlib font cache building on cold boot
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
import matplotlib
matplotlib.use("Agg")

# Threading & Memory Guards for Cloud Containers
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import torch
torch.set_num_threads(1)

import gc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from src.data_loader import fetch_crypto_data
from src.garch_model import fit_garch_model
from src.risk_engine import calculate_position_size

# Safe import for Automated Risk Alert Engine
try:
    from alert_engine import check_risk_alerts, send_telegram_alert
    HAS_ALERT_ENGINE = True
except ImportError:
    try:
        from src.alert_engine import check_risk_alerts, send_telegram_alert
        HAS_ALERT_ENGINE = True
    except ImportError:
        HAS_ALERT_ENGINE = False

# Safe import for Asymmetric Econometric Models (GJR-GARCH & EGARCH)
try:
    from asymmetric_garch import fit_gjr_garch, fit_egarch
    HAS_ASYMMETRIC_GARCH = True
except ImportError:
    try:
        from src.asymmetric_garch import fit_gjr_garch, fit_egarch
        HAS_ASYMMETRIC_GARCH = True
    except ImportError:
        HAS_ASYMMETRIC_GARCH = False

# Safe import for Backtester Module
try:
    from backtester import run_var_backtest
    HAS_BACKTESTER = True
except ImportError:
    try:
        from src.backtester import run_var_backtest
        HAS_BACKTESTER = True
    except ImportError:
        run_var_backtest = None
        HAS_BACKTESTER = False

# Safe import for Multi-Asset Portfolio Risk Module
try:
    from portfolio_risk import fetch_multi_asset_returns, calculate_portfolio_var
    HAS_PORTFOLIO_VAR = True
except ImportError:
    try:
        from src.portfolio_risk import fetch_multi_asset_returns, calculate_portfolio_var
        HAS_PORTFOLIO_VAR = True
    except ImportError:
        HAS_PORTFOLIO_VAR = False

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

# Safe import for Bybit Testnet Execution Engine
try:
    from src.bybit_execution import execute_bybit_testnet_order
    HAS_BYBIT_EXEC = True
except Exception:
    HAS_BYBIT_EXEC = False

# Safe import for Luno Ticker Integration
def get_luno_portfolio_value(btc_holdings: float = 0.00035) -> dict:
    try:
        url = "https://api.luno.com/api/1/ticker?pair=XBTNGN"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        last_price_ngn = float(data.get("last_trade", 0.0))
        total_ngn_value = btc_holdings * last_price_ngn if last_price_ngn > 0 else 5623.00
        return {"btc_price_ngn": last_price_ngn, "portfolio_ngn": total_ngn_value, "status": "SUCCESS"}
    except Exception as e:
        return {"btc_price_ngn": 0.0, "portfolio_ngn": 5623.00, "status": f"FALLBACK: {str(e)}"}

# ----------------------------------------------------------------------
# Page Configuration & Theme Setup
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

# Header & Metadata
st.title("📈 Crypto Volatility Forecasting & Quantitative Risk Suite")
st.caption("Institutional Multi-Model Benchmarking Engine (GARCH vs. GJR-GARCH vs. EGARCH vs. LSTM vs. Transformer) with Automated Risk Alerts")

# Live Luno Account Tracker Bar
luno_data = get_luno_portfolio_value()
luno_col1, luno_col2, luno_col3 = st.columns(3)
luno_col1.metric("Luno BTC/NGN Spot Ticker", f"₦{luno_data['btc_price_ngn']:,.2f}" if luno_data['btc_price_ngn'] > 0 else "₦0.00")
luno_col2.metric("Active Luno Portfolio Value", f"₦{luno_data['portfolio_ngn']:,.2f}")
luno_col3.metric("Micro-Capital Allocation", "100% BTC")

st.divider()

# Sidebar Controls
st.sidebar.header("🕹️ Institutional Controls")
symbol = st.sidebar.selectbox(
    "Asset Identifier", ["BTC-USD", "ETH-USD", "SOL-USD"], index=0
)

currency_mode = st.sidebar.radio("Base Currency Context", ["USD ($)", "NGN (₦)"])
curr_symbol = "₦" if currency_mode == "NGN (₦)" else "$"

if currency_mode == "NGN (₦)":
    capital = st.sidebar.number_input("Portfolio Capital (NGN)", value=5623.00, step=500.0, format="%.2f")
    default_alert_var = round(capital * 0.035, 2)
    alert_var_limit = st.sidebar.number_input("Max Allowable 1-Day VaR (NGN)", value=default_alert_var, step=10.0)
else:
    capital = st.sidebar.number_input("Portfolio Capital ($USD)", value=10000.0, step=1000.0, format="%.2f")
    alert_var_limit = st.sidebar.number_input("Max Allowable 1-Day VaR ($USD)", value=500.0, step=50.0)

confidence_level = st.sidebar.slider(
    "VaR Confidence Level (%)", min_value=90.0, max_value=99.0, value=95.0, step=1.0
) / 100.0
epochs = st.sidebar.number_input(
    "Neural Network Epochs", value=10, min_value=5, max_value=20, step=5
)

st.sidebar.divider()
st.sidebar.header("🚨 Automated Risk Alert Limits")
alert_vol_limit = st.sidebar.slider("Max Allowable Daily Volatility (%)", 1.0, 10.0, 5.0, 0.5)

st.sidebar.divider()
st.sidebar.info("💡 **Consultant Note:** Benchmarks symmetric & asymmetric econometrics against deep learning architectures to optimize risk-weighted capital allocation.")

# ----------------------------------------------------------------------
# Cached Data Fetcher Function
# ----------------------------------------------------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def load_cached_market_data(ticker_symbol: str) -> pd.DataFrame:
    """Fetches market data with a 30-minute TTL cache to avoid hitting API rate limits."""
    data = fetch_crypto_data(symbol=ticker_symbol)
    if data is not None and not data.empty:
        float_cols = data.select_dtypes(include=['float64']).columns
        data[float_cols] = data[float_cols].astype('float32')
    return data

# ----------------------------------------------------------------------
# Main Execution Engine (Session State Persistence)
# ----------------------------------------------------------------------
if "analytics_results" not in st.session_state:
    st.session_state["analytics_results"] = None

run_analytics = st.sidebar.button("🚀 Execute Quantitative Analytics", key="btn_run_analytics", use_container_width=True)

if run_analytics or st.session_state["analytics_results"] is None:
    with st.spinner("Retrieving market data feed, parameterizing econometric & neural models..."):
        
        # Load Market Data (With Auto-Fallback)
        df = load_cached_market_data(symbol)

        # 1. Fit Standard GARCH(1,1) Baseline
        _, garch_vol = fit_garch_model(df["Log_Return"])
        if isinstance(garch_vol, np.ndarray):
            garch_vol = pd.Series(garch_vol, index=df.index[-len(garch_vol):], dtype="float32")
        else:
            garch_vol = garch_vol.astype("float32")

        # 2. Fit Asymmetric Econometric Models (GJR-GARCH & EGARCH)
        if HAS_ASYMMETRIC_GARCH:
            try:
                gjr_params, gjr_vol = fit_gjr_garch(df["Log_Return"])
                egarch_params, egarch_vol = fit_egarch(df["Log_Return"])
            except Exception:
                gjr_params, gjr_vol = {}, garch_vol
                egarch_params, egarch_vol = {}, garch_vol
        else:
            gjr_params, gjr_vol = {}, garch_vol
            egarch_params, egarch_vol = {}, garch_vol

        # 3. Fit PyTorch Deep LSTM
        if HAS_LSTM:
            try:
                _, lstm_vol = train_lstm_volatility(df["Log_Return"], epochs=int(epochs))
                if isinstance(lstm_vol, np.ndarray):
                    lstm_vol = pd.Series(lstm_vol, index=df.index[-len(lstm_vol):], dtype="float32")
                else:
                    lstm_vol = lstm_vol.astype("float32")
            except Exception:
                lstm_vol = garch_vol
        else:
            lstm_vol = garch_vol

        # 4. Fit Multi-Head Transformer
        if HAS_TRANSFORMER:
            try:
                _, trans_vol = train_transformer_volatility(df["Log_Return"], epochs=int(epochs))
                if isinstance(trans_vol, np.ndarray):
                    trans_vol = pd.Series(trans_vol, index=df.index[-len(trans_vol):], dtype="float32")
                else:
                    trans_vol = trans_vol.astype("float32")
            except Exception:
                trans_vol = garch_vol
        else:
            trans_vol = garch_vol

        # Data Alignment
        min_len = min(len(garch_vol), len(gjr_vol), len(egarch_vol), len(lstm_vol), len(trans_vol))
        plot_dates = df.index[-min_len:]
        
        garch_series = (garch_vol.iloc[-min_len:] * 100).astype("float32")
        gjr_series = (gjr_vol.iloc[-min_len:] * 100).astype("float32")
        egarch_series = (egarch_vol.iloc[-min_len:] * 100).astype("float32")
        lstm_series = (lstm_vol.iloc[-min_len:] * 100).astype("float32")
        trans_series = (trans_vol.iloc[-min_len:] * 100).astype("float32")

        # Latest Spot Metrics
        latest_price = float(df["Close"].iloc[-1])
        latest_garch_vol = float(garch_vol.iloc[-1])
        latest_gjr_vol = float(gjr_vol.iloc[-1]) if HAS_ASYMMETRIC_GARCH else latest_garch_vol

        risk_summary = calculate_position_size(
            account_balance=capital, predicted_volatility=latest_gjr_vol
        )

        gc.collect()

        st.session_state["analytics_results"] = {
            "df": df,
            "garch_series": garch_series,
            "gjr_series": gjr_series,
            "egarch_series": egarch_series,
            "lstm_series": lstm_series,
            "trans_series": trans_series,
            "latest_price": latest_price,
            "latest_garch_vol": latest_garch_vol,
            "latest_gjr_vol": latest_gjr_vol,
            "gjr_params": gjr_params,
            "egarch_params": egarch_params,
            "risk_summary": risk_summary,
            "plot_dates": plot_dates,
            "symbol": symbol,
            "capital": capital
        }

# ----------------------------------------------------------------------
# Dashboard Rendering & Execution Engine
# ----------------------------------------------------------------------
if st.session_state["analytics_results"] is not None:
    res = st.session_state["analytics_results"]
    
    # Safe unpacking
    df = res["df"]
    garch_series = res["garch_series"]
    gjr_series = res.get("gjr_series", garch_series)
    egarch_series = res.get("egarch_series", garch_series)
    lstm_series = res["lstm_series"]
    trans_series = res["trans_series"]
    latest_price = res["latest_price"]
    latest_garch_vol = res["latest_garch_vol"]
    latest_gjr_vol = res.get("latest_gjr_vol", latest_garch_vol)
    risk_summary = res["risk_summary"]
    plot_dates = res["plot_dates"]
    current_symbol = res["symbol"]
    current_capital = res["capital"]

    # Section 1: Executive Key Metrics
    st.subheader("📌 Executive Risk & Volatility Dashboard")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Spot Price", f"${latest_price:,.2f}")
    m2.metric("GARCH(1,1) Vol", f"{latest_garch_vol*100:.2f}%")
    m3.metric("GJR-GARCH Vol", f"{latest_gjr_vol*100:.2f}%")
    m4.metric("1-Day Parametric VaR", f"{curr_symbol}{risk_summary.get('one_day_95pct_VaR', '0').replace('$', '')}")
    m5.metric("Target Max Allocation", f"{curr_symbol}{risk_summary.get('recommended_position_usd', '0').replace('$', '')}")

    # Automated Risk Threshold Audit Banner
    if HAS_ALERT_ENGINE:
        var_str_val = str(risk_summary.get("one_day_95pct_VaR", "$0")).replace("$", "").replace(",", "")
        try:
            var_val_usd = float(var_str_val)
        except Exception:
            var_val_usd = 0.0
            
        alert_res = check_risk_alerts(
            predicted_vol=latest_gjr_vol,
            portfolio_var_usd=var_val_usd,
            vol_threshold_pct=float(alert_vol_limit),
            var_threshold_usd=float(alert_var_limit),
            symbol=current_symbol
        )
        
        if alert_res["status"] == "ALERT":
            st.error(f"🚨 **RISK THRESHOLD BREACH DETECTED:**\n\n" + "\n".join(alert_res["alerts_triggered"]))
        else:
            st.success(f"✅ **Risk Status Normal:** Predicted Volatility ({latest_gjr_vol*100:.2f}%) and 1-Day VaR ({curr_symbol}{var_val_usd:,.2f}) are within specified limits.")

    st.divider()

    # Section 2: Forecast Horizon Plots
    tab_chart, tab_table, tab_stats = st.tabs([
        "📈 Forecast Comparison Plot", 
        "📋 Model Volatility Diagnostics", 
        "📊 Econometric Statistical Audit"
    ])

    with tab_chart:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=plot_dates, y=garch_series, name="GARCH(1,1)", line=dict(color="#1f77b4", width=2)))
        fig.add_trace(go.Scatter(x=plot_dates, y=gjr_series, name="GJR-GARCH (Asymmetric)", line=dict(color="#d62728", width=2.5)))
        fig.add_trace(go.Scatter(x=plot_dates, y=egarch_series, name="EGARCH (Exponential)", line=dict(color="#9467bd", width=2, dash="dot")))
        fig.add_trace(go.Scatter(x=plot_dates, y=lstm_series, name="LSTM Net", line=dict(color="#ff7f0e", width=2, dash="dash")))
        fig.add_trace(go.Scatter(x=plot_dates, y=trans_series, name="Transformer", line=dict(color="#2ca02c", width=2, dash="dashdot")))
        fig.update_layout(
            title=f"Multi-Model Daily Volatility Horizon (%) — {current_symbol}",
            xaxis_title="Timeline", yaxis_title="Conditional Volatility (%)",
            template="plotly_white", height=520
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_table:
        diag_df = pd.DataFrame({
            "Date": plot_dates[-15:].strftime("%Y-%m-%d"),
            "Asset Spot ($)": df["Close"].iloc[-15:].values,
            "Log Return (%)": df["Log_Return"].iloc[-15:].values * 100,
            "GARCH Vol (%)": garch_series.iloc[-15:].values,
            "GJR-GARCH Vol (%)": gjr_series.iloc[-15:].values,
            "EGARCH Vol (%)": egarch_series.iloc[-15:].values,
            "LSTM Vol (%)": lstm_series.iloc[-15:].values,
            "Transformer Vol (%)": trans_series.iloc[-15:].values
        })
        st.dataframe(diag_df.style.format({
            "Asset Spot ($)": "${:,.2f}", "Log Return (%)": "{:+.2f}%",
            "GARCH Vol (%)": "{:.2f}%", "GJR-GARCH Vol (%)": "{:.2f}%", "EGARCH Vol (%)": "{:.2f}%",
            "LSTM Vol (%)": "{:.2f}%", "Transformer Vol (%)": "{:.2f}%"
        }), use_container_width=True)

    with tab_stats:
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.write(pd.Series(garch_series).describe().to_frame("GARCH"))
        s2.write(pd.Series(gjr_series).describe().to_frame("GJR-GARCH"))
        s3.write(pd.Series(egarch_series).describe().to_frame("EGARCH"))
        s4.write(pd.Series(lstm_series).describe().to_frame("LSTM"))
        s5.write(pd.Series(trans_series).describe().to_frame("Transformer"))

    st.divider()

    # Section 3: Risk Sizing Engine & Telegram Alert Dispatch
    st.subheader("🛡️ Executive Risk-Targeted Position Sizing Framework & Alerts")
    r1, r2 = st.columns([1, 1])
    with r1:
        st.write(f"• **Portfolio Capital:** `{curr_symbol}{current_capital:,.2f}`")
        st.write(f"• **Predicted Daily Volatility (GJR-GARCH):** `{risk_summary.get('predicted_daily_volatility', 'N/A')}`")
        st.write(f"• **1-Day 95% Value-at-Risk (VaR):** `{curr_symbol}{str(risk_summary.get('one_day_95pct_VaR', 'N/A')).replace('$', '')}`")
        st.write(f"• **Maximum Allowable Risk Budget:** `{curr_symbol}{str(risk_summary.get('max_allowed_risk_usd', 'N/A')).replace('$', '')}`")
        st.write(f"• **Recommended Allocation Cap:** `{curr_symbol}{str(risk_summary.get('recommended_position_usd', 'N/A')).replace('$', '')}`")

        # Telegram Alert Dispatcher
        if HAS_ALERT_ENGINE:
            tg_token = os.environ.get("TELEGRAM_BOT_TOKEN") or st.secrets.get("TELEGRAM_BOT_TOKEN")
            tg_chat = os.environ.get("TELEGRAM_CHAT_ID") or st.secrets.get("TELEGRAM_CHAT_ID")
            
            if st.button("📲 Dispatch Telegram Risk Alert Payload", key="btn_send_telegram", use_container_width=True):
                with st.spinner("Transmitting risk payload to Telegram API..."):
                    tg_res = send_telegram_alert(
                        bot_token=tg_token,
                        chat_id=tg_chat,
                        message=alert_res["summary_msg"]
                    )
                    if tg_res["status"] == "SUCCESS":
                        st.success("✅ Telegram Risk Alert Sent!")
                    else:
                        st.warning(f"⚠️ Telegram Dispatch Note: {tg_res['message']}")

    with r2:
        pos_val_str = str(risk_summary.get("recommended_position_usd", "$0")).replace("$", "").replace(",", "")
        try:
            rec_pos_val = float(pos_val_str)
        except Exception:
            rec_pos_val = current_capital * 0.2
        cash_val = max(0.0, current_capital - rec_pos_val)
        alloc_df = pd.DataFrame({"Category": ["Target Position", "Risk Buffer"], "Amount": [rec_pos_val, cash_val]})
        fig_pie = px.pie(alloc_df, values="Amount", names="Category", hole=0.4)
        fig_pie.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_pie, use_container_width=True)

    # -------------------------------------------------------------
    # Section 4: Live Risk-Engineered Paper & Testnet Execution Panel
    # -------------------------------------------------------------
    st.divider()
    st.subheader("⚡ Live Risk-Engineered Execution Panel")
    
    bybit_key = os.environ.get("BYBIT_TESTNET_KEY") or st.secrets.get("BYBIT_TESTNET_KEY")
    bybit_secret = os.environ.get("BYBIT_TESTNET_SECRET") or st.secrets.get("BYBIT_TESTNET_SECRET")

    execution_mode = st.radio(
        "Select Execution Venue:",
        ["Virtual Paper Trading (Local)", "Bybit Testnet API (Live Sandbox)"],
        horizontal=True,
        key="radio_execution_venue"
    )

    portfolio = st.session_state["paper_portfolio"]
    
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

    with col_buy:
        rec_allocation_usd = float(str(risk_summary.get("recommended_position_usd", "$0")).replace("$", "").replace(",", ""))
        st.markdown(f"**Target Recommended Allocation:** `${rec_allocation_usd:,.2f}`")
        
        if st.button(f"📥 Execute VaR-Sized BUY ({current_symbol})", key="btn_execute_var_buy", use_container_width=True):
            if execution_mode == "Bybit Testnet API (Live Sandbox)":
                if not bybit_key or not bybit_secret:
                    st.error("❌ Bybit Testnet keys not found in Environment or Streamlit Secrets.")
                elif not HAS_BYBIT_EXEC:
                    st.error("❌ `src/bybit_execution.py` module not found or failed to load.")
                else:
                    with st.spinner("🚀 Transmitting VaR-sized order to Bybit V5 Testnet API..."):
                        exec_res = execute_bybit_testnet_order(
                            api_key=bybit_key,
                            api_secret=bybit_secret,
                            symbol=current_symbol,
                            amount_usd=rec_allocation_usd,
                            current_price=latest_price,
                            side="buy"
                        )
                        
                        if exec_res["status"] == "SUCCESS":
                            st.success(f"✅ Bybit Testnet Order Placed! Order ID: `{exec_res['order_id']}`")
                            portfolio["trade_history"].append({
                                "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "Type": "BUY (Bybit Testnet)",
                                "Asset": current_symbol,
                                "Price": latest_price,
                                "Amount ($)": rec_allocation_usd,
                                "Units": exec_res.get("filled_units", rec_allocation_usd / latest_price)
                            })
                        else:
                            st.error(f"❌ Bybit Execution Failed: {exec_res['message']}")
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
                        "Type": "BUY (Paper)",
                        "Asset": current_symbol,
                        "Price": latest_price,
                        "Amount ($)": rec_allocation_usd,
                        "Units": units_to_buy
                    })
                    st.success(f"Executed BUY order for {units_to_buy:.4f} {current_symbol} at ${latest_price:,.2f}")

    with col_sell:
        has_position = current_symbol in portfolio["positions"] and portfolio["positions"][current_symbol]["units"] > 0
        st.markdown(f"**Current Asset Holdings:** `{portfolio['positions'].get(current_symbol, {}).get('units', 0.0):.4f} {current_symbol}`")
        
        if st.button(f"📤 Liquidate Position (SELL ALL)", key="btn_execute_sell_all", use_container_width=True, disabled=not has_position):
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

    if portfolio["trade_history"]:
        # ==============================================================================
# SECTION 4: CURRENCY-AWARE ORDER EXECUTION & AUDIT LOGGING ENGINE
# ==============================================================================

try:
    # 1. Safely retrieve session variables or set robust defaults
    active_currency_mode = st.session_state.get("currency_mode", "NGN (₦)")
    active_venue = st.session_state.get("venue_toggle", "Paper")
    active_asset = st.session_state.get("selected_asset", "BTC-USD")
    
    # 2. Determine spot price in local currency & establish FX rate guard
    # Default USD/NGN rate fallback if live feed is unavailable
    usd_to_ngn_rate = float(st.session_state.get("usd_ngn_rate", 1500.0))
    
    # Retrieve raw spot price from model state
    raw_spot_price = float(st.session_state.get("spot_price", 0.0))
    position_allocation = float(st.session_state.get("var_position_size", 0.0))

    if "NGN" in active_currency_mode:
        curr_symbol = "₦"
        formatted_asset_pair = active_asset.replace("-USD", "-NGN")
        
        # If the incoming spot price is in USD, convert to NGN
        # Example: $82,214.98 * 1,500 = ₦123,322,470.00 / BTC
        if raw_spot_price < 500000.0:  # Threshold to detect USD price input
            execution_price = raw_spot_price * usd_to_ngn_rate
        else:
            execution_price = raw_spot_price
    else:
        curr_symbol = "$"
        formatted_asset_pair = active_asset.replace("-NGN", "-USD")
        execution_price = raw_spot_price

    # 3. Prevent DivisionByZero & calculate true position units
    if execution_price > 0:
        allocated_units = position_allocation / execution_price
    else:
        allocated_units = 0.0

    # 4. Construct production audit ledger record
    audit_record = {
        "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Type": f"BUY ({active_venue})",
        "Asset": formatted_asset_pair,
        f"Price ({curr_symbol})": round(execution_price, 2),
        f"Amount ({curr_symbol})": round(position_allocation, 2),
        "Units": round(allocated_units, 8)
    }

    # 5. Thread-safe appending to session state ledger
    if "execution_logs" not in st.session_state:
        st.session_state["execution_logs"] = []

    st.session_state["execution_logs"].append(audit_record)

except Exception as e:
        st.warning(f"Audit Logging Warning: {str(e)}")
        with st.expander("📜 Virtual Execution Audit Trail & Order History"):
            trade_df = pd.DataFrame(portfolio["trade_history"])
            st.dataframe(trade_df, use_container_width=True)

    # -------------------------------------------------------------
    # Section 5: Multi-Asset Portfolio VaR & Variance-Covariance Matrix
    # -------------------------------------------------------------
    st.divider()
    st.header("🌐 Multi-Asset Portfolio VaR & Covariance Matrix")
    st.caption("Parametric Portfolio Risk Sizing across BTC, ETH, and SOL using Variance-Covariance Matrices.")

    if not HAS_PORTFOLIO_VAR:
        st.error("⚠️ Module `src/portfolio_risk.py` not found. Please ensure `src/portfolio_risk.py` is committed in GitHub.")
    else:
        with st.expander("⚙️ Multi-Asset Basket Allocation Controls", expanded=True):
            p_col1, p_col2, p_col3 = st.columns(3)
            with p_col1:
                w_btc = st.slider("BTC Allocation (%)", 0, 100, 50, 5) / 100.0
            with p_col2:
                w_eth = st.slider("ETH Allocation (%)", 0, 100, 30, 5) / 100.0
            with p_col3:
                w_sol = st.slider("SOL Allocation (%)", 0, 100, 20, 5) / 100.0

        if st.button("🚀 Calculate Portfolio VaR & Covariance", type="primary", use_container_width=True):
            with st.spinner("Fetching multi-asset price series & constructing Variance-Covariance Matrix..."):
                basket_symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]
                weights_dict = {"BTC-USD": w_btc, "ETH-USD": w_eth, "SOL-USD": w_sol}
                
                multi_returns = fetch_multi_asset_returns(
                    symbols=basket_symbols,
                    data_fetcher_func=load_cached_market_data
                )
                
                p_results = calculate_portfolio_var(
                    returns_df=multi_returns,
                    weights=weights_dict,
                    portfolio_capital=float(current_capital),
                    confidence_level=confidence_level
                )

            if p_results.get("status") == "SUCCESS":
                st.success("✅ Multi-Asset Covariance & VaR Sizing Complete!")

                pv1, pv2, pv3, pv4 = st.columns(4)
                pv1.metric("Portfolio Capital", f"{curr_symbol}{p_results['portfolio_capital']:,.2f}")
                pv2.metric("Portfolio Daily Vol", f"{p_results['portfolio_daily_vol_pct']:.2f}%")
                pv3.metric(f"1-Day {p_results['confidence_level_pct']:.0f}% Portfolio VaR", f"{curr_symbol}{p_results['portfolio_var_usd']:,.2f}")
                pv4.metric(
                    "Diversification Benefit", 
                    f"{curr_symbol}{p_results['diversification_benefit_usd']:,.2f}",
                    delta=f"{p_results['diversification_pct']:.2f}% Risk Reduction"
                )

                tab_p1, tab_p2, tab_p3 = st.tabs([
                    "📊 Portfolio Allocation & Risk Breakdown",
                    "🔥 Asset Correlation Matrix (R)",
                    "📐 Covariance Matrix (Σ)"
                ])

                with tab_p1:
                    st.dataframe(p_results["allocation_breakdown"], use_container_width=True)
                    
                    portfolio_csv = p_results["allocation_breakdown"].to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Portfolio Allocation & VaR Audit Report (CSV)",
                        data=portfolio_csv,
                        file_name="multi_asset_portfolio_var_report.csv",
                        mime="text/csv",
                        key="btn_download_portfolio_csv",
                        use_container_width=True
                    )

                with tab_p2:
                    st.subheader("Cross-Asset Correlation Matrix (R)")
                    fig_corr = px.imshow(
                        p_results["corr_matrix_df"],
                        text_auto=".2f",
                        color_continuous_scale="Blues",
                        title="Asset Return Correlation Matrix"
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)

                with tab_p3:
                    st.subheader("Variance-Covariance Matrix (Σ)")
                    st.dataframe(p_results["cov_matrix_df"].style.format("{:.6f}"), use_container_width=True)

            else:
                st.error(f"❌ Portfolio VaR calculation failed: {p_results.get('message')}")

    # -------------------------------------------------------------
    # Section 6: Historical VaR Backtesting & Performance Audit Module
    # -------------------------------------------------------------
    st.divider()
    st.header("📈 Historical VaR Backtesting & Performance Audit Module")

    if run_var_backtest is None:
        st.error("⚠️ Module `src/backtester.py` not found. Please ensure `src/backtester.py` is present in the repository.")
    else:
        with st.expander("🛠️ Backtest Configuration & Controls", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                bt_capital = st.number_input("Initial Capital", value=float(current_capital), step=1000.0)
            with col2:
                bt_confidence = st.selectbox("VaR Confidence Level", [0.95, 0.99], index=0)
            with col3:
                bt_risk_pct = st.slider("Max Daily Risk Exposure (%)", 0.01, 0.05, 0.02, 0.005)

        if st.button("🚀 Run Backtest Analysis", type="primary", use_container_width=True):
            if df is None or df.empty:
                st.warning("⚠️ Please load market data first in the main dashboard.")
            else:
                with st.spinner("Executing historical backtest and building equity curves..."):
                    results = run_var_backtest(
                        df=df,
                        initial_capital=bt_capital,
                        confidence_level=bt_confidence,
                        max_risk_pct=bt_risk_pct
                    )

                if results.get("status") == "SUCCESS":
                    st.success("✅ Backtest Analysis Completed!")

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Strategy Final Capital", f"{curr_symbol}{results['final_strategy_equity']:,}", f"{results['strategy_total_return_pct']}%")
                    m2.metric("Buy & Hold Capital", f"{curr_symbol}{results['final_bh_equity']:,}", f"{results['buy_hold_total_return_pct']}%")
                    m3.metric("Sharpe Ratio", f"{results['strategy_sharpe']}")
                    m4.metric("Max Drawdown", f"{results['strategy_max_drawdown_pct']}%")

                    st.subheader("📊 Equity Curve: VaR Strategy vs. Buy & Hold")
                    st.line_chart(results["backtest_df"][['Strategy_Equity', 'Buy_Hold_Equity']])

                    st.info(
                        f"🛡️ **VaR Model Audit:** Observed **{results['var_breaches']} breaches** "
                        f"out of {results['total_periods_tested']} periods."
                    )

                    csv_bytes = results["backtest_df"].to_csv().encode('utf-8')
                    st.download_button(
                        label="📥 Download Backtest Audit Report (CSV)",
                        data=csv_bytes,
                        file_name=f"var_backtest_audit_report_{current_symbol}.csv",
                        mime="text/csv"
                    )
                else:
                    st.error(f"❌ Backtest failed: {results.get('message', 'Unknown Error')}")

    gc.collect()
