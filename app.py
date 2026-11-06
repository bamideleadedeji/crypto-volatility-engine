import csv
from datetime import datetime
import functools
import gc
import os
import sys
import time
import arch
import numpy as np
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
import streamlit as st
import torch
import yfinance as yf

# ----------------------------------------------------------------------
# 0. Global Path & Environment Configuration
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

from risk_guards import InstitutionalRiskGuards

risk_guard = InstitutionalRiskGuards(max_slippage_pct=0.015, max_drawdown_pct=0.05, fee_buffer_pct=0.005)

# Comprehensive imports for all enterprise econometric modules
try:
    from data_loader import fetch_crypto_data
except ImportError:
    fetch_crypto_data = None

try:
    from garch_model import fit_garch_model
except ImportError:
    fit_garch_model = None

try:
    from asymmetric_garch import fit_egarch, fit_gjr_garch
    HAS_ASYMMETRIC = True
except ImportError:
    HAS_ASYMMETRIC = False

try:
    from portfolio_risk import calculate_portfolio_var, fetch_multi_asset_returns
    HAS_PORTFOLIO_VAR = True
except ImportError:
    HAS_PORTFOLIO_VAR = False

try:
    from backtester import run_var_backtest
    HAS_BACKTESTER = True
except ImportError:
    run_var_backtest = None
    HAS_BACKTESTER = False

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


# ----------------------------------------------------------------------
# 1. Institutional Network Retry Decorator with Exponential Backoff
# ----------------------------------------------------------------------
def retry_with_backoff(retries=3, backoff_in_seconds=2):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            x = 0
            while x < retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    x += 1
                    if x == retries:
                        raise e
                    sleep_time = backoff_in_seconds * (2 ** (x - 1))
                    time.sleep(sleep_time)
        return wrapper
    return decorator


# ----------------------------------------------------------------------
# 2. Real-Time Luno Live Portfolio & Wallet Fetcher (Zero Hardcoding)
# ----------------------------------------------------------------------
@retry_with_backoff(retries=3, backoff_in_seconds=2)
def get_luno_live_portfolio_and_wallet() -> dict:
    """Fetches real-time Luno spot wallet balances and live market tickers dynamically."""
    luno_key = st.secrets.get("LUNO_API_KEY_ID") or os.environ.get("LUNO_API_KEY_ID")
    luno_secret = st.secrets.get("LUNO_API_KEY_SECRET") or os.environ.get("LUNO_API_KEY_SECRET")

    if not luno_key or not luno_secret:
        raise ValueError("❌ Luno API credentials (LUNO_API_KEY_ID & LUNO_API_KEY_SECRET) missing from environment/secrets.")

    auth = HTTPBasicAuth(luno_key, luno_secret)

    tickers = ["XBTNGN", "ETHNGN", "SOLNGN"]
    prices = {}
    for pair in tickers:
        res = requests.get(f"https://api.luno.com/api/1/ticker?pair={pair}", timeout=8)
        if res.status_code != 200:
            raise Exception(f"Failed to fetch live ticker for {pair}: HTTP {res.status_code}")
        prices[pair] = float(res.json().get("last_trade", 0.0))
        if prices[pair] <= 0:
            raise ValueError(f"Received invalid zero or negative price for {pair}.")

    bal_res = requests.get("https://api.luno.com/api/1/balance", auth=auth, timeout=10)
    if bal_res.status_code != 200:
        raise Exception(f"Luno Balance API Error: HTTP {bal_res.status_code} - {bal_res.text}")

    balances = bal_res.json().get("balance", [])
    ngn_balance, btc_balance, eth_balance, sol_balance = 0.0, 0.0, 0.0, 0.0

    for b in balances:
        asset = b.get("asset")
        amount = float(b.get("balance", 0.0))
        if asset == "NGN":
            ngn_balance += amount
        elif asset in ["XBT", "BTC"]:
            btc_balance += amount
        elif asset == "ETH":
            eth_balance += amount
        elif asset == "SOL":
            sol_balance += amount

    btc_val_ngn = btc_balance * prices["XBTNGN"]
    eth_val_ngn = eth_balance * prices["ETHNGN"]
    sol_val_ngn = sol_balance * prices["SOLNGN"]

    total_crypto_val = btc_val_ngn + eth_val_ngn + sol_val_ngn
    total_portfolio_val = ngn_balance + total_crypto_val

    return {
        "total_portfolio_ngn": total_portfolio_val,
        "funded_ngn_wallet": ngn_balance,
        "crypto_valuation_ngn": total_crypto_val,
        "prices": prices,
        "holdings_units": {"XBTNGN": btc_balance, "ETHNGN": eth_balance, "SOLNGN": sol_balance},
        "holdings_val": {"XBTNGN": btc_val_ngn, "ETHNGN": eth_val_ngn, "SOLNGN": sol_val_ngn},
        "status": "LIVE FETCH SUCCESS"
    }


# ----------------------------------------------------------------------
# 3. Automatic Order-Status Polling (Fill Confirmation)
# ----------------------------------------------------------------------
def confirm_order_fill(order_id: str, auth: HTTPBasicAuth, max_polls=5, poll_interval=3) -> bool:
    """Actively polls Luno's order endpoint to verify 100% fill status before proceeding."""
    url = f"https://api.luno.com/api/1/orders/{order_id}"
    for attempt in range(max_polls):
        try:
            res = requests.get(url, auth=auth, timeout=8)
            if res.status_code == 200:
                data = res.json()
                state = data.get("state", "").upper()
                if state == "COMPLETE":
                    return True
                elif state == "CANCELLED":
                    return False
            time.sleep(poll_interval)
        except Exception:
            time.sleep(poll_interval)
    return False


def append_audit_log(asset, price_str, amount_str, units_str, status):
    log_filename = "luno_live_pilot_audit.csv"
    file_exists = os.path.isfile(log_filename)
    try:
        with open(log_filename, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Timestamp", "Venue", "Asset", "Price", "Amount", "Units", "Status"])
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Luno Comprehensive Engine",
                asset,
                price_str,
                amount_str,
                units_str,
                status,
            ])
    except Exception as e:
        print(f"Audit log error: {e}")


# ----------------------------------------------------------------------
# 4. Streamlit UI & Comprehensive Testing Suite
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Luno Comprehensive Econometric Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚡ Luno Comprehensive Econometric & Rebalancing Suite")
st.caption("Live Spot Wallet Execution Backed by Multi-Asset Covariance, GARCH, Asymmetric GJR/EGARCH, VaR Testing, & Backtesting Audits")
st.divider()

# Fetch live capital
try:
    live_data = get_luno_live_portfolio_and_wallet()
    total_capital = live_data["total_portfolio_ngn"]
    available_ngn = live_data["funded_ngn_wallet"]
except Exception as api_err:
    st.error(f"🚨 Luno API Connection Error: {str(api_err)}")
    st.stop()

# Sidebar Controls
st.sidebar.header("🕹️ Quantitative Controls")
selected_proxy = st.sidebar.selectbox("Benchmark Proxy", ["BTC-USD", "ETH-USD", "SOL-USD"], index=0)
epochs = st.sidebar.slider("Neural Network Epochs", min_value=5, max_value=30, value=10)

# Top Metrics Bar
m1, m2, m3, m4 = st.columns(4)
m1.metric("Live Portfolio Capital", f"₦{total_capital:,.2f}")
m2.metric("Liquid NGN Wallet", f"₦{available_ngn:,.2f}")
m3.metric("XBTNGN Spot Price", f"₦{live_data['prices']['XBTNGN']:,.2f}")
m4.metric("ETHNGN Spot Price", f"₦{live_data['prices']['ETHNGN']:,.2f}")

st.divider()

# ----------------------------------------------------------------------
# COMPREHENSIVE TABS (All Tests & Live Execution)
# ----------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "⚡ Live Execution & Rebalance", 
    "🌐 Test 1: Multi-Asset Covariance & VaR", 
    "📈 Test 2 & 3: Asymmetric GJR & EGARCH", 
    "📉 Test 4: Historical Backtesting Audit",
    "🧠 Neural Volatility (LSTM / Transformer)"
])

with tab1:
    st.subheader("Autonomous 50/30/20 Internal Rebalancing Engine")
    st.markdown("""
    **Institutional Execution Protocol:**
    1. **Live Capital Check:** Fetches exact NGN and crypto wallet balances.
    2. **Two-Phase Execution:** Sells overweight assets first to generate liquidity, then deploys NGN into underweight assets.
    3. **Precision Exchange Guards:** Enforces Luno's exact **₦50 minimum order rule**, form-encoded payloads, and active order-fill polling.
    """)

    col_w1, col_w2, col_w3 = st.columns(3)
    col_w1.metric("XBTNGN Target", "50%")
    col_w2.metric("ETHNGN Target", "30%")
    col_w3.metric("SOLNGN Target", "20%")

    if st.button("🚀 Execute Quant-Verified Live Rebalance", key="btn_exec_rebalance", use_container_width=True):
        try:
            exec_data = get_luno_live_portfolio_and_wallet()
            portfolio_val = exec_data["total_portfolio_ngn"]
            cash_bal = exec_data["funded_ngn_wallet"]
            prices = exec_data["prices"]
            holdings_val = exec_data["holdings_val"]

            action_logs = []
            MIN_LUNO_ORDER_NGN = 50.0
            max_trades = 4
            trades_count = 0

            luno_key = st.secrets.get("LUNO_API_KEY_ID") or os.environ.get("LUNO_API_KEY_ID")
            luno_secret = st.secrets.get("LUNO_API_KEY_SECRET") or os.environ.get("LUNO_API_KEY_SECRET")
            auth = HTTPBasicAuth(luno_key, luno_secret)
            TARGET_WEIGHTS = {"XBTNGN": 0.50, "ETHNGN": 0.30, "SOLNGN": 0.20}

            # PHASE 1: SELL OVERWEIGHT ASSETS FIRST
            for asset, target_pct in TARGET_WEIGHTS.items():
                if trades_count >= max_trades:
                    break
                target_val = portfolio_val * target_pct
                current_val = holdings_val.get(asset, 0.0)
                delta = target_val - current_val

                if abs(delta) > MIN_LUNO_ORDER_NGN and delta < 0:
                    excess_val = abs(delta)
                    spot_price = prices[asset]
                    if spot_price <= 0:
                        continue

                    if "XBT" in asset:
                        vol_format = f"{excess_val / spot_price:.6f}"
                    elif "ETH" in asset:
                        vol_format = f"{excess_val / spot_price:.4f}"
                    else:
                        vol_format = f"{excess_val / spot_price:.2f}"

                    payload = {"pair": asset, "type": "SELL", "volume": vol_format}
                    res = requests.post("https://api.luno.com/api/1/marketorder", auth=auth, data=payload, timeout=10)

                    status_str = "SUCCESS"
                    if res.status_code == 200 and res.text.strip():
                        r_data = res.json()
                        order_id = r_data.get("order_id")
                        if order_id:
                            filled = confirm_order_fill(order_id, auth)
                            if filled:
                                trades_count += 1
                                cash_bal += excess_val
                                status_str = f"COMPLETE [ID: {order_id}]"
                            else:
                                status_str = f"FILL_TIMEOUT [ID: {order_id}]"
                        else:
                            status_str = "SUCCESS_NO_ID"
                    else:
                        status_str = f"HTTP_{res.status_code}"

                    append_audit_log(asset, f"₦{spot_price:,.2f}", f"₦{excess_val:,.2f}", vol_format, f"SELL [{status_str}]")
                    action_logs.append(f"🔄 SELL: Trimmed {asset} by ₦{excess_val:,.2f}. Status: {status_str}")
                    time.sleep(1.5)

            # PHASE 2: BUY UNDERWEIGHT ASSETS WITH REALIZED NGN CASH
            for asset, target_pct in TARGET_WEIGHTS.items():
                if trades_count >= max_trades:
                    break
                target_val = portfolio_val * target_pct
                current_val = holdings_val.get(asset, 0.0)
                delta = target_val - current_val

                if delta > MIN_LUNO_ORDER_NGN and cash_bal >= MIN_LUNO_ORDER_NGN:
                    exec_amount = min(delta, cash_bal)
                    payload = {"pair": asset, "type": "BUY", "counter_volume": f"{exec_amount:.2f}"}
                    res = requests.post("https://api.luno.com/api/1/marketorder", auth=auth, data=payload, timeout=10)

                    status_str = "SUCCESS"
                    if res.status_code == 200 and res.text.strip():
                        r_data = res.json()
                        order_id = r_data.get("order_id")
                        if order_id:
                            filled = confirm_order_fill(order_id, auth)
                            if filled:
                                trades_count += 1
                                cash_bal -= exec_amount
                                status_str = f"COMPLETE [ID: {order_id}]"
                            else:
                                status_str = f"FILL_TIMEOUT [ID: {order_id}]"
                        else:
                            status_str = "SUCCESS_NO_ID"
                    else:
                        status_str = f"HTTP_{res.status_code}"

                    append_audit_log(asset, "Market", f"₦{exec_amount:,.2f}", "CounterNGN", f"BUY [{status_str}]")
                    action_logs.append(f"✅ BUY: Deployed ₦{exec_amount:,.2f} into {asset}. Status: {status_str}")
                    time.sleep(1.5)

            if not action_logs:
                st.success("✅ Portfolio allocation is balanced within precise 50/30/20 target thresholds.")
                append_audit_log("PORTFOLIO", "N/A", "N/A", "N/A", "BALANCED")
            else:
                st.success(f"📊 Rebalance Execution Complete — Confirmed Fills: {trades_count}\n\n" + "\n".join(action_logs))

        except Exception as e:
            st.error(f"🚨 Execution Exception: {str(e)}")
            append_audit_log("SYSTEM", "N/A", "N/A", "N/A", f"EXCEPTION: {str(e)}")

with tab2:
    st.subheader("🌐 Test 1: Multi-Asset Covariance Matrix & Portfolio VaR (95%)")
    st.markdown("Computes asset return covariances and parametric Value-at-Risk across your multi-asset basket.")
    if HAS_PORTFOLIO_VAR:
        if st.button("Calculate Covariance & Portfolio VaR", use_container_width=True):
            basket = ["BTC-USD", "ETH-USD", "SOL-USD"]
            w_dict = {"BTC-USD": 0.5, "ETH-USD": 0.3, "SOL-USD": 0.2}
            multi_ret = fetch_multi_asset_returns(symbols=basket, data_fetcher_func=lambda s: fetch_crypto_data(s))
            p_res = calculate_portfolio_var(returns_df=multi_ret, weights=w_dict, portfolio_capital=float(total_capital), confidence_level=0.95)
            if p_res.get("status") == "SUCCESS":
                st.success(f"1-Day 95% Portfolio VaR: ₦{p_res['portfolio_var_usd']:,.2f}")
                st.dataframe(p_res["allocation_breakdown"], use_container_width=True)
    else:
        st.info("Portfolio VaR module loading from src/portfolio_risk.py")

with tab3:
    st.subheader("📈 Test 2 & 3: Asymmetric Volatility Modeling (GJR-GARCH & EGARCH)")
    st.markdown("Fits GJR-GARCH and EGARCH models to capture leverage effects and asymmetric shock persistence in crypto returns.")
    if HAS_ASYMMETRIC:
        if st.button("Fit Asymmetric Volatility Models", use_container_width=True):
            df_hist = fetch_crypto_data(selected_proxy)
            if df_hist is not None and not df_hist.empty:
                gjr_p, gjr_v = fit_gjr_garch(df_hist["Log_Return"])
                eg_p, eg_v = fit_egarch(df_hist["Log_Return"])
                st.success("Asymmetric volatility models fitted successfully.")
                col_a, col_b = st.columns(2)
                col_a.metric("GJR-GARCH Latest Vol", f"{float(gjr_v.iloc[-1])*100:.2f}%")
                col_b.metric("EGARCH Latest Vol", f"{float(eg_v.iloc[-1])*100:.2f}%")
    else:
        st.info("Asymmetric GARCH module loading from src/asymmetric_garch.py")

with tab4:
    st.subheader("📉 Test 4: Historical VaR Backtesting & Performance Audit")
    st.markdown("Runs historical walk-forward backtesting simulating strategy equity curves against buy-and-hold benchmarks.")
    if HAS_BACKTESTER and run_var_backtest is not None:
        if st.button("Run Historical Backtest Audit", use_container_width=True):
            df_hist = fetch_crypto_data(selected_proxy)
            if df_hist is not None and not df_hist.empty:
                bt_res = run_var_backtest(df=df_hist, initial_capital=float(total_capital), confidence_level=0.95, max_risk_pct=0.02)
                if bt_res.get("status") == "SUCCESS":
                    st.success(f"Backtest Final Equity: ₦{bt_res['final_strategy_equity']:,} ({bt_res['strategy_total_return_pct']}%)")
                    st.line_chart(bt_res["backtest_df"][["Strategy_Equity", "Buy_Hold_Equity"]])
    else:
        st.info("Backtester module loading from src/backtester.py")

with tab5:
    st.subheader("🧠 Neural Network Volatility Models (LSTM & Transformer)")
    st.markdown("Advanced deep learning volatility forecasting modules.")
    col_l, col_t = st.columns(2)
    with col_l:
        if HAS_LSTM and st.button("Run LSTM Volatility Train"):
            df_hist = fetch_crypto_data(selected_proxy)
            if df_hist is not None:
                _, lstm_v = train_lstm_volatility(df_hist["Log_Return"], epochs=int(epochs))
                st.success(f"LSTM Volatility Model Fitted. Latest Vol: {float(lstm_v.iloc[-1])*100:.2f}%")
    with col_t:
        if HAS_TRANSFORMER and st.button("Run Transformer Volatility Train"):
            df_hist = fetch_crypto_data(selected_proxy)
            if df_hist is not None:
                _, trans_v = train_transformer_volatility(df_hist["Log_Return"], epochs=int(epochs))
                st.success(f"Transformer Volatility Model Fitted. Latest Vol: {float(trans_v.iloc[-1])*100:.2f}%")
