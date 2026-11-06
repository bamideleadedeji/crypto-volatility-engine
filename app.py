import csv
from datetime import datetime
import gc
import os
import sys
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from requests.auth import HTTPBasicAuth
import streamlit as st
import torch

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
matplotlib.use("Agg")

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

from src.data_loader import fetch_crypto_data
from src.garch_model import fit_garch_model
from src.risk_engine import calculate_position_size
from risk_guards import InstitutionalRiskGuards

risk_guard = InstitutionalRiskGuards(max_slippage_pct=0.015, max_drawdown_pct=0.05, fee_buffer_pct=0.005)

# Safe imports for modular enterprise engines
try:
    from alert_engine import check_risk_alerts, send_telegram_alert
    HAS_ALERT_ENGINE = True
except ImportError:
    HAS_ALERT_ENGINE = False

try:
    from asymmetric_garch import fit_egarch, fit_gjr_garch
    HAS_ASYMMETRIC_GARCH = True
except ImportError:
    HAS_ASYMMETRIC_GARCH = False

try:
    from backtester import run_var_backtest
    HAS_BACKTESTER = True
except ImportError:
    run_var_backtest = None
    HAS_BACKTESTER = False

try:
    from portfolio_risk import calculate_portfolio_var, fetch_multi_asset_returns
    HAS_PORTFOLIO_VAR = True
except ImportError:
    HAS_PORTFOLIO_VAR = False

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
# Luno Spot Wallet-Aware Live Market Fetcher (Zero Hardcoding)
# ----------------------------------------------------------------------
def get_luno_portfolio_value() -> dict:
    """Fetches real-time live balances from Luno Spot Trading Wallets and live exchange tickers."""
    try:
        btc_res = requests.get("https://api.luno.com/api/1/ticker?pair=XBTNGN", timeout=5)
        eth_res = requests.get("https://api.luno.com/api/1/ticker?pair=ETHNGN", timeout=5)
        sol_res = requests.get("https://api.luno.com/api/1/ticker?pair=SOLNGN", timeout=5)

        btc_price_ngn = float(btc_res.json().get("last_trade", 0.0)) if btc_res.status_code == 200 else 0.0
        eth_price_ngn = float(eth_res.json().get("last_trade", 0.0)) if eth_res.status_code == 200 else 0.0
        sol_price_ngn = float(sol_res.json().get("last_trade", 0.0)) if sol_res.status_code == 200 else 0.0

        luno_key = st.secrets.get("LUNO_API_KEY_ID") or os.environ.get("LUNO_API_KEY_ID")
        luno_secret = st.secrets.get("LUNO_API_KEY_SECRET") or os.environ.get("LUNO_API_KEY_SECRET")

        ngn_bal = 0.0
        btc_bal = 0.0
        eth_bal = 0.0
        sol_bal = 0.0

        if luno_key and luno_secret:
            auth = HTTPBasicAuth(luno_key, luno_secret)
            bal_res = requests.get("https://api.luno.com/api/1/balance", auth=auth, timeout=8)
            if bal_res.status_code == 200:
                b_data = bal_res.json().get("balance", [])
                for b in b_data:
                    asset_name = b.get("asset")
                    bal_val = float(b.get("balance", 0.0))
                    if asset_name == "NGN":
                        ngn_bal += bal_val
                    elif asset_name in ["XBT", "BTC"]:
                        btc_bal += bal_val
                    elif asset_name == "ETH":
                        eth_bal += bal_val
                    elif asset_name == "SOL":
                        sol_bal += bal_val
            else:
                raise Exception(f"Luno Balance API Error: HTTP {bal_res.status_code} - {bal_res.text}")
        else:
            raise Exception("Luno API Credentials not found in Streamlit Secrets.")

        total_crypto_value = (btc_bal * btc_price_ngn) + (eth_bal * eth_price_ngn) + (sol_bal * sol_price_ngn)
        total_ngn_value = ngn_bal + total_crypto_value

        return {
            "btc_price_ngn": btc_price_ngn,
            "eth_price_ngn": eth_price_ngn,
            "sol_price_ngn": sol_price_ngn,
            "portfolio_ngn": total_ngn_value,
            "crypto_ngn": total_crypto_value,
            "ngn_balance": ngn_bal,
            "btc_balance": btc_bal,
            "eth_balance": eth_bal,
            "sol_balance": sol_bal,
            "status": "LIVE LUNO SPOT CONNECTED",
        }
    except Exception as e:
        st.error(f"🚨 Live Portfolio Fetch Exception: {str(e)}")
        return {
            "btc_price_ngn": 0.0,
            "eth_price_ngn": 0.0,
            "sol_price_ngn": 0.0,
            "portfolio_ngn": 0.0,
            "crypto_ngn": 0.0,
            "ngn_balance": 0.0,
            "btc_balance": 0.0,
            "eth_balance": 0.0,
            "sol_balance": 0.0,
            "status": f"ERROR: {str(e)}",
        }


def append_isolated_audit_log(venue, asset, price_str, amount_str, units_str, status):
    try:
        log_filename = "luno_live_pilot_audit.csv"
        file_exists = os.path.isfile(log_filename)
        with open(log_filename, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Timestamp", "Venue", "Asset", "Price", "Amount", "Units", "Status"])
            writer.writerow([
                pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                venue,
                asset,
                price_str,
                amount_str,
                units_str,
                status,
            ])
    except Exception as log_err:
        print(f"Audit log write exception: {log_err}")


def load_and_verify_audit_csvs():
    filepath = "luno_live_pilot_audit.csv"
    if os.path.isfile(filepath):
        try:
            return pd.read_csv(filepath)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame(columns=["Timestamp", "Venue", "Asset", "Price", "Amount", "Units", "Status"])


# ----------------------------------------------------------------------
# Page Configuration & Theme Setup
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Luno Live Real-Capital Rebalancing & Volatility Suite",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔴 Luno Live Real-Capital Rebalancing & Volatility Suite")
st.caption("Dedicated Luno Spot Trading Wallet Execution Engine — Strict 50% BTC / 30% ETH / 20% SOL Target Allocation")
st.divider()

luno_data = get_luno_portfolio_value()

l_col1, l_col2, l_col3, l_col4 = st.columns(4)
l_col1.metric("Total Luno Portfolio Value", f"₦{luno_data['portfolio_ngn']:,.2f}")
l_col2.metric("Liquid NGN Cash Balance", f"₦{luno_data['ngn_balance']:,.2f}")
l_col3.metric("BTC Holdings", f"{luno_data['btc_balance']:.8f} BTC")
l_col4.metric("ETH / SOL Holdings", f"{luno_data['eth_balance']:.4f} ETH | {luno_data['sol_balance']:.4f} SOL")

st.divider()

# Sidebar Controls
st.sidebar.header("🕹️ Institutional Controls")
symbol = st.sidebar.selectbox("Asset Identifier", ["BTC-USD", "ETH-USD", "SOL-USD"], index=0, key="selected_asset")
capital = st.sidebar.number_input("Portfolio Capital (NGN)", value=float(luno_data["portfolio_ngn"]) if luno_data["portfolio_ngn"] > 0 else 1000.0, step=500.0, format="%.2f")
epochs = st.sidebar.number_input("Neural Network Epochs", value=10, min_value=5, max_value=20, step=5)

luno_key = st.secrets.get("LUNO_API_KEY_ID") or os.environ.get("LUNO_API_KEY_ID")
luno_secret = st.secrets.get("LUNO_API_KEY_SECRET") or os.environ.get("LUNO_API_KEY_SECRET")

st.sidebar.divider()
if luno_key and luno_secret:
    st.sidebar.success(f"🔴 Luno Spot API Connected (`{luno_key[:4]}...`)")
else:
    st.sidebar.warning("🔴 Luno Spot API: Credentials Missing in Secrets")


@st.cache_data(ttl=1800, show_spinner=False)
def load_cached_market_data(ticker_symbol: str) -> pd.DataFrame:
    data = fetch_crypto_data(symbol=ticker_symbol)
    if data is not None and not data.empty:
        float_cols = data.select_dtypes(include=["float64"]).columns
        data[float_cols] = data[float_cols].astype("float32")
    return data


if "analytics_results" not in st.session_state:
    st.session_state["analytics_results"] = None

run_analytics = st.sidebar.button("🚀 Execute Quantitative Analytics", key="btn_run_analytics", use_container_width=True)

if run_analytics or st.session_state["analytics_results"] is None:
    with st.spinner("Retrieving market data & fitting volatility models..."):
        df = load_cached_market_data(symbol)

        _, garch_vol = fit_garch_model(df["Log_Return"])
        garch_vol = pd.Series(garch_vol, index=df.index[-len(garch_vol):], dtype="float32") if isinstance(garch_vol, np.ndarray) else garch_vol.astype("float32")

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

        if HAS_LSTM:
            try:
                _, lstm_vol = train_lstm_volatility(df["Log_Return"], epochs=int(epochs))
            except Exception:
                lstm_vol = garch_vol
        else:
            lstm_vol = garch_vol

        if HAS_TRANSFORMER:
            try:
                _, trans_vol = train_transformer_volatility(df["Log_Return"], epochs=int(epochs))
            except Exception:
                trans_vol = garch_vol
        else:
            trans_vol = garch_vol

        min_len = min(len(garch_vol), len(gjr_vol), len(egarch_vol), len(lstm_vol), len(trans_vol))
        latest_price_usd = float(df["Close"].iloc[-1])
        latest_garch_vol = float(garch_vol.iloc[-1])
        latest_gjr_vol = float(gjr_vol.iloc[-1]) if HAS_ASYMMETRIC_GARCH else latest_garch_vol

        risk_summary = calculate_position_size(account_balance=capital, predicted_volatility=latest_gjr_vol)
        gc.collect()

        st.session_state["analytics_results"] = {
            "df": df,
            "latest_price": latest_price_usd,
            "latest_garch_vol": latest_garch_vol,
            "latest_gjr_vol": latest_gjr_vol,
            "risk_summary": risk_summary,
            "symbol": symbol,
            "capital": capital,
        }

if st.session_state["analytics_results"] is not None:
    res = st.session_state["analytics_results"]
    df = res["df"]
    latest_price_usd = res["latest_price"]
    latest_garch_vol = res["latest_garch_vol"]
    latest_gjr_vol = res["latest_gjr_vol"]
    risk_summary = res["risk_summary"]

    usd_ngn_fx = luno_data["btc_price_ngn"] / 82000.0 if luno_data["btc_price_ngn"] > 0 else 1500.0
    display_spot = luno_data["btc_price_ngn"] if "BTC" in symbol and luno_data["btc_price_ngn"] > 0 else latest_price_usd * usd_ngn_fx

    st.subheader("📌 Executive Risk & Volatility Dashboard")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Spot Price", f"₦{display_spot:,.2f}")
    m2.metric("GARCH(1,1) Vol", f"{latest_garch_vol*100:.2f}%")
    m3.metric("GJR-GARCH Vol", f"{latest_gjr_vol*100:.2f}%")
    m4.metric("1-Day Parametric VaR", f"₦{risk_summary.get('one_day_95pct_VaR', '0').replace('$', '')}")

    st.divider()

    # -------------------------------------------------------------
    # Section 4: Luno Live Autonomous Internal Rebalance Engine (50/30/20)
    # -------------------------------------------------------------
    st.subheader("⚡ Section 4: Luno Live Autonomous Internal Rebalancing Engine (50% BTC / 30% ETH / 20% SOL)")
    st.caption("Executes live spot market orders on Luno Spot Trading Wallets using form-encoded POST data (`data=payload`), strict base/quote volume formatting, and the ₦50 exchange minimum threshold.")

    TARGET_WEIGHTS = {"XBTNGN": 0.50, "ETHNGN": 0.30, "SOLNGN": 0.20}

    col_w1, col_w2, col_w3 = st.columns(3)
    col_w1.metric("XBTNGN Target", "50%")
    col_w2.metric("ETHNGN Target", "30%")
    col_w3.metric("SOLNGN Target", "20%")

    if st.button("🔄 Execute Live Luno Rebalance", key="btn_luno_rebalance", use_container_width=True):
        try:
            total_portfolio_val = float(luno_data["portfolio_ngn"])
            available_cash = float(luno_data["ngn_balance"])

            asset_prices = {
                "XBTNGN": luno_data["btc_price_ngn"],
                "ETHNGN": luno_data["eth_price_ngn"],
                "SOLNGN": luno_data["sol_price_ngn"]
            }

            current_holdings_val = {
                "XBTNGN": luno_data["btc_balance"] * asset_prices["XBTNGN"],
                "ETHNGN": luno_data["eth_balance"] * asset_prices["ETHNGN"],
                "SOLNGN": luno_data["sol_balance"] * asset_prices["SOLNGN"]
            }

            action_logs = []
            MIN_LUNO_ORDER_NGN = 50.0  # Luno exact exchange minimum order threshold (₦50)
            max_trades_allowed = 3
            trades_executed_count = 0

            # Step 1: Execute Sells on Overweight Assets (Base Asset Volume)
            for asset, target_pct in TARGET_WEIGHTS.items():
                if trades_executed_count >= max_trades_allowed:
                    break

                target_val = total_portfolio_val * target_pct
                current_val = current_holdings_val.get(asset, 0.0)
                delta = target_val - current_val  # Negative = overweight

                if abs(delta) > MIN_LUNO_ORDER_NGN and delta < 0:
                    excess_val = abs(delta)
                    spot_price = asset_prices[asset]
                    if spot_price <= 0:
                        continue
                    
                    if "XBT" in asset:
                        vol_format = f"{excess_val / spot_price:.6f}"
                    elif "ETH" in asset:
                        vol_format = f"{excess_val / spot_price:.4f}"
                    else:
                        vol_format = f"{excess_val / spot_price:.2f}"

                    order_status = "SUCCESS"
                    order_id_str = "MARKET_SELL"

                    if luno_key and luno_secret:
                        auth = HTTPBasicAuth(luno_key, luno_secret)
                        payload = {"pair": asset, "type": "SELL", "volume": vol_format}
                        res = requests.post("https://api.luno.com/api/1/marketorder", auth=auth, data=payload, timeout=10)
                        if res.status_code == 200 and res.text and res.text.strip():
                            r_data = res.json()
                            order_id_str = r_data.get("order_id", "SOLD")
                            trades_executed_count += 1
                            available_cash += excess_val
                        else:
                            order_status = f"HTTP_{res.status_code}"
                            order_id_str = res.text[:150] if res.text else "Error"
                    else:
                        trades_executed_count += 1
                        available_cash += excess_val

                    append_isolated_audit_log("Luno Spot Live", asset, f"₦{spot_price:,.2f}", f"₦{excess_val:,.2f}", vol_format, f"LIVE SELL [{order_status}]: {order_id_str}")
                    action_logs.append(f"🔄 LIVE SELL: Trimmed {asset} by ₦{excess_val:,.2f} ({vol_format} units). Status: {order_status} ({order_id_str})")
                    time.sleep(1)

            # Step 2: Execute Buys on Underweight Assets (Quote Asset Counter-Volume NGN >= ₦50)
            for asset, target_pct in TARGET_WEIGHTS.items():
                if trades_executed_count >= max_trades_allowed:
                    break

                target_val = total_portfolio_val * target_pct
                current_val = current_holdings_val.get(asset, 0.0)
                delta = target_val - current_val  # Positive = underweight

                if delta > MIN_LUNO_ORDER_NGN and available_cash >= MIN_LUNO_ORDER_NGN:
                    execution_amount = min(delta, available_cash)

                    order_status = "SUCCESS"
                    order_id_str = "MARKET_BUY"

                    if luno_key and luno_secret:
                        auth = HTTPBasicAuth(luno_key, luno_secret)
                        payload = {"pair": asset, "type": "BUY", "counter_volume": f"{execution_amount:.2f}"}
                        res = requests.post("https://api.luno.com/api/1/marketorder", auth=auth, data=payload, timeout=10)
                        if res.status_code == 200 and res.text and res.text.strip():
                            r_data = res.json()
                            order_id_str = r_data.get("order_id", "BOUGHT")
                            trades_executed_count += 1
                            available_cash -= execution_amount
                        else:
                            order_status = f"HTTP_{res.status_code}"
                            order_id_str = res.text[:150] if res.text else "Error"
                    else:
                        trades_executed_count += 1
                        available_cash -= execution_amount

                    append_isolated_audit_log("Luno Spot Live", asset, "Market", f"₦{execution_amount:,.2f}", "CounterNGN", f"LIVE BUY [{order_status}]: {order_id_str}")
                    action_logs.append(f"✅ LIVE BUY: Deployed ₦{execution_amount:,.2f} into {asset}. Status: {order_status} ({order_id_str})")
                    time.sleep(1)

            if not action_logs:
                st.success("✅ Portfolio allocation is balanced or order sizes are below Luno's ₦50 minimum threshold.")
                append_isolated_audit_log("Luno Spot Live", "PORTFOLIO", "N/A", "N/A", "N/A", "BALANCED_OR_BELOW_MIN")
            else:
                st.success(f"📊 Live Luno Rebalance Complete — Executed Trades: {trades_executed_count}\n\n" + "\n".join(action_logs))

        except Exception as live_err:
            st.error(f"🚨 Live Rebalance Execution Exception: {str(live_err)}")
            append_isolated_audit_log("Luno Spot Live", "SYSTEM", "N/A", "N/A", "N/A", f"EXCEPTION: {str(live_err)}")

    st.markdown("### 📜 Persistent Luno Audit Ledger Inspection")
    audit_df = load_and_verify_audit_csvs()
    if not audit_df.empty:
        st.dataframe(audit_df, use_container_width=True)
    else:
        st.info("No audit logs recorded yet.")

    # -------------------------------------------------------------
    # Section 5: Multi-Asset Portfolio VaR & Covariance Matrix
    # -------------------------------------------------------------
    st.divider()
    st.header("🌐 Multi-Asset Portfolio VaR & Covariance Matrix")
    if HAS_PORTFOLIO_VAR:
        if st.button("🚀 Calculate Covariance & VaR", use_container_width=True):
            basket_symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]
            weights_dict = {"BTC-USD": 0.5, "ETH-USD": 0.3, "SOL-USD": 0.2}
            multi_returns = fetch_multi_asset_returns(symbols=basket_symbols, data_fetcher_func=load_cached_market_data)
            p_results = calculate_portfolio_var(returns_df=multi_returns, weights=weights_dict, portfolio_capital=float(capital), confidence_level=0.95)
            if p_results.get("status") == "SUCCESS":
                st.success(f"1-Day 95% Portfolio VaR: ₦{p_results['portfolio_var_usd']:,.2f}")
                st.dataframe(p_results["allocation_breakdown"], use_container_width=True)

    # -------------------------------------------------------------
    # Section 6: Historical VaR Backtesting & Performance Audit Module
    # -------------------------------------------------------------
    st.divider()
    st.header("📈 Historical VaR Backtesting & Performance Audit Module")
    if run_var_backtest is not None:
        if st.button("🚀 Run Backtest Analysis", use_container_width=True):
            results = run_var_backtest(df=df, initial_capital=float(capital), confidence_level=0.95, max_risk_pct=0.02)
            if results.get("status") == "SUCCESS":
                st.success(f"Strategy Final Capital: ₦{results['final_strategy_equity']:,} ({results['strategy_total_return_pct']}%)")
                st.line_chart(results["backtest_df"][["Strategy_Equity", "Buy_Hold_Equity"]])
