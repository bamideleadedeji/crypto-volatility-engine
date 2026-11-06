import csv
from datetime import datetime, timedelta
import functools
import gc
import json
import os
import shutil
import sqlite3
import sys
import time
import logging
from logging.handlers import RotatingFileHandler

import arch
import numpy as np
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
import streamlit as st
import torch
import yfinance as yf

# ----------------------------------------------------------------------
# 0. Global Path, Logging & Configuration Management
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

# Professional Rotating Logger Setup (Prevents log bloat)
log_handler = RotatingFileHandler("engine.log", maxBytes=5 * 1024 * 1024, backupCount=3)
log_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

logger = logging.getLogger("LunoInstitutionalEngine")
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(log_handler)

DB_NAME = "luno_institutional_engine.db"
BACKUP_DIR = "db_backups"
CONFIG_FILE = "config.json"

# Default Institutional Configuration
DEFAULT_CONFIG = {
    "target_weights": {"XBTNGN": 0.50, "ETHNGN": 0.30, "SOLNGN": 0.20},
    "drift_tolerance": 0.03,
    "max_failures": 5,
    "cooldown_minutes": 15,
    "min_order_ngn": 50.0,
    "max_trades_per_run": 4
}

def load_or_create_config():
    """Loads configuration from JSON or creates default config file."""
    if not os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "w") as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)
    try:
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return DEFAULT_CONFIG

CONFIG = load_or_create_config()

def init_institutional_database():
    """Initializes SQLite with WAL mode and creates expanded institutional tables."""
    os.makedirs(BACKUP_DIR, exist_ok=True)
    with sqlite3.connect(DB_NAME, timeout=30) as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL;")
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                venue TEXT,
                asset TEXT,
                price REAL,
                amount REAL,
                units TEXT,
                status TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS circuit_breaker_state (
                id INTEGER PRIMARY KEY,
                failure_count INTEGER,
                is_tripped BOOLEAN,
                last_trip_time TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS risk_metrics_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                portfolio_value REAL,
                var_95 REAL,
                cvar_95 REAL,
                max_drift REAL
            )
        ''')
        
        cursor.execute("SELECT COUNT(*) FROM circuit_breaker_state")
        if cursor.fetchone()[0] == 0:
            cursor.execute("INSERT INTO circuit_breaker_state (id, failure_count, is_tripped, last_trip_time) VALUES (1, 0, 0, '')")
        conn.commit()
    logger.info("Database initialized successfully in WAL mode.")

def perform_automated_backup():
    """Performs an automated timestamped backup of the SQLite database."""
    try:
        if os.path.exists(DB_NAME):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(BACKUP_DIR, f"luno_backup_{timestamp}.db")
            shutil.copy2(DB_NAME, backup_path)
            logger.info(f"Automated database backup created at {backup_path}")
    except Exception as e:
        logger.error(f"Database backup failed: {e}")

init_institutional_database()
perform_automated_backup()


# ----------------------------------------------------------------------
# 1. Institutional Risk Manager Class (Fixed Drift Signature)
# ----------------------------------------------------------------------
class RiskManager:
    """Encapsulates self-healing circuit breakers, drift tolerances, and CVaR calculations."""
    
    def __init__(self, config: dict):
        self.config = config
        self.drift_tolerance = config.get("drift_tolerance", 0.03)
        self.max_failures = config.get("max_failures", 5)
        self.cooldown_minutes = config.get("cooldown_minutes", 15)

    def check_circuit_breaker(self) -> bool:
        """Self-healing circuit breaker with automatic recovery cooldown."""
        with sqlite3.connect(DB_NAME, timeout=30) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT failure_count, is_tripped, last_trip_time FROM circuit_breaker_state WHERE id=1")
            row = cursor.fetchone()
            
        if not row:
            return True
            
        failure_count, is_tripped, last_trip_time = row[0], row[1], row[2]
        
        if is_tripped:
            if last_trip_time:
                trip_dt = datetime.strptime(last_trip_time, "%Y-%m-%d %H:%M:%S")
                if datetime.now() - trip_dt > timedelta(minutes=self.cooldown_minutes):
                    logger.info("🔄 Circuit breaker cooldown elapsed. Executing automatic reset.")
                    self.reset_circuit_breaker()
                    return True
            logger.warning("🚨 CIRCUIT BREAKER ACTIVE: Trading halted.")
            return False
        return True

    def record_api_failure(self):
        """Increments failure counter and trips circuit breaker if threshold is reached."""
        with sqlite3.connect(DB_NAME, timeout=30) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT failure_count FROM circuit_breaker_state WHERE id=1")
            count = cursor.fetchone()[0] + 1
            
            is_tripped = 1 if count >= self.max_failures else 0
            trip_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S") if is_tripped else ""
            
            cursor.execute('''
                UPDATE circuit_breaker_state 
                SET failure_count = ?, is_tripped = ?, last_trip_time = ? 
                WHERE id = 1
            ''', (count, is_tripped, trip_time))
            conn.commit()
        if is_tripped:
            logger.error(f"🚨 Circuit breaker tripped after {self.max_failures} consecutive API errors.")

    def reset_circuit_breaker(self):
        """Resets circuit breaker state to normal."""
        with sqlite3.connect(DB_NAME, timeout=30) as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE circuit_breaker_state SET failure_count = 0, is_tripped = 0, last_trip_time = '' WHERE id=1")
            conn.commit()
        logger.info("Circuit breaker successfully reset to operational state.")

    def evaluate_portfolio_drift(self, current_weights: dict, target_weights: dict, drift_tolerance: float = None) -> bool:
        """Evaluates whether portfolio drift exceeds tolerance to avoid fee bleed."""
        tolerance = drift_tolerance if drift_tolerance is not None else self.drift_tolerance
        max_drift = 0.0
        for asset, target_pct in target_weights.items():
            curr_pct = current_weights.get(asset, 0.0)
            drift = abs(curr_pct - target_pct)
            if drift > max_drift:
                max_drift = drift

        logger.info(f"Portfolio Drift Analysis -> Max Drift: {max_drift*100:.2f}% (Threshold: {tolerance*100}%)")
        if max_drift < tolerance:
            logger.info("⏸️ Drift below threshold. Rebalancing skipped.")
            return False
        logger.info("⚡ Drift exceeds threshold. Rebalancing authorized.")
        return True

    def calculate_expected_shortfall(self, returns_series: pd.Series, portfolio_capital: float, confidence: float = 0.95) -> dict:
        """Computes Basel III Expected Shortfall (CVaR)."""
        if returns_series.empty:
            return {"cvar_pct": 0.0, "cvar_ngn": 0.0}
        alpha = 1.0 - confidence
        var_threshold = np.percentile(returns_series, alpha * 100)
        tail_losses = returns_series[returns_series <= var_threshold]
        cvar_pct = -tail_losses.mean() if not tail_losses.empty else abs(var_threshold)
        cvar_ngn = portfolio_capital * cvar_pct
        return {
            "cvar_pct": float(cvar_pct * 100.0),
            "cvar_ngn": float(cvar_ngn)
        }

    def log_trade_audit(self, venue: str, asset: str, price: float, amount: float, units: str, status: str):
        """Thread-safe transactional audit log insert."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(DB_NAME, timeout=30) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO audit_logs (timestamp, venue, asset, price, amount, units, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (timestamp, venue, asset, price, amount, units, status))
            conn.commit()
        logger.info(f"Audit Log Recorded -> {asset} [{status}] Amount: ₦{amount:,.2f}")

risk_manager = RiskManager(CONFIG)


# ----------------------------------------------------------------------
# 2. Institutional Network Retry Decorator with Exponential Backoff
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
                    risk_manager.record_api_failure()
                    if x == retries:
                        raise e
                    sleep_time = backoff_in_seconds * (2 ** (x - 1))
                    time.sleep(sleep_time)
        return wrapper
    return decorator


# ----------------------------------------------------------------------
# 3. Real-Time Luno Live Portfolio & Wallet Fetcher
# ----------------------------------------------------------------------
@retry_with_backoff(retries=3, backoff_in_seconds=2)
def get_luno_live_portfolio_and_wallet() -> dict:
    """Fetches real-time Luno spot wallet balances and live market tickers dynamically."""
    if not risk_manager.check_circuit_breaker():
        raise Exception("Trading suspended due to active circuit breaker.")

    luno_key = st.secrets.get("LUNO_API_KEY_ID") or os.environ.get("LUNO_API_KEY_ID")
    luno_secret = st.secrets.get("LUNO_API_KEY_SECRET") or os.environ.get("LUNO_API_KEY_SECRET")

    if not luno_key or not luno_secret:
        raise ValueError("❌ Luno API credentials missing from environment/secrets.")

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

    risk_manager.reset_circuit_breaker()

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
# 4. Automatic Order-Status Polling (Fill Confirmation)
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


# Safe imports for modular econometric modules
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
# 5. Streamlit UI & Comprehensive Testing Suite
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Luno Institutional Econometric Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚡ Luno Institutional Econometric & Rebalancing Suite")
st.caption("Live Spot Wallet Execution Backed by RiskManager, CVaR, GARCH, Asymmetric GJR/EGARCH, & SQLite WAL Persistence")
st.divider()

try:
    live_data = get_luno_live_portfolio_and_wallet()
    total_capital = live_data["total_portfolio_ngn"]
    available_ngn = live_data["funded_ngn_wallet"]
except Exception as api_err:
    st.error(f"🚨 Luno API Connection Error: {str(api_err)}")
    st.stop()

st.sidebar.header("🕹️ Quantitative Controls")
selected_proxy = st.sidebar.selectbox("Benchmark Proxy", ["BTC-USD", "ETH-USD", "SOL-USD"], index=0)
epochs = st.sidebar.slider("Neural Network Epochs", min_value=5, max_value=30, value=10)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Live Portfolio Capital", f"₦{total_capital:,.2f}")
m2.metric("Liquid NGN Wallet", f"₦{available_ngn:,.2f}")
m3.metric("XBTNGN Spot Price", f"₦{live_data['prices']['XBTNGN']:,.2f}")
m4.metric("ETHNGN Spot Price", f"₦{live_data['prices']['ETHNGN']:,.2f}")

st.divider()

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
    1. **Circuit Breaker & Health Check:** Ensures API stability and self-healing recovery.
    2. **Portfolio Drift Gate (>3%):** Prevents fee bleed from micro-fluctuations.
    3. **Two-Phase Execution:** Sells overweight assets first to generate liquidity, then deploys NGN into underweight assets with order-fill polling.
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

            current_weights = {
                "XBTNGN": holdings_val.get("XBTNGN", 0.0) / portfolio_val if portfolio_val > 0 else 0,
                "ETHNGN": holdings_val.get("ETHNGN", 0.0) / portfolio_val if portfolio_val > 0 else 0,
                "SOLNGN": holdings_val.get("SOLNGN", 0.0) / portfolio_val if portfolio_val > 0 else 0,
            }
            TARGET_WEIGHTS = CONFIG.get("target_weights", {"XBTNGN": 0.50, "ETHNGN": 0.30, "SOLNGN": 0.20})
            drift_tol = CONFIG.get("drift_tolerance", 0.03)

            if not risk_manager.evaluate_portfolio_drift(current_weights, TARGET_WEIGHTS, drift_tolerance=drift_tol):
                st.info("⏸️ Rebalance skipped: Maximum portfolio drift is below the 3% institutional tolerance threshold.")
                risk_manager.log_trade_audit("Luno Engine", "PORTFOLIO", 0.0, 0.0, "N/A", "DRIFT_SKIPPED")
                st.stop()

            action_logs = []
            MIN_LUNO_ORDER_NGN = CONFIG.get("min_order_ngn", 50.0)
            max_trades = CONFIG.get("max_trades_per_run", 4)
            trades_count = 0

            luno_key = st.secrets.get("LUNO_API_KEY_ID") or os.environ.get("LUNO_API_KEY_ID")
            luno_secret = st.secrets.get("LUNO_API_KEY_SECRET") or os.environ.get("LUNO_API_KEY_SECRET")
            auth = HTTPBasicAuth(luno_key, luno_secret)

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

                    risk_manager.log_trade_audit("Luno Spot Live", asset, spot_price, excess_val, vol_format, f"SELL [{status_str}]")
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

                    risk_manager.log_trade_audit("Luno Spot Live", asset, prices[asset], exec_amount, "CounterNGN", f"BUY [{status_str}]")
                    action_logs.append(f"✅ BUY: Deployed ₦{exec_amount:,.2f} into {asset}. Status: {status_str}")
                    time.sleep(1.5)

            if not action_logs:
                st.success("✅ Portfolio allocation is balanced within precise 50/30/20 target thresholds.")
            else:
                st.success(f"📊 Rebalance Execution Complete — Confirmed Fills: {trades_count}\n\n" + "\n".join(action_logs))

        except Exception as e:
            st.error(f"🚨 Execution Exception: {str(e)}")
            risk_manager.log_trade_audit("Luno Engine", "SYSTEM", 0.0, 0.0, "N/A", f"EXCEPTION: {str(e)}")

with tab2:
    st.subheader("🌐 Test 1: Multi-Asset Covariance Matrix & Portfolio VaR (95%)")
    if HAS_PORTFOLIO_VAR and fetch_crypto_data is not None:
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
    if HAS_ASYMMETRIC and fetch_crypto_data is not None:
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
    if HAS_BACKTESTER and run_var_backtest is not None and fetch_crypto_data is not None:
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
    col_l, col_t = st.columns(2)
    with col_l:
        if HAS_LSTM and fetch_crypto_data is not None and st.button("Run LSTM Volatility Train"):
            df_hist = fetch_crypto_data(selected_proxy)
            if df_hist is not None:
                _, lstm_v = train_lstm_volatility(df_hist["Log_Return"], epochs=int(epochs))
                st.success(f"LSTM Volatility Fitted. Latest Vol: {float(lstm_v.iloc[-1])*100:.2f}%")
    with col_t:
        if HAS_TRANSFORMER and fetch_crypto_data is not None and st.button("Run Transformer Volatility Train"):
            df_hist = fetch_crypto_data(selected_proxy)
            if df_hist is not None:
                _, trans_v = train_transformer_volatility(df_hist["Log_Return"], epochs=int(epochs))
                st.success(f"Transformer Volatility Fitted. Latest Vol: {float(trans_v.iloc[-1])*100:.2f}%")

# ----------------------------------------------------------------------
# 6. SQLite Audit Ledger Inspection in Sidebar / Footer
# ----------------------------------------------------------------------
st.divider()
st.subheader("📜 SQLite Database Audit Ledger Inspection")
try:
    with sqlite3.connect(DB_NAME, timeout=30) as conn:
        audit_query_df = pd.read_sql_query("SELECT * FROM audit_logs ORDER BY id DESC LIMIT 50", conn)
    if not audit_query_df.empty:
        st.dataframe(audit_query_df, use_container_width=True)
    else:
        st.info("No audit logs recorded in SQLite database yet.")
except Exception as db_load_err:
    st.info(f"Audit log table initializing: {db_load_err}")
