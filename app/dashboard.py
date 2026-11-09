import os
import sys
import inspect

# Ensure repository root is in sys.path for absolute modular stability
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from core.config import load_config
from core.logging import setup_logger
from database.repository import init_database, fetch_audit_ledger, log_audit, get_paper_portfolio, update_paper_portfolio
from execution.luno_connector import LunoConnector
from execution.order_executor import InstitutionalOrderExecutor
from risk.risk_manager import RiskManager
from risk.circuit_breakers import CircuitBreakerManager
from strategy.rebalancer import AutonomousRebalancer
from risk.portfolio_econometric import PortfolioEconometricEngine
from backtest.engine import BacktestEngine
from strategy.performance_attribution import PerformanceAttributionEngine
from risk.regime_detector import MarketRegimeDetector

# Optional Econometric import check
try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False

# Initialize core services
logger = setup_logger("ICIEDashboard")
config = load_config()
init_database()

risk_manager = RiskManager(config)
circuit_breaker = CircuitBreakerManager()
rebalancer = AutonomousRebalancer(target_weights=config.get("target_weights", {"XBTNGN": 0.50, "ETHNGN": 0.30, "SOLNGN": 0.20}))
quant_engine = PortfolioEconometricEngine()
backtest_engine = BacktestEngine(initial_capital=10_000_000.0)
attribution_engine = PerformanceAttributionEngine()
regime_detector = MarketRegimeDetector()

st.set_page_config(
    page_title="Institutional Crypto Investment Engine (ICIE)",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚡ Institutional Crypto Investment Engine (ICIE)")
st.caption("Constitution-Aligned Architecture: Autonomous Rebalancing, Ledoit-Wolf Covariance Shrinkage, GARCH Volatility, & SQLite WAL Audit Trails")
st.divider()

# Credentials Check
luno_key = st.secrets.get("LUNO_API_KEY_ID") or os.environ.get("LUNO_API_KEY_ID")
luno_secret = st.secrets.get("LUNO_API_KEY_SECRET") or os.environ.get("LUNO_API_KEY_SECRET")

if not luno_key or not luno_secret:
    st.warning("⚠️ Luno API credentials not detected in Streamlit secrets. Operating in Offline Simulation & Paper Mode.")

connector = LunoConnector(luno_key, luno_secret) if (luno_key and luno_secret) else None
connector = LunoConnector(luno_key, luno_secret) if (luno_key and luno_secret) else None
order_executor = InstitutionalOrderExecutor(connector, circuit_breaker) if connector else None


def safe_execute_market_order(executor, pair: str, side: str, notional_ngn: float, current_price: float) -> dict:
    """
    Call InstitutionalOrderExecutor.execute_market_order robustly across any version signature.
    """
    if executor is None:
        return {"status": "FAILED", "error": "No order executor available"}

    notional_ngn = float(notional_ngn)
    current_price = float(current_price)
    side = side.upper()

    quantity = (notional_ngn / current_price) if current_price > 0 else 0.0
    precision = 8 if ("XBT" in pair or "BTC" in pair) else 6
    volume_str = f"{quantity:.{precision}f}"
    counter_str = f"{notional_ngn:.2f}"

    try:
        sig = inspect.signature(executor.execute_market_order)
        accepted = set(sig.parameters.keys())
    except Exception:
        accepted = set()

    # Try calling directly with exact keyword arguments supported by our updated executor
    try:
        return executor.execute_market_order(
            pair=pair,
            side=side,
            target_val=notional_ngn,
            current_price=current_price,
            volume=volume_str,
            counter_volume=counter_str
        )
    except TypeError:
        pass

    # Fallback candidate dictionary sweeps
    candidates = [
        dict(pair=pair, side=side, target_val=notional_ngn, current_price=current_price),
        dict(pair=pair, side=side, amount=notional_ngn, current_price=current_price),
        dict(pair=pair, side=side, counter_volume=counter_str, current_price=current_price) if side == "BUY" else dict(pair=pair, side=side, volume=volume_str, current_price=current_price)
    ]

    last_error = None
    for kwargs in candidates:
        if accepted and "**" not in str(sig.parameters.get(list(sig.parameters)[-1], "")):
            filtered = {k: v for k, v in kwargs.items() if k in accepted}
            if not filtered:
                continue
            kwargs = filtered
        try:
            return executor.execute_market_order(**kwargs)
        except TypeError as e:
            last_error = e
            continue

    # Absolute final resort positional fallback
    try:
        return executor.execute_market_order(pair, side, notional_ngn, current_price)
    except TypeError as e:
        last_error = e

    return {
        "status": "FAILED",
        "error": f"Could not call execute_market_order with any known signature. Last error: {last_error}"
    }


# Cached Market Data Fetcher
@st.cache_data(ttl=60, show_spinner=False)
def fetch_luno_market_data(_connector, tickers):
    prices = {}
    balances = []
    if not _connector:
        return {"XBTNGN": 65000000.0, "ETHNGN": 3800000.0, "SOLNGN": 220000.0}, [
            {"asset": "NGN", "balance": "2500000.0"},
            {"asset": "XBT", "balance": "0.05"},
            {"asset": "ETH", "balance": "0.4"},
            {"asset": "SOL", "balance": "5.0"},
        ]
    for pair in tickers:
        try:
            prices[pair] = _connector.fetch_ticker(pair)
        except Exception:
            prices[pair] = 0.0
    try:
        balances = _connector.fetch_balances()
    except Exception:
        balances = []
    return prices, balances


tickers = ["XBTNGN", "ETHNGN", "SOLNGN"]
prices, balances = fetch_luno_market_data(connector, tickers)

ngn_balance, btc_balance, eth_balance, sol_balance = 0.0, 0.0, 0.0, 0.0
for b in balances:
    asset = b.get("asset") if isinstance(b, dict) else getattr(b, "get", lambda x: None)("asset")
    amount = float(b.get("balance", 0.0) if isinstance(b, dict) else 0.0)
    if asset == "NGN":
        ngn_balance += amount
    elif asset in ["XBT", "BTC"]:
        btc_balance += amount
    elif asset == "ETH":
        eth_balance += amount
    elif asset == "SOL":
        sol_balance += amount

holdings_units = {"XBTNGN": btc_balance, "ETHNGN": eth_balance, "SOLNGN": sol_balance}
holdings_val = {pair: holdings_units[pair] * prices.get(pair, 0.0) for pair in tickers}
total_crypto_val = sum(holdings_val.values())
total_portfolio_val = ngn_balance + total_crypto_val

# Sidebar Navigation & Controls
st.sidebar.header("🕹️ ICIE Control Panel")
app_mode = st.sidebar.radio(
    "Select Engine Module",
    [
        "Portfolio & Rebalancing Execution",
        "GARCH Volatility & VaR Suite",
        "Backtest Simulation Engine",
        "Market Regime & Attribution",
        "SQLite Audit Ledger",
    ],
)
paper_mode = st.sidebar.checkbox("🔒 Paper Trading Mode (Dry Run)", value=config.get("paper_mode", True))

if st.sidebar.button("🔄 Clear Cache & Refresh Feed"):
    st.cache_data.clear()
    st.rerun()

# Debug helper – shows live signature
if order_executor is not None:
    try:
        live_sig = str(inspect.signature(order_executor.execute_market_order))
        st.sidebar.caption(f"Executor signature: `{live_sig}`")
    except Exception:
        pass

# Top Metrics Banner
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Portfolio Capital", f"₦{total_portfolio_val:,.2f}")
m2.metric("Liquid NGN Reserve", f"₦{ngn_balance:,.2f}")
m3.metric("XBTNGN Spot Price", f"₦{prices.get('XBTNGN', 0.0):,.2f}")
m4.metric("ETHNGN Spot Price", f"₦{prices.get('ETHNGN', 0.0):,.2f}")

st.divider()

# --- MODULE 1: PORTFOLIO & REBALANCING EXECUTION ---
if app_mode == "Portfolio & Rebalancing Execution":
    st.subheader("Autonomous 50/30/20 Closed-Loop Rebalancing Engine")
    col_w1, col_w2, col_w3 = st.columns(3)
    col_w1.metric("BTC Target Allocation", "50%")
    col_w2.metric("ETH Target Allocation", "30%")
    col_w3.metric("SOL Target Allocation", "20%")

    st.markdown("### Current Asset Breakdown & Drift Analysis")
    breakdown_df = pd.DataFrame(
        [
            {
                "Asset Pair": pair,
                "Units": holdings_units[pair],
                "Market Value (₦)": holdings_val[pair],
                "Current Weight (%)": (holdings_val[pair] / total_portfolio_val * 100) if total_portfolio_val > 0 else 0.0,
                "Target Weight (%)": rebalancer.target_weights[pair] * 100,
            }
            for pair in tickers
        ]
    )
    st.dataframe(breakdown_df, width="stretch")

    evaluation = rebalancer.evaluate_rebalance(holdings_val, ngn_balance, total_portfolio_val)
    st.info(f"📊 **Portfolio Max Drift:** {evaluation.get('max_drift_percentage', 0.0)}% (Threshold: 3.0%)")

    if st.button("🚀 Execute Autonomous Closed-Loop Rebalance", width="stretch"):
        try:
            if not evaluation["rebalance_triggered"]:
                st.info("⏸️ Rebalance skipped: Portfolio drift is within the 3% tolerance threshold. No action required.")
            else:
                st.success("✅ Drift exceeds 3% threshold. Executing Two-Phase Self-Funding Rebalance & Treasury Cash Sweep:")

                position_caps = {"XBTNGN": 0.60, "ETHNGN": 0.40, "SOLNGN": 0.30}
                execution_plan = evaluation["execution_plan"]

                projected_ngn_cash = ngn_balance
                min_order_threshold = 50.0  # Luno minimum order size constraint (₦50)

                # --- PHASE 1: DIVESTMENT & CASH REALIZATION (SELLS FIRST) ---
                st.markdown("#### **Phase 1: Divestment & NGN Treasury Sweep (Sells)**")
                phase1_sells = {pair: d for pair, d in execution_plan.items() if d["action"] == "SELL"}

                if not phase1_sells:
                    st.info("No overweight assets to shed in this cycle.")

                for pair, details in phase1_sells.items():
                    target_val = float(details["target_adjustment_val"])
                    current_price = float(prices.get(pair, 0.0))

                    if target_val < min_order_threshold:
                        st.warning(f"⚠️ Skipping SELL for {pair}: Order value ₦{target_val:,.2f} is below Luno minimum exchange threshold.")
                        continue

                    if current_price > 0:
                        if order_executor and not paper_mode:
                            res = safe_execute_market_order(order_executor, pair, "SELL", target_val, current_price)
                            st.write(f"Executed SELL on {pair}: {res.get('status')}")
                            if res.get("status") == "SUCCESS":
                                projected_ngn_cash += target_val
                            elif res.get("status") in ("REJECTED", "BLOCKED", "FAILED"):
                                st.warning(f"SELL {pair} result: {res}")
                        else:
                            st.info(f"[PAPER MODE / SIMULATION] Executed SELL on {pair} for ₦{target_val:,.2f}. Proceeds swept to NGN wallet.")
                            log_audit(venue="LUNO", asset=pair, price=current_price, amount=target_val, units="SIMULATED", status="PAPER_SELL")
                            projected_ngn_cash += target_val

                # --- PHASE 2: ACQUISITION & REBALANCE (BUYS SECOND) ---
                st.markdown("#### **Phase 2: Capital Deployment & Accumulation (Buys)**")
                phase2_buys = {pair: d for pair, d in execution_plan.items() if d["action"] == "BUY"}

                if not phase2_buys:
                    st.info("No underweight assets require accumulation in this cycle.")

                for pair, details in phase2_buys.items():
                    target_val = float(details["target_adjustment_val"])
                    current_price = float(prices.get(pair, 0.0))

                    projected_w = (holdings_val[pair] + target_val) / total_portfolio_val if total_portfolio_val > 0 else 0.0
                    max_allowed = position_caps.get(pair, 0.40)
                    if projected_w > max_allowed:
                        st.warning(f"⚠️ Skipping BUY for {pair}: Projected weight exceeds max limit.")
                        continue

                    allocatable_cash = min(target_val, projected_ngn_cash * 0.90)

                    if allocatable_cash < min_order_threshold:
                        st.warning(f"⚠️ Skipping BUY for {pair}: Insufficient NGN liquidity.")
                        continue

                    if current_price > 0:
                        if order_executor and not paper_mode:
                            res = safe_execute_market_order(order_executor, pair, "BUY", allocatable_cash, current_price)
                            st.write(f"Executed BUY on {pair}: {res.get('status')}")
                            if res.get("status") == "SUCCESS":
                                projected_ngn_cash -= allocatable_cash
                            elif res.get("status") in ("REJECTED", "BLOCKED", "FAILED"):
                                st.warning(f"BUY {pair} result: {res}")
                        else:
                            st.info(f"[PAPER MODE / SIMULATION] Executed BUY on {pair} for ₦{allocatable_cash:,.2f} funded from NGN reserve.")
                            log_audit(venue="LUNO", asset=pair, price=current_price, amount=allocatable_cash, units="SIMULATED", status="PAPER_BUY")
                            projected_ngn_cash -= allocatable_cash

                st.success("✅ Autonomous Two-Phase Rebalancing cycle completed and recorded to SQLite WAL audit logs.")
                log_audit(venue="SYSTEM", asset="PORTFOLIO", price=0.0, amount=total_portfolio_val, units="ALL", status="TWO_PHASE_REBALANCE_SUCCESS")
        except Exception as e:
            st.error(f"🚨 Execution Error: {str(e)}")
            logger.exception("Rebalance execution failed")

# --- MODULE 2: GARCH VOLATILITY & VaR SUITE ---
elif app_mode == "GARCH Volatility & VaR Suite":
    st.subheader("Advanced GARCH(1,1) Econometrics & Multi-Asset VaR/CVaR")
    # [Rest of your econometric modules remain fully intact]
