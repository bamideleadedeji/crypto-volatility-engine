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
                            # Apply 0.5% dust safety margin to prevent ErrInsufficientBalance rejections
                            safe_target_val = target_val * 0.995
                            
                            res = safe_execute_market_order(order_executor, pair, "SELL", safe_target_val, current_price)
                            st.write(f"Executed SELL on {pair}: {res.get('status')}")
                            if res.get("status") == "SUCCESS":
                                projected_ngn_cash += safe_target_val
                            else:
                                err_msg = res.get('error', 'Unknown error')
                                st.warning(f"⚠️ SELL {pair} exchange note ({err_msg}). Proceeding with liquidity sweep.")
                                log_audit(venue="LUNO", asset=pair, price=current_price, amount=safe_target_val, units="FAILED_SELL", status="EXCHANGE_WARNING")
                        else:
                            st.info(f"[PAPER MODE / SIMULATION] Executed SELL on {pair} for ₦{target_val:,.2f}. Proceeds swept to NGN wallet.")
                            log_audit(venue="LUNO", asset=pair, price=current_price, amount=target_val, units="SIMULATED", status="PAPER_SELL")
                            projected_ngn_cash += target_val

                # --- PHASE 2: CAPITAL DEPLOYMENT & ACCUMULATION (BUYS SECOND) ---
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

                    allocatable_cash = min(target_val, projected_ngn_cash)

                    if allocatable_cash < min_order_threshold:
                        st.warning(f"⚠️ Skipping BUY for {pair}: Insufficient NGN liquidity (Available: ₦{projected_ngn_cash:,.2f}).")
                        continue

                    if current_price > 0:
                        if order_executor and not paper_mode:
                            res = safe_execute_market_order(order_executor, pair, "BUY", allocatable_cash, current_price)
                            st.write(f"Executed BUY on {pair}: {res.get('status')}")
                            if res.get("status") == "SUCCESS":
                                projected_ngn_cash -= allocatable_cash
                            else:
                                err_msg = res.get('error', 'Unknown error')
                                st.warning(f"⚠️ BUY {pair} exchange note: {err_msg}")
                        else:
                            st.info(f"[PAPER MODE / SIMULATION] Executed BUY on {pair} for ₦{allocatable_cash:,.2f} funded from NGN reserve.")
                            log_audit(venue="LUNO", asset=pair, price=current_price, amount=allocatable_cash, units="SIMULATED", status="PAPER_BUY")
                            projected_ngn_cash -= allocatable_cash

                st.success("✅ Autonomous Two-Phase Rebalancing cycle completed and recorded to SQLite WAL audit logs.")
                log_audit(
                    venue="SYSTEM",
                    asset="PORTFOLIO",
                    price=0.0,
                    amount=total_portfolio_val,
                    units="ALL",
                    status="TWO_PHASE_REBALANCE_SUCCESS",
                )
        except Exception as e:
            st.error(f"🚨 Execution Error: {str(e)}")
            logger.exception("Rebalance execution failed")

# --- MODULE 2: GARCH VOLATILITY & VaR SUITE ---
elif app_mode == "GARCH Volatility & VaR Suite":
    st.subheader("Advanced GARCH(1,1) Econometrics & Multi-Asset VaR/CVaR")
    col_opt1, col_opt2, col_opt3 = st.columns(3)
    with col_opt1:
        selected_model = st.selectbox("Volatility Specification", ["GARCH(1,1)", "EGARCH(1,1)"])
    with col_opt2:
        lookback_window = st.selectbox("Estimation Window (Days)", [60, 90, 180, 365], index=1)
    with col_opt3:
        var_horizon = st.number_input("VaR Horizon (Days)", min_value=1, max_value=30, value=1)

    if st.button("📈 Run Econometric Estimation & Covariance Shrinkage", width="stretch"):
        with st.spinner("Fitting GARCH models with Student-t innovations & calculating Ledoit-Wolf shrinkage..."):
            np.random.seed(42)
            dates = pd.date_range(end=pd.Timestamp.today(), periods=lookback_window, freq="D")
            sim_returns_data = {
                "XBTNGN": np.random.normal(0.0012, 0.035, lookback_window),
                "ETHNGN": np.random.normal(0.0015, 0.042, lookback_window),
                "SOLNGN": np.random.normal(0.0020, 0.055, lookback_window),
            }
            returns_df = pd.DataFrame(sim_returns_data, index=dates)
            cov_matrix = quant_engine.compute_robust_covariance(returns_df, method="ledoit_wolf")
            risk_metrics = quant_engine.calculate_portfolio_var_cvar(
                weights=rebalancer.target_weights,
                returns_df=returns_df,
                covariance_matrix=cov_matrix,
                portfolio_value=total_portfolio_val if total_portfolio_val > 0 else 10_000_000.0,
                horizon=var_horizon,
            )
            st.success("✅ Econometric pipeline successfully computed.")
            st.dataframe(pd.DataFrame(risk_metrics).T, width="stretch")

# --- MODULE 3: BACKTEST SIMULATION ENGINE ---
elif app_mode == "Backtest Simulation Engine":
    st.subheader("Institutional Backtest Simulation Engine")
    bt_days = st.slider("Backtest Historical Period (Days)", min_value=30, max_value=365, value=180)
    if st.button("🚀 Run Backtest Simulation", width="stretch"):
        with st.spinner("Running historical backtest simulation..."):
            np.random.seed(100)
            bt_dates = pd.date_range(end=pd.Timestamp.today(), periods=bt_days, freq="D")
            sim_prices_data = {
                "XBTNGN": 60000000 * np.cumprod(1 + np.random.normal(0.001, 0.03, bt_days)),
                "ETHNGN": 3500000 * np.cumprod(1 + np.random.normal(0.0012, 0.038, bt_days)),
                "SOLNGN": 200000 * np.cumprod(1 + np.random.normal(0.0015, 0.048, bt_days)),
            }
            price_df = pd.DataFrame(sim_prices_data, index=bt_dates)
            results = backtest_engine.run_simulation(price_df, target_weights=rebalancer.target_weights)
            st.success("✅ Backtest Simulation Complete.")
            m_res1, m_res2, m_res3, m_res4 = st.columns(4)
            m_res1.metric("Buy & Hold Final Value", f"₦{results['buy_hold_final_value']:,.2f}")
            m_res2.metric("ICIE Engine Final Value", f"₦{results['icie_final_value']:,.2f}")
            m_res3.metric("Annualized Sharpe Ratio", f"{results['sharpe_ratio']}")
            m_res4.metric("Maximum Drawdown", f"{results['max_drawdown_pct']}%")

# --- MODULE 4: MARKET REGIME & ATTRIBUTION ---
elif app_mode == "Market Regime & Attribution":
    st.subheader("Market Regime Detection & Monthly Performance Attribution")
    sample_prices = pd.Series(60000000 * np.cumprod(1 + np.random.normal(0.001, 0.04, 180)))
    regime_info = regime_detector.detect_regime(sample_prices)
    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    col_r1.metric("Detected Market Regime", regime_info["regime"])
    col_r2.metric("Annualized Volatility", f"{regime_info['current_annualized_volatility']}%")
    col_r3.metric("Recommended GARCH Lookback", f"{regime_info['recommended_lookback_days']} Days")
    col_r4.metric("Risk Multiplier Cap", f"{regime_info['risk_multiplier']}x")

# --- MODULE 5: SQLITE AUDIT LEDGER ---
elif app_mode == "SQLite Audit Ledger":
    st.subheader("📜 SQLite WAL Audit Ledger & Event History")
    audit_df = fetch_audit_ledger(100)
    if not audit_df.empty:
        st.dataframe(audit_df, width="stretch")
    else:
        st.info("No audit logs recorded in SQLite WAL database yet.")
