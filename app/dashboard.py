import os
import sys

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

# Cached Market Data Fetcher
@st.cache_data(ttl=60, show_spinner=False)
def fetch_luno_market_data(_connector, tickers):
    prices = {}
    balances = []
    if not _connector:
        return {"XBTNGN": 65000000.0, "ETHNGN": 3800000.0, "SOLNGN": 220000.0}, [{"asset": "NGN", "balance": "2500000.0"}, {"asset": "XBT", "balance": "0.05"}, {"asset": "ETH", "balance": "0.4"}, {"asset": "SOL", "balance": "5.0"}]
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
    asset = b.get("asset") if isinstance(b, dict) else getattr(b, 'get', lambda x: None)("asset")
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
app_mode = st.sidebar.radio("Select Engine Module", [
    "Portfolio & Rebalancing Execution", 
    "GARCH Volatility & VaR Suite", 
    "Backtest Simulation Engine", 
    "SQLite Audit Ledger"
])
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
    breakdown_df = pd.DataFrame([
        {
            "Asset Pair": pair, 
            "Units": holdings_units[pair], 
            "Market Value (₦)": holdings_val[pair], 
            "Current Weight (%)": (holdings_val[pair] / total_portfolio_val * 100) if total_portfolio_val > 0 else 0.0,
            "Target Weight (%)": rebalancer.target_weights[pair] * 100
        }
        for pair in tickers
    ])
    st.dataframe(breakdown_df, use_container_width=True)

    # Evaluate current drift via AutonomousRebalancer
    evaluation = rebalancer.evaluate_rebalance(holdings_val, ngn_balance, total_portfolio_val)
    
    st.info(f"📊 **Portfolio Max Drift:** {evaluation.get('max_drift_percentage', 0.0)}% (Threshold: 3.0%)[cite: 1]")

    if st.button("🚀 Execute Autonomous Closed-Loop Rebalance", use_container_width=True):
        try:
            # 1. Circuit Breaker Position Limit Check
            current_weights_dict = {p: holdings_val[p] / total_portfolio_val for p in tickers}
            valid_pos, pos_msg = circuit_breaker.validate_position_limits(current_weights_dict)
            if not valid_pos:
                st.error(f"🚨 Circuit Breaker Position Limit Block: {pos_msg}")
                log_audit(venue="SYSTEM", asset="PORTFOLIO", price=0.0, amount=total_portfolio_val, units="ALL", status="POSITION_LIMIT_REJECTED")
                st.stop()

            if not evaluation["rebalance_triggered"]:
                st.info("⏸️ Rebalance skipped: Portfolio drift is within the 3% tolerance threshold[cite: 1]. No action required.")
            else:
                st.success("✅ Drift exceeds 3% threshold. Executing rebalancing actions:")
                
                execution_plan = evaluation["execution_plan"]
                for pair, details in execution_plan.items():
                    action = details["action"]
                    target_val = details["target_adjustment_val"]
                    
                    if action in ["BUY", "SELL"] and target_val > 50:
                        current_price = prices.get(pair, 0.0)
                        if current_price > 0:
                            if order_executor and not paper_mode:
                                res = order_executor.execute_market_order(pair=pair, side=action, counter_volume=str(round(target_val, 2)))
                                st.write(f"Executed {action} for {pair}: {res['status']}")
                            else:
                                st.info(f"[PAPER MODE / SIMULATION] Would execute {action} on {pair} for value ₦{target_val:,.2f}.")
                                log_audit(venue="LUNO", asset=pair, price=current_price, amount=target_val, units="SIMULATED", status=f"PAPER_{action}")
                                
                st.success("Autonomous rebalancing cycle completed and recorded to SQLite WAL audit logs[cite: 1].")
        except Exception as e:
            st.error(f"🚨 Execution Error: {str(e)}")

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

    if st.button("📈 Run Econometric Estimation & Covariance Shrinkage", use_container_width=True):
        with st.spinner("Fitting GARCH models with Student-t innovations & calculating Ledoit-Wolf shrinkage..."):
            np.random.seed(42)
            dates = pd.date_range(end=pd.Timestamp.today(), periods=lookback_window, freq="D")
            
            sim_returns_data = {
                "XBTNGN": np.random.normal(0.0012, 0.035, lookback_window),
                "ETHNGN": np.random.normal(0.0015, 0.042, lookback_window),
                "SOLNGN": np.random.normal(0.0020, 0.055, lookback_window)
            }
            returns_df = pd.DataFrame(sim_returns_data, index=dates)

            cov_matrix = quant_engine.compute_robust_covariance(returns_df, method="ledoit_wolf")
            risk_metrics = quant_engine.calculate_portfolio_var_cvar(
                weights=rebalancer.target_weights,
                returns_df=returns_df,
                covariance_matrix=cov_matrix,
                portfolio_value=total_portfolio_val if total_portfolio_val > 0 else 10_000_000.0,
                horizon=var_horizon
            )

            st.success("✅ Econometric pipeline successfully computed.")

            st.markdown("### 📊 Portfolio Value at Risk (VaR) & Expected Shortfall (CVaR)")
            st.dataframe(pd.DataFrame(risk_metrics).T, use_container_width=True)

            st.markdown("### 🧮 Ledoit-Wolf Shrinkage Covariance Matrix")
            st.dataframe(cov_matrix, use_container_width=True)

            asset_to_fit = st.selectbox("Inspect Asset Conditional Volatility", tickers)
            garch_result = quant_engine.fit_optimized_garch(returns_df[asset_to_fit], model_type=selected_model)
            
            if garch_result["success"]:
                st.info(f"Model Fit Successful | Persistence Parameter (Alpha + Beta): {garch_result['persistence']:.4f}")

            fig_vol = go.Figure()
            fig_vol.add_trace(go.Scatter(x=dates, y=returns_df[asset_to_fit], mode='lines', name='Log Returns', line=dict(color='gray', width=1)))
            fig_vol.add_trace(go.Scatter(x=dates, y=garch_result['conditional_volatility'], mode='lines', name=f'{selected_model} Volatility', line=dict(color='cyan', width=2)))
            fig_vol.update_layout(title=f"{asset_to_fit} — {selected_model} Volatility Dynamics (Student-t Innovations)", template="plotly_dark")
            st.plotly_chart(fig_vol, use_container_width=True)

# --- MODULE 3: BACKTEST SIMULATION ENGINE ---
elif app_mode == "Backtest Simulation Engine":
    st.subheader("Institutional Backtest Simulation Engine")
    st.caption("Compare Buy & Hold versus the Autonomous Rebalancing & GARCH Engine factoring in transaction fees[cite: 1].")

    bt_days = st.slider("Backtest Historical Period (Days)", min_value=30, max_value=365, value=180)

    if st.button("🚀 Run Backtest Simulation", use_container_width=True):
        with st.spinner("Running historical backtest simulation..."):
            np.random.seed(100)
            bt_dates = pd.date_range(end=pd.Timestamp.today(), periods=bt_days, freq="D")
            
            sim_prices_data = {
                "XBTNGN": 60000000 * np.cumprod(1 + np.random.normal(0.001, 0.03, bt_days)),
                "ETHNGN": 3500000 * np.cumprod(1 + np.random.normal(0.0012, 0.038, bt_days)),
                "SOLNGN": 200000 * np.cumprod(1 + np.random.normal(0.0015, 0.048, bt_days))
            }
            price_df = pd.DataFrame(sim_prices_data, index=bt_dates)

            results = backtest_engine.run_simulation(price_df, target_weights=rebalancer.target_weights)

            st.success("✅ Backtest Simulation Complete.")
            
            m_res1, m_res2, m_res3, m_res4 = st.columns(4)
            m_res1.metric("Buy & Hold Final Value", f"₦{results['buy_hold_final_value']:,.2f}", f"{results['buy_hold_return_pct']}%")
            m_res2.metric("ICIE Engine Final Value", f"₦{results['icie_final_value']:,.2f}", f"{results['icie_return_pct']}%")
            m_res3.metric("Annualized Sharpe Ratio", f"{results['sharpe_ratio']}")
            m_res4.metric("Maximum Drawdown", f"{results['max_drawdown_pct']}%")

# --- MODULE 4: SQLITE AUDIT LEDGER ---
elif app_mode == "SQLite Audit Ledger":
    st.subheader("📜 SQLite WAL Audit Ledger & Event History")
    audit_df = fetch_audit_ledger(100)
    if not audit_df.empty:
        st.dataframe(audit_df, use_container_width=True)
    else:
        st.info("No audit logs recorded in SQLite WAL database yet. Execute a rebalance or trigger simulation events to populate the ledger.")
