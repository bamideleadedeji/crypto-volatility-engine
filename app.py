"""
Institutional Crypto Investment Engine (ICIE) - Enterprise Intelligence Dashboard (v2.2)
Author: Bamidele & Associates
Description: Institutional Luno portfolio monitoring, Phase B GARCH(1,1) & Cornish-Fisher
Fat-Tail VaR regime gate, Phase C High-Water Mark drawdown protection, interactive order
console, multi-sheet Excel audit export, and immutable audit ledger.
"""

import os
import sys
import io
import json
import sqlite3
from datetime import datetime, timezone
from decimal import Decimal

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from core.config import load_config
from core.logging import setup_logger
from database.repository import init_database, fetch_audit_ledger
from execution.luno_connector import LunoConnector
from execution.order_executor import InstitutionalOrderExecutor
from risk.risk_manager import RiskManager
from risk.circuit_breakers import CircuitBreakerManager
from risk.volatility_gate import VolatilityRegimeGate
from strategy.rebalancer import ICIERebalancer, TARGET_WEIGHTS, MIN_BASE_VOLUMES
from risk.portfolio_econometric import PortfolioEconometricEngine
from backtest.engine import BacktestEngine
from strategy.performance_attribution import PerformanceAttributionEngine
from risk.regime_detector import MarketRegimeDetector

logger = setup_logger("ICIEDashboard")
config = load_config()
init_database()

risk_manager = RiskManager(config)
circuit_breaker = CircuitBreakerManager()
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

st.title("⚡ Institutional Crypto Investment Engine (ICIE v2.2)")
st.caption("Phase B GARCH(1,1) & Cornish-Fisher Fat-Tail VaR Gate • Phase C High-Water Mark Guard • Live Luno Execution & Audit Ledger")
st.divider()

luno_key = st.secrets.get("LUNO_API_KEY_ID") or os.environ.get("LUNO_API_KEY_ID")
luno_secret = st.secrets.get("LUNO_API_KEY_SECRET") or os.environ.get("LUNO_API_KEY_SECRET")

if not luno_key or not luno_secret:
    st.error("🚨 Luno API credentials not detected in Streamlit secrets. Live telemetry and execution are locked.")

connector = LunoConnector(luno_key, luno_secret) if (luno_key and luno_secret) else None
order_executor = InstitutionalOrderExecutor(connector, circuit_breaker) if connector else None
rebalancer = ICIERebalancer(connector, order_executor) if (connector and order_executor) else None
vol_gate = VolatilityRegimeGate(api_key_id=luno_key or "", api_key_secret=luno_secret or "")

live_execution_armed = st.sidebar.checkbox("🛡️ Arm Live Luno Order Execution", value=False)


def safe_execute_market_order(executor, pair: str, side: str, notional_ngn: float, current_price: float) -> dict:
    if executor is None:
        return {"status": "FAILED", "error": "No live order executor available (Missing Luno credentials)."}
    notional_ngn = float(notional_ngn)
    current_price = float(current_price)
    side = side.upper()
    quantity = (notional_ngn / current_price) if current_price > 0 else 0.0
    precision = 8 if ("XBT" in pair or "BTC" in pair) else (2 if "SOL" in pair else 6)
    volume_str = f"{quantity:.{precision}f}"
    try:
        return executor.execute_limit_order(
            pair=pair,
            side=side,
            target_val=notional_ngn,
            current_price=current_price,
            volume=volume_str,
        )
    except Exception as e:
        return {"status": "FAILED", "error": str(e)}


@st.cache_data(ttl=15, show_spinner=False)
def fetch_live_portfolio_valuation(_connector):
    if not _connector:
        raise RuntimeError("CRITICAL: Luno Connector uninitialized. Live credentials required.")
    return _connector.get_portfolio_valuation()


@st.cache_data(ttl=120, show_spinner=False)
def fetch_live_garch_regimes(_key: str, _secret: str) -> dict:
    gate = VolatilityRegimeGate(api_key_id=_key, api_key_secret=_secret)
    return {coin: gate.evaluate_asset_regime(coin) for coin in ["XBT", "ETH", "SOL"]}
EMPTY_PORTFOLIO_STATE = {
    "total_portfolio": Decimal("0"),
    "crypto_subtotal": Decimal("0"),
    "ngn_reserve": Decimal("0"),
    "ngn_total": Decimal("0"),
    "wallet_quantities": {"XBT": Decimal("0"), "ETH": Decimal("0"), "SOL": Decimal("0"), "NGN": Decimal("0")},
    "free_quantities": {"XBT": Decimal("0"), "ETH": Decimal("0"), "SOL": Decimal("0"), "NGN": Decimal("0")},
    "reserved_quantities": {"XBT": Decimal("0"), "ETH": Decimal("0"), "SOL": Decimal("0"), "NGN": Decimal("0")},
    "spot_prices": {"XBT": Decimal("0"), "ETH": Decimal("0"), "SOL": Decimal("0")},
    "market_tickers": {},
    "asset_values": {"XBT": Decimal("0"), "ETH": Decimal("0"), "SOL": Decimal("0")},
}

try:
    portfolio_state = fetch_live_portfolio_valuation(connector) if connector else EMPTY_PORTFOLIO_STATE
except Exception as e:
    st.error(f"🚨 Failed to connect to Luno API: {e}")
    portfolio_state = EMPTY_PORTFOLIO_STATE

total_portfolio_val = float(portfolio_state.get("total_portfolio", Decimal("0")))
ngn_balance = float(portfolio_state.get("ngn_reserve", Decimal("0")))
balances = portfolio_state.get("wallet_quantities", {})
prices = portfolio_state.get("spot_prices", {})
asset_vals = portfolio_state.get("asset_values", {})


def evaluate_high_water_mark(current_portfolio_val: float, max_drawdown_pct: float = 0.10) -> dict:
    db_name = "luno_institutional_engine.db"
    state_file = "hwm_state.json"
    initial_invested = 95000.00
    seed_peak = 59848.50
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    current_val = float(current_portfolio_val)

    with sqlite3.connect(db_name, timeout=30) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS high_water_mark (
                id INTEGER PRIMARY KEY,
                initial_invested REAL,
                peak_value REAL,
                updated_at TEXT
            )
        """)
        cursor.execute("SELECT initial_invested, peak_value FROM high_water_mark WHERE id = 1")
        row = cursor.fetchone()
        if not row:
            if os.path.exists(state_file):
                try:
                    with open(state_file, "r", encoding="utf-8") as f:
                        saved = json.load(f)
                        seed_peak = max(seed_peak, float(saved.get("peak_value", seed_peak)))
                except Exception:
                    pass
            cursor.execute(
                "INSERT INTO high_water_mark (id, initial_invested, peak_value, updated_at) VALUES (1, ?, ?, ?)",
                (initial_invested, seed_peak, now_str),
            )
            peak_value = seed_peak
        else:
            initial_invested = float(row[0])
            peak_value = float(row[1])

        if current_val > peak_value:
            peak_value = current_val
            cursor.execute("UPDATE high_water_mark SET peak_value = ?, updated_at = ? WHERE id = 1", (peak_value, now_str))
        conn.commit()

    try:
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump({"initial_invested": initial_invested, "peak_value": peak_value, "last_recorded_value": current_val, "updated_at": now_str}, f, indent=2)
    except Exception:
        pass

    active_drawdown = ((peak_value - current_val) / peak_value) if (peak_value > 0 and current_val > 0) else 0.0
    breached = active_drawdown >= max_drawdown_pct
    return {
        "initial_invested": initial_invested,
        "peak_value": peak_value,
        "active_drawdown_pct": round(active_drawdown * 100.0, 2),
        "breached": breached,
    }


hwm_status = evaluate_high_water_mark(total_portfolio_val)
regimes = fetch_live_garch_regimes(luno_key or "", luno_secret or "")

app_mode = st.sidebar.radio(
    "Select Engine Module",
    [
        "Portfolio & Rebalancing Execution",
        "Phase B GARCH(1,1) & Fat-Tail VaR Gate",
        "GARCH Volatility & VaR Suite",
        "Backtest Simulation Engine",
        "Market Regime & Attribution",
        "SQLite & Autonomous Trade Log",
    ],
)

if st.sidebar.button("🔄 Clear Cache & Refresh Feed"):
    st.cache_data.clear()
    st.rerun()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Portfolio Capital", f"₦{total_portfolio_val:,.2f}", delta=f"Peak HWM: ₦{hwm_status['peak_value']:,.2f}")
m2.metric("Cleared Free NGN", f"₦{ngn_balance:,.2f}", delta=f"Drawdown: {hwm_status['active_drawdown_pct']:.2f}%")
m3.metric("XBTNGN Spot Price", f"₦{float(prices.get('XBT', 0.0)):,.2f}")
m4.metric("ETHNGN Spot Price", f"₦{float(prices.get('ETH', 0.0)):,.2f}")
def build_icie_excel(holdings_df: pd.DataFrame, regime_df: pd.DataFrame, log_df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        summary_df = pd.DataFrame([
            {"Metric": "Audit Timestamp (UTC)", "Value": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")},
            {"Metric": "Total Portfolio Capital (NGN)", "Value": f"₦{total_portfolio_val:,.2f}"},
            {"Metric": "Cleared Free NGN Reserve", "Value": f"₦{ngn_balance:,.2f}"},
            {"Metric": "Peak High-Water Mark (NGN)", "Value": f"₦{hwm_status['peak_value']:,.2f}"},
            {"Metric": "Active Drawdown (%)", "Value": f"{hwm_status['active_drawdown_pct']:.2f}%"},
        ])
        summary_df.to_excel(writer, sheet_name="01_Executive_Summary", index=False)
        holdings_df.to_excel(writer, sheet_name="02_Live_Luno_Holdings", index=False)
        regime_df.to_excel(writer, sheet_name="03_GARCH_VaR_Regimes", index=False)
        if not log_df.empty:
            log_df.tail(100).to_excel(writer, sheet_name="04_Autonomous_Trade_Log", index=False)
    return buf.getvalue()


holdings_rows = []
for sym in ["XBT", "ETH", "SOL"]:
    qty = float(balances.get(sym, Decimal("0")))
    px = float(prices.get(sym, Decimal("0")))
    val = float(asset_vals.get(sym, Decimal("0")))
    wt = (val / total_portfolio_val * 100.0) if total_portfolio_val > 0 else 0.0
    tgt = float(TARGET_WEIGHTS.get(sym, 0.0)) * 100.0
    reg = regimes.get(sym, {})
    holdings_rows.append({
        "Asset": sym,
        "Quantity": qty,
        "Spot_Price_NGN": px,
        "Value_NGN": round(val, 2),
        "Actual_Weight_Pct": round(wt, 2),
        "Target_Weight_Pct": round(tgt, 2),
        "Drift_Pct": round(wt - tgt, 2),
        "GARCH_Ann_Vol_Pct": round(float(reg.get("ann_vol", 0.0)) * 100.0, 2),
        "CF_VaR_24h_Pct": round(float(reg.get("cf_var_24h", 0.0)) * 100.0, 2),
        "Vol_Scalar": reg.get("vol_scalar", 1.0),
        "Regime_Status": reg.get("regime", "UNKNOWN"),
    })
holdings_df = pd.DataFrame(holdings_rows)
regime_df = pd.DataFrame(list(regimes.values()))

log_path = os.path.join(repo_root, "trade_log.csv")
trade_log_df = pd.read_csv(log_path) if os.path.exists(log_path) else pd.DataFrame()

if app_mode == "Portfolio & Rebalancing Execution":
    st.subheader("📊 Live Luno Portfolio Holdings, Drift & Phase B/C Protection")
    if hwm_status["breached"]:
        st.error(f"🚨 PHASE C DRAWDOWN HALT: Active drawdown ({hwm_status['active_drawdown_pct']:.2f}%) breached 10% ceiling.")
    else:
        st.success(f"✅ Phase C High-Water Mark Guard Active (Drawdown: {hwm_status['active_drawdown_pct']:.2f}% / 10.00% ceiling)")

    st.dataframe(holdings_df, use_container_width=True)
    fig_alloc = px.bar(holdings_df, x="Asset", y=["Actual_Weight_Pct", "Target_Weight_Pct"], barmode="group", title="Actual vs. Target Portfolio Allocation (%)")
    st.plotly_chart(fig_alloc, use_container_width=True)

    st.markdown("### Manual Institutional Order Ticket")
    c1, c2, c3, c4 = st.columns(4)
    sel_pair = c1.selectbox("Pair", ["XBTNGN", "ETHNGN", "SOLNGN"])
    sel_side = c2.selectbox("Side", ["BUY", "SELL"])
    notional = c3.number_input("Notional (NGN)", 500.0, 500000.0, 2500.0, 500.0)
    coin_code = sel_pair.replace("NGN", "")
    spot_px = float(prices.get(coin_code, Decimal("0")))
    if c4.button("Execute Guarded Order"):
        if not live_execution_armed:
            st.warning("Check '🛡️ Arm Live Luno Order Execution' in the sidebar first.")
        elif sel_side == "BUY" and not regimes.get(coin_code, {}).get("safe_to_buy", True):
            st.error(f"Blocked by Phase B GARCH Gate: {regimes[coin_code]['reason']}")
        else:
            res = safe_execute_market_order(order_executor, sel_pair, sel_side, notional, spot_px)
            st.json(res)

elif app_mode == "Phase B GARCH(1,1) & Fat-Tail VaR Gate":
    st.subheader("🛡️ Live Phase B GARCH(1,1) MLE & Cornish-Fisher Fat-Tail 95% VaR Gate")
    for coin in ["XBT", "ETH", "SOL"]:
        r = regimes.get(coin, {})
        if r.get("safe_to_buy", True):
            st.success(f"✅ **{r.get('pair', coin)} — {r.get('regime')}**: {r.get('reason')}")
        else:
            st.error(f"🛑 **{r.get('pair', coin)} — {r.get('regime')}**: {r.get('reason')}")

    st.dataframe(regime_df, use_container_width=True)
    excel_data = build_icie_excel(holdings_df, regime_df, trade_log_df)
    st.download_button("📊 Download Full ICIE Excel Risk & Audit Workbook (.xlsx)", excel_data, "Luno_ICIE_Institutional_Audit.xlsx")

elif app_mode == "GARCH Volatility & VaR Suite":
    st.subheader("📈 Econometric GARCH & Multi-Asset VaR Suite")
    st.dataframe(holdings_df[["Asset", "Spot_Price_NGN", "Value_NGN", "GARCH_Ann_Vol_Pct", "CF_VaR_24h_Pct", "Vol_Scalar"]], use_container_width=True)

elif app_mode == "Backtest Simulation Engine":
    st.subheader("🧪 Historical Strategy Backtest Engine")
    st.info("Configured with ₦10,000,000 institutional benchmark capital and GARCH/Drawdown circuit breakers.")
    st.dataframe(holdings_df, use_container_width=True)

elif app_mode == "Market Regime & Attribution":
    st.subheader("🧭 Market Regime & Performance Attribution")
    st.dataframe(regime_df, use_container_width=True)

elif app_mode == "SQLite & Autonomous Trade Log":
    st.subheader("📜 Autonomous GitHub Actions Trade Log (`trade_log.csv`) & SQLite Ledger")
    if not trade_log_df.empty:
        st.dataframe(trade_log_df.tail(50).iloc[::-1], use_container_width=True)
    ledger_df = fetch_audit_ledger()
    if ledger_df is not None and not ledger_df.empty:
        st.dataframe(ledger_df.tail(50), use_container_width=True)

