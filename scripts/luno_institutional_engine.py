#!/usr/bin/env python3
"""
ICIE Autonomous Institutional Engine (Luno Live Production Node v2.2)
Author: Bamidele Adedeji, Principal Consultant (Bamidele & Associates)
Objective: Live Institutional Crypto Investment & Cashless Self-Rebalancing (50% BTC, 30% ETH, 20% SOL)
Architecture: Zero-Simulation Production Node with Phase B GARCH(1,1) & Fat-Tail VaR Gate,
Phase C High-Water Mark Drawdown Circuit Breaker, Sequential Settlement & Dual Audit Trails.
"""

import os
import sys
import json
import sqlite3
import pandas as pd
from datetime import datetime
from decimal import Decimal

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

TRADE_LOG_FILE = "trade_log.csv"
LOG_FILE = "institutional_audit.log"

if not os.path.exists(TRADE_LOG_FILE):
    df_init = pd.DataFrame(columns=["timestamp", "asset", "side", "volume", "execution_price", "total_value_ngn", "status"])
    df_init.to_csv(TRADE_LOG_FILE, index=False)

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"{datetime.utcnow().isoformat()} [INFO] ICIE Live Audit Log Initialized.\n")

from core.config import load_config
from core.logging import setup_logger
from database.repository import init_database, log_audit
from execution.luno_connector import LunoConnector
from execution.order_executor import InstitutionalOrderExecutor
from risk.circuit_breakers import CircuitBreakerManager
from risk.volatility_gate import VolatilityRegimeGate
from strategy.rebalancer import ICIERebalancer, TARGET_WEIGHTS

logger = setup_logger("ICIEEngine")


def evaluate_autonomous_hwm(current_portfolio_val: float, max_drawdown_pct: float = 0.10) -> dict:
    """Updates SQLite & JSON Peak High-Water Mark and enforces the 10% Phase C drawdown circuit breaker."""
    db_name = "luno_institutional_engine.db"
    state_file = "hwm_state.json"
    initial_invested = 95000.00
    seed_peak = 59848.50
    now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
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
            initial_invested = float(row)
            peak_value = float(row)

        if current_val > peak_value:
            peak_value = current_val
            cursor.execute("UPDATE high_water_mark SET peak_value = ?, updated_at = ? WHERE id = 1", (peak_value, now_str))
        conn.commit()

    try:
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(
                {"initial_invested": initial_invested, "peak_value": peak_value, "last_recorded_value": current_val, "updated_at": now_str},
                f,
                indent=2,
            )
    except Exception:
        pass

    active_dd = ((peak_value - current_val) / peak_value) if (peak_value > 0 and current_val > 0) else 0.0
    return {
        "peak_value": peak_value,
        "active_dd_pct": round(active_dd * 100.0, 2),
        "breached": active_dd >= max_drawdown_pct,
    }
def run_institutional_engine():
    """
    Executes the Live ICIE Closed-Loop Two-Phase Cashless Rebalancing Engine with
    Phase B GARCH(1,1) / Cornish-Fisher VaR telemetry and Phase C High-Water Mark protection.
    """
    logger.info("[START] Initializing ICIE Autonomous Institutional Engine v2.2 (Luno Live Production Node)...")

    init_database()
    config = load_config()

    luno_key = os.environ.get("LUNO_API_KEY_ID")
    luno_secret = os.environ.get("LUNO_API_KEY_SECRET")

    if not luno_key or not luno_secret:
        logger.error("CRITICAL [CIRCUIT BREAKER]: LUNO_API_KEY_ID or LUNO_API_KEY_SECRET missing from environment. Aborting live execution.")
        try:
            log_audit(venue="ICIE", asset="SYSTEM", price=0.0, amount=0.0, units="N/A", status="CIRCUIT_BREAKER_ABORT: Missing Luno API credentials")
        except Exception:
            pass
        sys.exit(1)

    try:
        connector = LunoConnector(luno_key, luno_secret)
        circuit_breaker = CircuitBreakerManager()
        order_executor = InstitutionalOrderExecutor(connector, circuit_breaker)
        vol_gate = VolatilityRegimeGate(api_key_id=luno_key, api_key_secret=luno_secret)
        target_weights_cfg = config.get("target_weights", TARGET_WEIGHTS)

        rebalancer = ICIERebalancer(
            connector=connector,
            order_executor=order_executor,
            target_weights=target_weights_cfg,
        )
    except Exception as init_err:
        logger.error(f"CRITICAL: Engine initialization failed: {init_err}")
        sys.exit(1)

    logger.info("[LUNO API] Querying live exchange portfolio valuation, wallet breakdown, and order-book spreads...")

    try:
        portfolio_state = connector.get_portfolio_valuation()
    except Exception as e:
        logger.error(f"CRITICAL [CIRCUIT BREAKER]: Failed to retrieve live portfolio state from Luno: {e}")
        try:
            log_audit(venue="ICIE", asset="SYSTEM", price=0.0, amount=0.0, units="N/A", status=f"CIRCUIT_BREAKER_ABORT: Valuation fetch failed - {str(e)}")
        except Exception:
            pass
        sys.exit(1)

    total_portfolio = portfolio_state.get("total_portfolio", Decimal("0"))
    ngn_reserve = portfolio_state.get("ngn_reserve", Decimal("0"))
    asset_vals = portfolio_state.get("asset_values", {})
    tickers = portfolio_state.get("market_tickers", {})

    if total_portfolio <= 0:
        logger.error("CRITICAL: Total live portfolio capital evaluated to zero or negative. Aborting execution.")
        sys.exit(1)

    hwm = evaluate_autonomous_hwm(float(total_portfolio))
    logger.info(
        f"[AUDIT] Live Portfolio Capital (TPV): ₦{total_portfolio:,.2f} | Free NGN: ₦{ngn_reserve:,.2f} | "
        f"Peak HWM: ₦{hwm['peak_value']:,.2f} | Drawdown: {hwm['active_dd_pct']:.2f}%"
    )
    logger.info(
        f"[AUDIT] Live Holdings -> BTC: ₦{asset_vals.get('XBT', 0):,.2f} | "
        f"ETH: ₦{asset_vals.get('ETH', 0):,.2f} | SOL: ₦{asset_vals.get('SOL', 0):,.2f}"
    )

    for coin in ["XBT", "ETH", "SOL"]:
        reg = vol_gate.evaluate_asset_regime(coin)
        logger.info(
            f"[PHASE B GARCH] {reg.get('pair', coin)}: Regime={reg.get('regime')} | "
            f"AnnVol={float(reg.get('ann_vol', 0))*100:.1f}% | CF_VaR24h={float(reg.get('cf_var_24h', 0))*100:.2f}% | "
            f"Scalar={reg.get('vol_scalar', 1.0):.2f}x | SafeToBuy={reg.get('safe_to_buy')}"
        )

    for coin, t_info in tickers.items():
        logger.info(
            f"[ORDER BOOK] {t_info['pair']}: Bid ₦{t_info['bid']:,.2f} | "
            f"Ask ₦{t_info['ask']:,.2f} | Spread {t_info['spread_pct']*100:.2f}%"
        )

    if hwm["breached"]:
        logger.warning(f"🚨 [PHASE C CIRCUIT BREAKER] Active drawdown ({hwm['active_dd_pct']:.2f}%) >= 10.0% ceiling. Halting rebalance deployment.")
        try:
            log_audit(venue="ICIE", asset="PORTFOLIO", price=0.0, amount=float(total_portfolio), units="N/A", status=f"PHASE_C_HWM_HALT: {hwm['active_dd_pct']:.2f}% DD")
        except Exception:
            pass
        return

    drift_detected, drift_analysis = rebalancer.evaluate_drift(portfolio_state)
    max_drift = max([abs(d.get("drift", 0)) for d in drift_analysis.values()]) if drift_analysis else 0.0

    for asset, stats in drift_analysis.items():
        logger.info(
            f"[DRIFT AUDIT] {asset}: Actual {stats['actual_weight']*100:.2f}% vs "
            f"Target {stats['target_weight']*100:.2f}% | Drift: {stats['drift']*100:+.2f}% | "
            f"Threshold: {stats['dynamic_threshold']*100:.2f}% | Breached: {stats['breached']}"
        )

    if not drift_detected:
        logger.info(f"[REBALANCE] Portfolio drift ({max_drift*100:.2f}%) is within Constitution & exchange-floor tolerance. Holding position.")
        logger.info("[COMPLETE] Live institutional audit cycle finished successfully.")
        try:
            log_audit(venue="ICIE", asset="PORTFOLIO", price=0.0, amount=float(total_portfolio), units="N/A", status="NO_DRIFT_HOLD")
        except Exception:
            pass
        return

    logger.info(f"✅ Drift threshold breached (Max Drift: {max_drift*100:.2f}%). Initiating Live Cashless Self-Rebalance Sequence.")

    try:
        rebalance_result = rebalancer.execute_rebalance(initial_state=portfolio_state)
        logger.info(f"[EXECUTION COMPLETE] ICIE Live Rebalance Result: {rebalance_result}")
        try:
            log_audit(
                venue="ICIE",
                asset="PORTFOLIO",
                price=0.0,
                amount=float(total_portfolio),
                units="N/A",
                status=f"REBALANCE_CYCLE_RESULT: {rebalance_result.get('status', 'COMPLETED')}",
            )
        except Exception:
            pass
    except Exception as exec_err:
        logger.error(f"CRITICAL: Live rebalance execution sequence failed: {exec_err}")
        try:
            log_audit(venue="ICIE", asset="PORTFOLIO", price=0.0, amount=float(total_portfolio), units="N/A", status=f"REBALANCE_FAILED: {str(exec_err)}")
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    run_institutional_engine()
