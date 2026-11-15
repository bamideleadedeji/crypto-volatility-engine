
### File 4 of 4: `scripts/luno_institutional_engine.py`

```python
#!/usr/bin/env python3
"""
ICIE Autonomous Institutional Engine (Luno Live Production Node)
Author: Bamidele Adedeji, Principal Consultant (Bamidele & Associates)
Objective: Live Institutional Crypto Investment & Cashless Self-Rebalancing (50% BTC, 30% ETH, 20% SOL)
Architecture: Zero-Simulation Production Node with Sequential Settlement Verification & Dual Audit Trails
"""

import os
import sys
import pandas as pd
from datetime import datetime
from decimal import Decimal

# ==========================================
# 0. ABSOLUTE PATH RESOLUTION & LOG INITIALIZATION
# ==========================================
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
from strategy.rebalancer import ICIERebalancer, TARGET_WEIGHTS

# ==========================================
# 1. INITIALIZATION & STRICT LIVE CREDENTIAL GATE
# ==========================================
logger = setup_logger("ICIEEngine")


def run_institutional_engine():
    """
    Executes the Live ICIE Closed-Loop Two-Phase Cashless Rebalancing Engine.
    Strictly enforces live Luno API connectivity—aborts via Circuit Breaker if credentials
    are missing or exchange telemetry fails. Zero simulation code.
    """
    logger.info("[START] Initializing ICIE Autonomous Institutional Engine (Luno Live Production Node)...")

    init_database()
    config = load_config()

    luno_key = os.environ.get("LUNO_API_KEY_ID")
    luno_secret = os.environ.get("LUNO_API_KEY_SECRET")

    if not luno_key or not luno_secret:
        logger.error("CRITICAL [CIRCUIT BREAKER]: LUNO_API_KEY_ID or LUNO_API_KEY_SECRET missing from environment. Aborting live execution.")
        try:
            log_audit(
                venue="ICIE",
                asset="SYSTEM",
                price=0.0,
                amount=0.0,
                units="N/A",
                status="CIRCUIT_BREAKER_ABORT: Missing Luno API credentials"
            )
        except Exception:
            pass
        sys.exit(1)

    try:
        connector = LunoConnector(luno_key, luno_secret)
        circuit_breaker = CircuitBreakerManager()
        order_executor = InstitutionalOrderExecutor(connector, circuit_breaker)
        target_weights_cfg = config.get("target_weights", TARGET_WEIGHTS)

        rebalancer = ICIERebalancer(
            connector=connector,
            order_executor=order_executor,
            target_weights=target_weights_cfg
        )
    except Exception as init_err:
        logger.error(f"CRITICAL: Engine initialization failed: {init_err}")
        sys.exit(1)

    # ==========================================
    # 2. FETCH REAL LIVE VALUATION & ORDER BOOK TELEMETRY
    # ==========================================
    logger.info("[LUNO API] Querying live exchange portfolio valuation, wallet breakdown, and order-book spreads...")

    try:
        portfolio_state = connector.get_portfolio_valuation()
    except Exception as e:
        logger.error(f"CRITICAL [CIRCUIT BREAKER]: Failed to retrieve live portfolio state from Luno: {e}")
        try:
            log_audit(
                venue="ICIE",
                asset="SYSTEM",
                price=0.0,
                amount=0.0,
                units="N/A",
                status=f"CIRCUIT_BREAKER_ABORT: Valuation fetch failed - {str(e)}"
            )
        except Exception:
            pass
        sys.exit(1)

    total_portfolio = portfolio_state.get('total_portfolio', Decimal('0'))
    ngn_reserve = portfolio_state.get('ngn_reserve', Decimal('0'))
    asset_vals = portfolio_state.get('asset_values', {})
    tickers = portfolio_state.get('market_tickers', {})

    if total_portfolio <= 0:
        logger.error("CRITICAL: Total live portfolio capital evaluated to zero or negative. Aborting execution.")
        sys.exit(1)

    logger.info(f"[AUDIT] Live Portfolio Capital (TPV): ₦{total_portfolio:,.2f} | Free Cleared NGN: ₦{ngn_reserve:,.2f}")
    logger.info(
        f"[AUDIT] Live Holdings -> BTC: ₦{asset_vals.get('XBT', 0):,.2f} | "
        f"ETH: ₦{asset_vals.get('ETH', 0):,.2f} | SOL: ₦{asset_vals.get('SOL', 0):,.2f}"
    )

    for coin, t_info in tickers.items():
        logger.info(
            f"[ORDER BOOK] {t_info['pair']}: Bid ₦{t_info['bid']:,.2f} | "
            f"Ask ₦{t_info['ask']:,.2f} | Spread {t_info['spread_pct']*100:.2f}%"
        )

    # ==========================================
    # 3. EVALUATE REBALANCE DRIFT
    # ==========================================
    drift_detected, drift_analysis = rebalancer.evaluate_drift(portfolio_state)
    max_drift = max([abs(d.get('drift', 0)) for d in drift_analysis.values()]) if drift_analysis else 0.0

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
            log_audit(
                venue="ICIE",
                asset="PORTFOLIO",
                price=0.0,
                amount=float(total_portfolio),
                units="N/A",
                status="NO_DRIFT_HOLD"
            )
        except Exception:
            pass
        return

    logger.info(f"✅ Drift threshold breached (Max Drift: {max_drift*100:.2f}%). Initiating Live Cashless Self-Rebalance Sequence.")

    # ==========================================
    # 4. EXECUTE LIVE TWO-PHASE CASHLESS REBALANCE
    # ==========================================
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
                status=f"REBALANCE_CYCLE_RESULT: {rebalance_result.get('status', 'COMPLETED')}"
            )
        except Exception:
            pass
    except Exception as exec_err:
        logger.error(f"CRITICAL: Live rebalance execution sequence failed: {exec_err}")
        try:
            log_audit(
                venue="ICIE",
                asset="PORTFOLIO",
                price=0.0,
                amount=float(total_portfolio),
                units="N/A",
                status=f"REBALANCE_FAILED: {str(exec_err)}"
            )
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    run_institutional_engine()
