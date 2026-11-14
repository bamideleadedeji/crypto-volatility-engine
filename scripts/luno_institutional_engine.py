#!/usr/bin/env python3
"""
ICIE Autonomous Institutional Engine (Luno Live Production Node)
Author: Bamidele Adedeji, Principal Consultant (Bamidele & Associates)
Objective: Institutional Crypto Investment & Cashless Self-Rebalancing (50% BTC, 30% ETH, 20% SOL)
Architecture: Constitution-Aligned Sequential Settlement Verification, Treasury Management, & Immutable Audit Trails
"""

import os
import sys
import time
import logging
import pandas as pd
from datetime import datetime
from decimal import Decimal, ROUND_DOWN

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
        f.write(f"{datetime.utcnow().isoformat()} [INFO] ICIE Audit Log Initialized.\n")

from core.config import load_config
from core.logging import setup_logger
from database.repository import init_database, log_audit, get_paper_portfolio, update_paper_portfolio, record_order
from execution.luno_connector import LunoConnector
from execution.order_executor import InstitutionalOrderExecutor
from risk.circuit_breakers import CircuitBreakerManager
from strategy.rebalancer import ICIERebalancer, TARGET_WEIGHTS, AutonomousRebalancer

# ==========================================
# 1. INITIALIZATION & CONFIGURATION
# ==========================================
logger = setup_logger("ICIEEngine")

# Luno Exchange Minimum Base Volume Limits (Strict Exchange Compliance)
MIN_BASE_VOLUMES = {
    "XBTNGN": Decimal('0.0005'),
    "ETHNGN": Decimal('0.005'),
    "SOLNGN": Decimal('0.05')
}

MIN_ORDER_NGN = Decimal('50.00')
LUNO_TAKER_FEE = Decimal('0.0075')  # 0.75% taker fee
AGGRESSIVE_LIMIT_BUFFER = Decimal('0.0015')  # 0.15% aggressive price cross

def run_institutional_engine():
    """
    Executes the ICIE Closed-Loop Two-Phase Autonomous Rebalancing Engine.
    Enforces strict live fetching, balance + reserved wallet aggregation via LunoConnector,
    Decimal precision arithmetic, and SQLite WAL immutable audit trails.
    """
    logger.info("[START] Initializing ICIE Autonomous Institutional Engine (Luno Live Production Node)...")
    
    init_database()
    config = load_config()
    
    luno_key = os.environ.get("LUNO_API_KEY_ID")
    luno_secret = os.environ.get("LUNO_API_KEY_SECRET")
    
    paper_mode = config.get("paper_mode", False)
    if not luno_key or not luno_secret:
        logger.warning("⚠️ Luno API credentials not found in environment. Forcing Paper Trading / Simulation Mode.")
        paper_mode = True

    connector = LunoConnector(luno_key, luno_secret) if (luno_key and luno_secret) else None
    circuit_breaker = CircuitBreakerManager()
    order_executor = InstitutionalOrderExecutor(connector, circuit_breaker) if connector else None
    
    # Target weights matching the ICIE Constitution (BTC 50%, ETH 30%, SOL 20%)
    target_weights_dict = config.get("target_weights", {"XBTNGN": 0.50, "ETHNGN": 0.30, "SOLNGN": 0.20})
    
    rebalancer = ICIERebalancer(
        connector=connector,
        order_executor=order_executor,
        dry_run=paper_mode,
        target_weights=TARGET_WEIGHTS
    )

    tickers = ["XBTNGN", "ETHNGN", "SOLNGN"]
    
    # ==========================================
    # 2. FETCH REAL LIVE VALUATION & WALLET BALANCES (ATOMIC)
    # ==========================================
    logger.info("[LUNO API] Querying live exchange portfolio valuation and wallet balances...")
    
    try:
        if connector and not paper_mode:
            # Atomic portfolio valuation utilizing verified LunoConnector methods
            portfolio_state = connector.get_portfolio_valuation()
        else:
            if not paper_mode:
                raise RuntimeError("CRITICAL: Live mode requested, but Luno connector is uninitialized.")
            logger.info("[PAPER MODE] Initializing simulated portfolio state...")
            initial_cash = Decimal('1500.00')
            initial_units = {"XBT": Decimal('0.00025'), "ETH": Decimal('0.0065'), "SOL": Decimal('0.05')}
            ngn_res, units_dict = get_paper_portfolio(float(initial_cash), {k+"NGN": float(v) for k, v in initial_units.items()})
            
            spot_dict = {'XBT': Decimal('101425618.00'), 'ETH': Decimal('2780346.00'), 'SOL': Decimal('220000.00')}
            crypto_sub = sum(Decimal(str(units_dict.get(k+"NGN", 0))) * spot_dict[k] for k in ['XBT', 'ETH', 'SOL'])
            
            portfolio_state = {
                'total_portfolio': Decimal(str(ngn_res)) + crypto_sub,
                'crypto_subtotal': crypto_sub,
                'ngn_reserve': Decimal(str(ngn_res)),
                'wallet_quantities': {'XBT': Decimal(str(units_dict.get("XBTNGN", 0))), 'ETH': Decimal(str(units_dict.get("ETHNGN", 0))), 'SOL': Decimal(str(units_dict.get("SOLNGN", 0))), 'NGN': Decimal(str(ngn_res))},
                'spot_prices': spot_dict,
                'asset_values': {k: Decimal(str(units_dict.get(k+"NGN", 0))) * spot_dict[k] for k in ['XBT', 'ETH', 'SOL']}
            }
    except Exception as e:
        logger.error(f"CRITICAL: Failed to retrieve portfolio state from exchange: {e}")
        try:
            log_audit(venue="ICIE", asset="SYSTEM", price=0.0, amount=0.0, units="N/A", status=f"Valuation fetch failed: {str(e)}")
        except Exception:
            pass
        sys.exit(1)

    total_portfolio = portfolio_state.get('total_portfolio', Decimal('0'))
    ngn_reserve = portfolio_state.get('ngn_reserve', Decimal('0'))
    asset_vals = portfolio_state.get('asset_values', {})
    spot_prices = portfolio_state.get('spot_prices', {})
    balances = portfolio_state.get('wallet_quantities', {})

    if total_portfolio <= 0:
        logger.error("CRITICAL: Total portfolio capital evaluated to zero or negative. Aborting execution.")
        sys.exit(1)

    logger.info(f"[AUDIT] Live Portfolio Capital (TPV): ₦{total_portfolio:,.2f} | Liquid NGN Reserve: ₦{ngn_reserve:,.2f}")
    logger.info(f"[AUDIT] Asset Market Values -> BTC: ₦{asset_vals.get('XBT', 0):,.2f} | ETH: ₦{asset_vals.get('ETH', 0):,.2f} | SOL: ₦{asset_vals.get('SOL', 0):,.2f}")

    # Convert asset valuation into pair-based pricing for legacy audit compatibility
    prices = {
        "XBTNGN": float(spot_prices.get('XBT', 0)),
        "ETHNGN": float(spot_prices.get('ETH', 0)),
        "SOLNGN": float(spot_prices.get('SOL', 0))
    }
    
    holdings_units = {
        "XBTNGN": float(balances.get('XBT', 0)),
        "ETHNGN": float(balances.get('ETH', 0)),
        "SOLNGN": float(balances.get('SOL', 0))
    }
    
    curr_weights_pct = {
        pair: (float(asset_vals.get(pair.replace("NGN","").replace("XBT","XBT"), Decimal('0'))) / float(total_portfolio) * 100) if total_portfolio > 0 else 0.0 
        for pair in tickers
    }

    # ==========================================
    # 3. EVALUATE REBALANCE DRIFT
    # ==========================================
    drift_detected, drift_analysis = rebalancer.evaluate_drift(portfolio_state)
    max_drift = max([abs(d.get('drift', 0)) for d in drift_analysis.values()]) if drift_analysis else 0.0

    if not drift_detected:
        logger.info(f"[REBALANCE] Portfolio drift ({max_drift*100:.2f}%) is within exchange-floor tolerance. Holding position.")
        logger.info("[COMPLETE] Institutional rebalancing audit cycle finished successfully.")
        try:
            log_audit(venue="ICIE", asset="PORTFOLIO", price=0.0, amount=float(total_portfolio), units="N/A", status="NO_DRIFT_HOLD")
        except Exception:
            pass
        return

    logger.info(f"✅ Drift exceeds threshold (Max Drift: {max_drift*100:.2f}%). Initiating Two-Phase Cashless Rebalance Sequence.")

    # ==========================================
    # 4. EXECUTE TWO-PHASE CASHLESS REBALANCE VIA REBALANCER MODULE
    # ==========================================
    try:
        rebalance_result = rebalancer.execute_rebalance()
        logger.info(f"[EXECUTION COMPLETE] ICIE Rebalance Result: {rebalance_result}")
        try:
            log_audit(
                venue="ICIE",
                asset="PORTFOLIO",
                price=0.0,
                amount=float(total_portfolio),
                units="N/A",
                status=f"REBALANCE_SUCCESS: {rebalance_result.get('status', 'COMPLETED')}"
            )
        except Exception:
            pass
    except Exception as exec_err:
        logger.error(f"CRITICAL: Rebalance execution sequence failed: {exec_err}")
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
