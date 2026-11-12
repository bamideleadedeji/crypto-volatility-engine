#!/usr/bin/env python3
"""
ICIE Autonomous Institutional Engine (Luno Live Production Node)
Author: Bamidele Adedeji, Principal Consultant (Bamidele & Associates)
Objective: Institutional Crypto Investment & Cashless Self-Rebalancing (50% BTC, 30% ETH, 20% SOL)
"""

import os
import sys
import time
import logging
import pandas as pd
from datetime import datetime

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
from strategy.rebalancer import AutonomousRebalancer

# ==========================================
# 1. INITIALIZATION & CONFIGURATION
# ==========================================
logger = setup_logger("ICIEEngine")

# Luno Exchange Minimum Base Volume Limits (Strict Exchange Compliance)
MIN_BASE_VOLUMES = {
    "XBTNGN": 0.0005,
    "ETHNGN": 0.005,
    "SOLNGN": 0.05
}

MIN_ORDER_NGN = 50.0
FEE_BUFFER_PCT = 0.002
AGGRESSIVE_LIMIT_BUFFER = 0.0015  # 0.15% aggressive price cross

def run_institutional_engine():
    """
    Executes the ICIE Closed-Loop Two-Phase Autonomous Rebalancing Engine.
    Enforces strict live fetching without silent simulation fallbacks when credentials exist.
    """
    logger.info("[START] Initializing ICIE Autonomous Institutional Engine (Luno Live Production Node)...")
    
    init_database()
    config = load_config()
    
    luno_key = os.environ.get("LUNO_API_KEY_ID")
    luno_secret = os.environ.get("LUNO_API_KEY_SECRET")
    
    paper_mode = config.get("paper_mode", True)
    if not luno_key or not luno_secret:
        logger.warning("⚠️ Luno API credentials not found in environment. Forcing Paper Trading / Simulation Mode.")
        paper_mode = True
    else:
        # If credentials exist, disable paper mode unless explicitly configured otherwise
        paper_mode = config.get("paper_mode", False)

    connector = LunoConnector(luno_key, luno_secret) if (luno_key and luno_secret) else None
    circuit_breaker = CircuitBreakerManager()
    order_executor = InstitutionalOrderExecutor(connector, circuit_breaker)
    
    target_weights = config.get("target_weights", {"XBTNGN": 0.50, "ETHNGN": 0.30, "SOLNGN": 0.20})
    rebalancer = AutonomousRebalancer(target_weights=target_weights)

    tickers = ["XBTNGN", "ETHNGN", "SOLNGN"]
    
    # ==========================================
    # 2. FETCH REAL LIVE PRICES & PORTFOLIO BALANCES
    # ==========================================
    prices = {}
    for pair in tickers:
        try:
            if connector and not paper_mode:
                prices[pair] = connector.fetch_ticker(pair)
            else:
                prices["XBTNGN"] = 101425618.0
                prices["ETHNGN"] = 2780346.0
                prices["SOLNGN"] = 220000.0
        except Exception as e:
            logger.error(f"Failed to fetch ticker for {pair}: {e}")
            prices[pair] = 0.0

    ngn_balance, btc_balance, eth_balance, sol_balance = 0.0, 0.0, 0.0, 0.0
    
    if connector and not paper_mode:
        # CRITICAL FIX: Direct live fetch with zero silent fallbacks. 
        # If Luno API fails here, it raises the exception so we know immediately.
        logger.info("[LUNO API] Fetching live account balances directly from exchange...")
        balances = connector.fetch_balances()
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
    else:
        if not paper_mode:
            raise RuntimeError("CRITICAL: Live mode requested, but Luno connector is uninitialized.")
        initial_cash = 1500.0
        initial_units = {"XBTNGN": 0.00025, "ETHNGN": 0.0065, "SOLNGN": 0.05}
        ngn_balance, units = get_paper_portfolio(initial_cash, initial_units)
        btc_balance = units.get("XBTNGN", 0.0)
        eth_balance = units.get("ETHNGN", 0.0)
        sol_balance = units.get("SOLNGN", 0.0)

    holdings_units = {"XBTNGN": btc_balance, "ETHNGN": eth_balance, "SOLNGN": sol_balance}
    holdings_val = {pair: holdings_units[pair] * prices.get(pair, 0.0) for pair in tickers}
    total_crypto_val = sum(holdings_val.values())
    
    portfolio_val = total_crypto_val

    if portfolio_val <= 0:
        logger.error("CRITICAL: Total crypto portfolio valuation evaluated to zero. Aborting execution.")
        return

    logger.info(f"[AUDIT] Live Crypto Portfolio Valuation (TPV): ₦{portfolio_val:,.2f} | Reserve NGN: ₦{ngn_balance:,.2f}")
    
    curr_weights_pct = {pair: (holdings_val[pair] / portfolio_val * 100) if portfolio_val > 0 else 0.0 for pair in tickers}
    logger.info(f"[AUDIT] Live Current Weights -> BTC: {curr_weights_pct.get('XBTNGN', 0):.2f}% | ETH: {curr_weights_pct.get('ETHNGN', 0):.2f}% | SOL: {curr_weights_pct.get('SOLNGN', 0):.2f}%")

    # ==========================================
    # 3. EVALUATE REBALANCE DRIFT
    # ==========================================
    evaluation = rebalancer.evaluate_rebalance(holdings_val, ngn_balance, portfolio_val)
    
    if not evaluation.get("rebalance_triggered", False):
        logger.info(f"[REBALANCE] Portfolio drift ({evaluation.get('max_drift_percentage', 0.0)}%) is within the 3.0% tolerance threshold. Holding position.")
        logger.info("[COMPLETE] Institutional rebalancing audit cycle finished successfully.")
        return

    logger.info(f"✅ Drift exceeds threshold ({evaluation.get('max_drift_percentage', 0.0)}%). Initiating Two-Phase Cashless Rebalance Sequence.")

    # ==========================================
    # 4. PHASE 1: LIQUIDATE OVERWEIGHT ASSETS (SELL LEG)
    # ==========================================
    sells_executed = False
    internal_proceeds_ngn = 0.0

    for pair in tickers:
        current_w = curr_weights_pct.get(pair, 0.0)
        target_w = target_weights.get(pair, 0.0) * 100
        drift = current_w - target_w  # Positive = Overweight

        if drift > 1.5:  # Overweight threshold trigger
            excess_val = (drift / 100.0) * portfolio_val
            market_price = prices.get(pair, 0.0)
            
            if market_price <= 0:
                continue

            limit_sell_price = market_price * (1.0 - AGGRESSIVE_LIMIT_BUFFER)
            sell_volume = excess_val / limit_sell_price
            min_vol_limit = MIN_BASE_VOLUMES.get(pair, 0.001)

            logger.info(f"[REBALANCE - PHASE 1] Asset {pair} is overweight (Actual: {current_w:.2f}% > Target: {target_w:.1f}% | Drift: +{drift:.2f}%). Limit Sell Price: ₦{limit_sell_price:,.2f}")

            if sell_volume < min_vol_limit:
                logger.warning(f"[COMPLIANCE_GATE] Calculated sell volume ({sell_volume:.6f}) for {pair} is below Luno minimum base volume ({min_vol_limit}). Skipping sell leg.")
                log_audit("LUNO", pair, market_price, excess_val, "SKIPPED", "BELOW_BASE_VOLUME")
                continue

            if excess_val >= MIN_ORDER_NGN:
                res = order_executor.execute_limit_order(
                    pair=pair,
                    side="SELL",
                    volume=sell_volume,
                    price=limit_sell_price
                )
                logger.info(f"[EXECUTION RESULT - LIMIT SELL {pair}]: {res}")
                
                if res.get("status") == "SUCCESS":
                    sells_executed = True
                    net_proceeds = excess_val * (1.0 - FEE_BUFFER_PCT)
                    internal_proceeds_ngn += net_proceeds
                    
                    if paper_mode:
                        holdings_units[pair] -= sell_volume
                        update_paper_portfolio(ngn_balance, holdings_units)

    # ==========================================
    # 5. SETTLEMENT GATE & LEDGER VERIFICATION
    # ==========================================
    logger.info(f"[SETTLEMENT_GATE] Internal liquidation proceeds generated: ₦{internal_proceeds_ngn:,.2f}")
    if sells_executed:
        logger.info("[SETTLEMENT_GATE] Sells confirmed. Proceeding strictly with internal proceeds for Phase 2 reallocation.")
        time.sleep(5)
    else:
        logger.info("[SETTLEMENT_GATE] No liquidations executed. Halting rebalance cycle to prevent cash injection.")
        return

    # ==========================================
    # 6. PHASE 2: REALLOCATE INTERNAL PROCEEDS TO UNDERWEIGHT ASSETS (BUY LEG)
    # ==========================================
    for pair in tickers:
        current_w = curr_weights_pct.get(pair, 0.0)
        target_w = target_weights.get(pair, 0.0) * 100
        drift = target_w - current_w  # Positive = Underweight

        if drift > 1.5:  # Underweight threshold trigger
            deficit_val = (drift / 100.0) * portfolio_val
            market_price = prices.get(pair, 0.0)

            if market_price <= 0 or internal_proceeds_ngn < MIN_ORDER_NGN:
                logger.info(f"[REBALANCE - PHASE 2] Insufficient internal liquidation proceeds (₦{internal_proceeds_ngn:,.2f}) to fund underweight asset {pair}.")
                continue

            limit_buy_price = market_price * (1.0 + AGGRESSIVE_LIMIT_BUFFER)
            allocation_val = min(deficit_val, internal_proceeds_ngn)

            if allocation_val >= MIN_ORDER_NGN:
                buy_volume = allocation_val / limit_buy_price
                logger.info(f"[REBALANCE - PHASE 2] Asset {pair} is underweight (Actual: {current_w:.2f}% < Target: {target_w:.1f}%). Limit Buy Price: ₦{limit_buy_price:,.2f} | Allocation: ₦{allocation_val:,.2f}")
                
                res = order_executor.execute_limit_order(
                    pair=pair,
                    side="BUY",
                    volume=buy_volume,
                    price=limit_buy_price
                )
                logger.info(f"[EXECUTION RESULT - LIMIT BUY {pair}]: {res}")

                if res.get("status") == "SUCCESS":
                    total_cost_deduction = allocation_val * (1.0 + FEE_BUFFER_PCT)
                    internal_proceeds_ngn -= total_cost_deduction
                    
                    if paper_mode:
                        holdings_units[pair] += buy_volume
                        update_paper_portfolio(ngn_balance, holdings_units)

    logger.info("[COMPLETE] Institutional cashless rebalancing audit cycle finished successfully.")

if __name__ == "__main__":
    run_institutional_engine()
