import os
import time
import logging
import pandas as pd
from datetime import datetime

from core.config import load_config
from core.logging import setup_logger
from database.repository import init_database, log_audit, get_paper_portfolio, update_paper_portfolio, record_order
from execution.luno_connector import LunoConnector
from execution.order_executor import InstitutionalOrderExecutor
from risk.circuit_breakers import CircuitBreakerManager
from strategy.rebalancer import AutonomousRebalancer

# Initialize Logger
logger = setup_logger("ICIEEngine")

# Luno Exchange Minimum Base Volume Limits (Strict Exchange Compliance)
MIN_BASE_VOLUMES = {
    "BTCNGN": 0.0005,
    "ETHNGN": 0.005,
    "SOLNGN": 0.05
}

# Strict Minimum Order Threshold in NGN
MIN_ORDER_NGN = 50.0

def run_institutional_engine():
    """
    Executes the ICIE Closed-Loop Autonomous Rebalancing Engine.
    Enforces exchange compliance, minimum base volumes, and self-funding NGN recycling.
    """
    logger.info("[START] Initializing ICIE Autonomous Institutional Engine (Luno Node)...")
    
    # 1. Initialize Database & Config
    init_database()
    config = load_config()
    
    # Credentials Check from Environment or Streamlit Secrets
    luno_key = os.environ.get("LUNO_API_KEY_ID")
    luno_secret = os.environ.get("LUNO_API_KEY_SECRET")
    
    paper_mode = config.get("paper_mode", True)
    if not luno_key or not luno_secret:
        logger.warning("⚠️ Luno API credentials not found in environment. Forcing Paper Trading / Simulation Mode.")
        paper_mode = True

    connector = LunoConnector(luno_key, luno_secret) if not paper_mode else None
    circuit_breaker = CircuitBreakerManager()
    order_executor = InstitutionalOrderExecutor(connector, circuit_breaker)
    
    target_weights = config.get("target_weights", {"XBTNGN": 0.50, "ETHNGN": 0.30, "SOLNGN": 0.20})
    rebalancer = AutonomousRebalancer(target_weights=target_weights)

    tickers = ["XBTNGN", "ETHNGN", "SOLNGN"]
    
    # 2. Fetch Live Prices and Balances
    prices = {}
    for pair in tickers:
        try:
            if connector:
                prices[pair] = connector.fetch_ticker(pair)
            else:
                # Simulated live baseline prices if offline
                prices["XBTNGN"] = 101425618.0
                prices["ETHNGN"] = 2780346.0
                prices["SOLNGN"] = 220000.0
        except Exception as e:
            logger.error(f"Failed to fetch ticker for {pair}: {e}")
            prices[pair] = 0.0

    # Fetch Balances (Live or Paper Portfolio)
    ngn_balance, btc_balance, eth_balance, sol_balance = 0.0, 0.0, 0.0, 0.0
    if connector and not paper_mode:
        try:
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
        except Exception as e:
            logger.error(f"Failed to fetch live balances: {e}")
    else:
        # Load Paper Portfolio baseline matching user state
        initial_cash = 34.01
        initial_units = {"XBTNGN": 0.0003, "ETHNGN": 0.0100, "SOLNGN": 0.0753}
        ngn_balance, units = get_paper_portfolio(initial_cash, initial_units)
        btc_balance = units.get("XBTNGN", 0.0)
        eth_balance = units.get("ETHNGN", 0.0)
        sol_balance = units.get("SOLNGN", 0.0)

    holdings_units = {"XBTNGN": btc_balance, "ETHNGN": eth_balance, "SOLNGN": sol_balance}
    holdings_val = {pair: holdings_units[pair] * prices.get(pair, 0.0) for pair in tickers}
    total_crypto_val = sum(holdings_val.values())
    portfolio_val = ngn_balance + total_crypto_val

    logger.info(f"[AUDIT] Portfolio Value: ₦{portfolio_val:,.2f} | NGN Cash: ₦{ngn_balance:,.2f}")
    
    curr_weights_pct = {pair: (holdings_val[pair] / portfolio_val * 100) if portfolio_val > 0 else 0.0 for pair in tickers}
    logger.info(f"[AUDIT] Current Weights -> BTC: {curr_weights_pct.get('XBTNGN', 0):.1f}% | ETH: {curr_weights_pct.get('ETHNGN', 0):.1f}% | SOL: {curr_weights_pct.get('SOLNGN', 0):.1f}%")

    # 3. Evaluate Rebalance Drift
    evaluation = rebalancer.evaluate_rebalance(holdings_val, ngn_balance, portfolio_val)
    
    if not evaluation.get("rebalance_triggered", False):
        logger.info(f"[REBALANCE] Portfolio drift ({evaluation.get('max_drift_percentage', 0.0)}%) is within the 3.0% tolerance threshold. No action required.")
        return

    logger.info(f"✅ Drift exceeds threshold ({evaluation.get('max_drift_percentage', 0.0)}%). Executing Closed-Loop Rebalance Sequence.")

    # 4. Sequential Execution: Prioritize Overweight Asset Liquidation (Sell Leg) First
    for pair in tickers:
        current_w = curr_weights_pct.get(pair, 0.0)
        target_w = target_weights.get(pair, 0.0) * 100
        drift = current_w - target_w  # Positive drift = Overweight

        if drift > 1.5:  # Overweight threshold trigger
            excess_val = (drift / 100.0) * portfolio_val
            asset_price = prices.get(pair, 0.0)
            
            if asset_price <= 0:
                continue

            sell_volume = excess_val / asset_price
            min_vol_limit = MIN_BASE_VOLUMES.get(pair, 0.001)

            logger.info(f"[REBALANCE] Asset {pair} is overweight (Actual: {current_w:.1f}% > Target: {target_w:.1f}% | Drift: +{drift:.1f}%). Initiating Sell Leg.")

            # Enforce Luno Minimum Base Volume Gate before hitting connector
            if sell_volume < min_vol_limit:
                logger.warning(f"[COMPLIANCE_GATE] Calculated sell volume ({sell_volume:.6f}) for {pair} is below Luno minimum base volume ({min_vol_limit}). Skipping sell leg until accumulation threshold is met.")
                log_audit("LUNO", pair, asset_price, excess_val, "SKIPPED", "BELOW_BASE_VOLUME")
                continue

            if excess_val >= MIN_ORDER_NGN:
                res = order_executor.execute_market_order(
                    pair=pair,
                    side="SELL",
                    target_val=excess_val,
                    current_price=asset_price
                )
                logger.info(f"[EXECUTION RESULT - SELL {pair}]: {res}")
                
                # If paper mode, update internal paper ledger
                if paper_mode and res.get("status") == "SUCCESS":
                    ngn_balance += excess_val
                    holdings_units[pair] -= sell_volume
                    update_paper_portfolio(ngn_balance, holdings_units)

    # 5. Proportional Redistribution: Use Internal NGN Liquidity for Underweight Assets (Buy Leg)
    # Refresh NGN balance after internal liquidations
    for pair in tickers:
        current_w = curr_weights_pct.get(pair, 0.0)
        target_w = target_weights.get(pair, 0.0) * 100
        drift = target_w - current_w  # Positive drift = Underweight

        if drift > 1.5:  # Underweight threshold trigger
            deficit_val = (drift / 100.0) * portfolio_val
            asset_price = prices.get(pair, 0.0)

            if asset_price <= 0 or ngn_balance < MIN_ORDER_NGN:
                continue

            # Allocate available internal NGN cash up to the required deficit
            allocation_val = min(deficit_val, ngn_balance)

            if allocation_val >= MIN_ORDER_NGN:
                logger.info(f"[REBALANCE] Asset {pair} is underweight (Actual: {current_w:.1f}% < Target: {target_w:.1f}%). Initiating Buy Leg with internal NGN liquidity: ₦{allocation_val:,.2f}")
                
                res = order_executor.execute_market_order(
                    pair=pair,
                    side="BUY",
                    target_val=allocation_val,
                    current_price=asset_price
                )
                logger.info(f"[EXECUTION RESULT - BUY {pair}]: {res}")

                if paper_mode and res.get("status") == "SUCCESS":
                    ngn_balance -= allocation_val
                    bought_units = allocation_val / asset_price
                    holdings_units[pair] += bought_units
                    update_paper_portfolio(ngn_balance, holdings_units)

    logger.info("[COMPLETE] Institutional rebalancing audit cycle finished successfully.")

if __name__ == "__main__":
    run_institutional_engine()
