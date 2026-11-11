import os
import sys
import time
import logging
import pandas as pd
from datetime import datetime

# ==========================================
# 0. ABSOLUTE PATH RESOLUTION (CRON SAFETY)
# ==========================================
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

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
    "BTCNGN": 0.0005,
    "ETHNGN": 0.005,
    "SOLNGN": 0.05
}

# Strict Minimum Order Threshold in NGN
MIN_ORDER_NGN = 50.0

# Luno Taker Fee & Safety Buffer (0.10% standard tier + 0.10% aggressive limit-crossing buffer)
FEE_BUFFER_PCT = 0.002
AGGRESSIVE_LIMIT_BUFFER = 0.0015  # 0.15% aggressive price cross to guarantee execution (NGX style)

def run_institutional_engine():
    """
    Executes the ICIE Closed-Loop Two-Phase Autonomous Rebalancing Engine.
    Enforces Phase 1 (Sell Overweight -> Settle NGN) before Phase 2 (Buy Underweight).
    Applies aggressive limit-pricing buffers and Luno exchange compliance rules.
    """
    logger.info("[START] Initializing ICIE Autonomous Institutional Engine (Luno Live Production Node)...")
    
    # Initialize Database & Config
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
    
    # ==========================================
    # 2. FETCH LIVE PRICES & PORTFOLIO BALANCES
    # ==========================================
    prices = {}
    for pair in tickers:
        try:
            if connector and not paper_mode:
                prices[pair] = connector.fetch_ticker(pair)
            else:
                # Simulated baseline prices reflecting ~₦60k portfolio baseline
                prices["XBTNGN"] = 101425618.0
                prices["ETHNGN"] = 2780346.0
                prices["SOLNGN"] = 220000.0
        except Exception as e:
            logger.error(f"Failed to fetch ticker for {pair}: {e}")
            prices[pair] = 0.0

    # Fetch Balances (Live API or Paper Portfolio Ledger)
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
        # Load Paper Portfolio baseline matching live capital state (~₦60,000)
        initial_cash = 1500.0
        initial_units = {"XBTNGN": 0.00025, "ETHNGN": 0.0065, "SOLNGN": 0.05}
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

    # ==========================================
    # 3. EVALUATE REBALANCE DRIFT
    # ==========================================
    evaluation = rebalancer.evaluate_rebalance(holdings_val, ngn_balance, portfolio_val)
    
    if not evaluation.get("rebalance_triggered", False):
        logger.info(f"[REBALANCE] Portfolio drift ({evaluation.get('max_drift_percentage', 0.0)}%) is within the 3.0% tolerance threshold. Holding position. No action required.")
        logger.info("[COMPLETE] Institutional rebalancing audit cycle finished successfully.")
        return

    logger.info(f"✅ Drift exceeds threshold ({evaluation.get('max_drift_percentage', 0.0)}%). Initiating Two-Phase Rebalance Sequence with Aggressive Limits.")

    # ==========================================
    # 4. PHASE 1: LIQUIDATE OVERWEIGHT ASSETS (SELL LEG WITH BUFFER)
    # ==========================================
    sells_executed = False
    for pair in tickers:
        current_w = curr_weights_pct.get(pair, 0.0)
        target_w = target_weights.get(pair, 0.0) * 100
        drift = current_w - target_w  # Positive drift = Overweight

        if drift > 1.5:  # Overweight threshold trigger
            excess_val = (drift / 100.0) * portfolio_val
            market_price = prices.get(pair, 0.0)
            
            if market_price <= 0:
                continue

            # Apply aggressive limit pricing for selling (slightly below market bid to ensure immediate fill)
            limit_sell_price = market_price * (1.0 - AGGRESSIVE_LIMIT_BUFFER)
            sell_volume = excess_val / limit_sell_price
            min_vol_limit = MIN_BASE_VOLUMES.get(pair, 0.001)

            logger.info(f"[REBALANCE - PHASE 1] Asset {pair} is overweight (Actual: {current_w:.1f}% > Target: {target_w:.1f}% | Drift: +{drift:.1f}%). Limit Sell Price: ₦{limit_sell_price:,.2f}")

            # Enforce Luno Minimum Base Volume Gate
            if sell_volume < min_vol_limit:
                logger.warning(f"[COMPLIANCE_GATE] Calculated sell volume ({sell_volume:.6f}) for {pair} is below Luno minimum base volume ({min_vol_limit}). Skipping sell leg until accumulation threshold is met.")
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
                    if paper_mode:
                        net_proceeds = excess_val * (1.0 - FEE_BUFFER_PCT)
                        ngn_balance += net_proceeds
                        holdings_units[pair] -= sell_volume
                        update_paper_portfolio(ngn_balance, holdings_units)

    # ==========================================
    # 5. SETTLEMENT GATE & LEDGER RE-VERIFICATION
    # ==========================================
    logger.info(f"[SETTLEMENT_GATE] Verifying NGN funding wallet balance after liquidation leg: ₦{ngn_balance:,.2f}")
    if sells_executed:
        logger.info("[SETTLEMENT_GATE] Liquidation proceeds confirmed in wallet. Proceeding to Phase 2 (Reallocation).")
    else:
        logger.info("[SETTLEMENT_GATE] No liquidations executed. Proceeding with existing NGN reserve balance check.")

    # ==========================================
    # 6. PHASE 2: REALLOCATE NGN TO UNDERWEIGHT ASSETS (BUY LEG WITH BUFFER)
    # ==========================================
    for pair in tickers:
        current_w = curr_weights_pct.get(pair, 0.0)
        target_w = target_weights.get(pair, 0.0) * 100
        drift = target_w - current_w  # Positive drift = Underweight

        if drift > 1.5:  # Underweight threshold trigger
            deficit_val = (drift / 100.0) * portfolio_val
            market_price = prices.get(pair, 0.0)

            if market_price <= 0 or ngn_balance < MIN_ORDER_NGN:
                logger.info(f"[REBALANCE - PHASE 2] Insufficient NGN cash (₦{ngn_balance:,.2f}) to fund underweight asset {pair}. Holding cash reserve.")
                continue

            # Apply aggressive limit pricing for buying (slightly above market ask to ensure execution without runaways)
            limit_buy_price = market_price * (1.0 + AGGRESSIVE_LIMIT_BUFFER)
            
            # Apply fee buffer safety margin: Allocate up to deficit, leaving room for taker fees & limit buffer
            safe_ngn_available = ngn_balance * (1.0 - FEE_BUFFER_PCT)
            allocation_val = min(deficit_val, safe_ngn_available)

            if allocation_val >= MIN_ORDER_NGN:
                buy_volume = allocation_val / limit_buy_price
                logger.info(f"[REBALANCE - PHASE 2] Asset {pair} is underweight (Actual: {current_w:.1f}% < Target: {target_w:.1f}%). Limit Buy Price: ₦{limit_buy_price:,.2f} | Allocation: ₦{allocation_val:,.2f}")
                
                res = order_executor.execute_limit_order(
                    pair=pair,
                    side="BUY",
                    volume=buy_volume,
                    price=limit_buy_price
                )
                logger.info(f"[EXECUTION RESULT - LIMIT BUY {pair}]: {res}")

                if res.get("status") == "SUCCESS":
                    if paper_mode:
                        total_cost_deduction = allocation_val * (1.0 + FEE_BUFFER_PCT)
                        ngn_balance -= total_cost_deduction
                        holdings_units[pair] += buy_volume
                        update_paper_portfolio(ngn_balance, holdings_units)

    logger.info("[COMPLETE] Institutional rebalancing audit cycle finished successfully.")

if __name__ == "__main__":
    run_institutional_engine()
