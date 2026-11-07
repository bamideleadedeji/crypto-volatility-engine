import os
import sys
from datetime import datetime

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.config import load_config
from core.logging import setup_logger
from database.repository import init_database, log_audit
from execution.luno_connector import LunoConnector
from execution.order_executor import InstitutionalOrderExecutor
from risk.circuit_breakers import CircuitBreakerManager
from strategy.rebalancer import AutonomousRebalancer

logger = setup_logger("AutonomousEngineRunner")

def run_background_rebalance():
    logger.info("⚡ Starting Autonomous ICIE Background Execution Loop...")
    init_database()
    
    config = load_config()
    circuit_breaker = CircuitBreakerManager()
    rebalancer = AutonomousRebalancer(target_weights=config.get("target_weights", {"XBTNGN": 0.50, "ETHNGN": 0.30, "SOLNGN": 0.20}))
    
    # Load Luno API credentials from environment variables (injected by GitHub Secrets)
    luno_key = os.environ.get("LUNO_API_KEY_ID")
    luno_secret = os.environ.get("LUNO_API_KEY_SECRET")
    
    if not luno_key or not luno_secret:
        logger.error("🚨 Luno API credentials missing from environment secrets. Aborting automated run.")
        return

    connector = LunoConnector(luno_key, luno_secret)
    order_executor = InstitutionalOrderExecutor(connector, circuit_breaker)
    
    tickers = ["XBTNGN", "ETHNGN", "SOLNGN"]
    prices = {}
    for pair in tickers:
        try:
            prices[pair] = connector.fetch_ticker(pair)
        except Exception as e:
            logger.error(f"Failed to fetch ticker for {pair}: {e}")
            prices[pair] = 0.0

    try:
        balances = connector.fetch_balances()
    except Exception as e:
        logger.error(f"Failed to fetch Luno balances: {e}")
        balances = []

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

    logger.info(f"Portfolio Valuation -> Total: ₦{total_portfolio_val:,.2f} | NGN Cash: ₦{ngn_balance:,.2f}")

    evaluation = rebalancer.evaluate_rebalance(holdings_val, ngn_balance, total_portfolio_val)
    logger.info(f"Max Drift Evaluation: {evaluation.get('max_drift_percentage', 0.0)}% | Rebalance Triggered: {evaluation['rebalance_triggered']}")

    if evaluation["rebalance_triggered"]:
        position_caps = {"XBTNGN": 0.60, "ETHNGN": 0.40, "SOLNGN": 0.30}
        execution_plan = evaluation["execution_plan"]
        
        for pair, details in execution_plan.items():
            action = details["action"]
            target_val = details["target_adjustment_val"]
            
            if action == "BUY":
                projected_w = (holdings_val[pair] + target_val) / total_portfolio_val
                max_allowed = position_caps.get(pair, 0.40)
                if projected_w > max_allowed:
                    logger.warning(f"Skipping BUY for {pair}: Projected weight exceeds max limit.")
                    continue

            if action in ["BUY", "SELL"] and target_val > 50:
                res = order_executor.execute_market_order(pair=pair, side=action, counter_volume=str(round(target_val, 2)))
                logger.info(f"Executed live {action} on {pair} for ₦{target_val:,.2f} | Status: {res.get('status')}")
                log_audit(venue="LUNO", asset=pair, price=prices.get(pair, 0.0), amount=target_val, units="LIVE_AUTO", status=f"AUTO_{action}")

        log_audit(venue="SYSTEM", asset="PORTFOLIO", price=0.0, amount=total_portfolio_val, units="ALL", status="AUTO_REBALANCE_SUCCESS")
        logger.info("Autonomous rebalancing cycle completed successfully.")
    else:
        logger.info("Drift within tolerance. No rebalance required.")

if __name__ == "__main__":
    run_background_rebalance()
