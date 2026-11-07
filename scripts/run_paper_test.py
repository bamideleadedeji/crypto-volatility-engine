import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.logging import setup_logger
from database.repository import init_database, log_audit, fetch_audit_ledger, get_paper_portfolio, update_paper_portfolio
from risk.circuit_breakers import CircuitBreakerManager
from strategy.rebalancer import AutonomousRebalancer
from execution.order_executor import InstitutionalOrderExecutor
from execution.broker_interface import BrokerInterface

logger = setup_logger("PaperTestRunner")

class MockBroker(BrokerInterface):
    """Mock broker interface for local paper testing simulation."""
    def __init__(self, mock_prices):
        self.mock_prices = mock_prices

    def fetch_balances(self) -> list:
        return [
            {"asset": "NGN", "balance": "3000000.0"},
            {"asset": "XBT", "balance": "0.075"},
            {"asset": "ETH", "balance": "0.8"},
            {"asset": "SOL", "balance": "12.0"}
        ]

    def fetch_ticker(self, pair: str) -> float:
        return self.mock_prices.get(pair, 1000.0)

    def place_market_order(self, pair: str, side: str, volume: str = None, counter_volume: str = None) -> dict:
        return {"order_id": f"MOCK-ORD-{int(datetime.now().timestamp())}", "closed": True, "pair": pair, "side": side}

    def check_order_status(self, order_id: str) -> str:
        return "COMPLETE"


def run_7_day_paper_simulation():
    print("⚡ Initializing ICIE 7-Day Local Paper Trading Simulation...\n")
    
    # 1. Initialize Database & Services
    init_database()
    circuit_breaker = CircuitBreakerManager()
    rebalancer = AutonomousRebalancer(target_weights={"XBTNGN": 0.50, "ETHNGN": 0.30, "SOLNGN": 0.20})
    
    # Base prices
    base_prices = {"XBTNGN": 65_000_000.0, "ETHNGN": 3_800_000.0, "SOLNGN": 220_000.0}
    mock_broker = MockBroker(base_prices)
    order_executor = InstitutionalOrderExecutor(mock_broker, circuit_breaker)

    # Initial Portfolio Holdings
    ngn_cash = 3_000_000.0
    holdings_units = {"XBTNGN": 0.075, "ETHNGN": 0.8, "SOLNGN": 12.0}

    np.random.seed(42)
    start_date = datetime.now() - timedelta(days=7)

    for day in range(1, 8):
        current_date = start_date + timedelta(days=day)
        print(f"--- Simulation Day {day} ({current_date.strftime('%Y-%m-%d')}) ---")

        # Simulate minor daily price drift (-3% to +4%)
        for pair in base_prices:
            shock = np.random.normal(0.005, 0.025)
            base_prices[pair] *= (1.0 + shock)

        # Calculate current asset valuations
        holdings_val = {pair: holdings_units[pair] * base_prices[pair] for pair in base_prices}
        total_crypto_val = sum(holdings_val.values())
        total_portfolio_val = ngn_cash + total_crypto_val

        print(f"💰 Portfolio Value: ₦{total_portfolio_val:,.2f} | NGN Cash: ₦{ngn_cash:,.2f}")

        # Evaluate Rebalance & Drift
        evaluation = rebalancer.evaluate_rebalance(holdings_val, ngn_cash, total_portfolio_val)
        print(f"📊 Max Drift: {evaluation['max_drift_percentage']}% | Rebalance Triggered: {evaluation['rebalance_triggered']}")

        if evaluation["rebalance_triggered"]:
            execution_plan = evaluation["execution_plan"]
            for pair, details in execution_plan.items():
                action = details["action"]
                target_val = details["target_adjustment_val"]

                if action in ["BUY", "SELL"] and target_val > 50:
                    res = order_executor.execute_market_order(pair=pair, side=action, counter_volume=str(round(target_val, 2)))
                    print(f"   -> Executed {action} on {pair} for ₦{target_val:,.2f} | Status: {res['status']}")
                    
                    # Update simulated units
                    current_price = base_prices[pair]
                    if action == "BUY":
                        units_bought = target_val / current_price
                        holdings_units[pair] += units_bought
                        ngn_cash -= target_val
                    elif action == "SELL":
                        units_sold = target_val / current_price
                        holdings_units[pair] = max(0.0, holdings_units[pair] - units_sold)
                        ngn_cash += target_val # Treasury NGN Cash Sweep recapitalization
        print("")

    # 3. Verify SQLite WAL Audit Ledger
    print("=" * 60)
    print("📜 Verifying SQLite WAL Audit Ledger Entries:")
    audit_df = fetch_audit_ledger(10)
    if not audit_df.empty:
        print(audit_df.to_string(index=False))
    else:
        print("⚠️ No audit records found.")
    print("=" * 60)
    print("✅ Local Paper Trading Simulation Completed Successfully.")

if __name__ == "__main__":
    run_7_day_paper_simulation()
