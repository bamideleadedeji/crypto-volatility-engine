import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("AutonomousRebalancer")

class AutonomousRebalancer:
    """
    Enforces disciplined 50/30/20 portfolio rebalancing[cite: 1] without emotional bias.
    Applies the 3% drift rule[cite: 1], directional constraints, and NGN treasury sweeps.
    """

    def __init__(self, target_weights: dict = None, drift_threshold: float = 0.03, fee_rate: float = 0.001):
        self.target_weights = target_weights or {"XBTNGN": 0.50, "ETHNGN": 0.30, "SOLNGN": 0.20}
        self.drift_threshold = drift_threshold  # 3% drift tolerance[cite: 1]
        self.fee_rate = fee_rate  # Exchange fee simulation

    def evaluate_rebalance(self, holdings_val: dict, ngn_balance: float, total_portfolio_value: float) -> dict:
        """
        Evaluates portfolio drift, computes required buy/sell orders, 
        and enforces institutional rebalancing rules.
        """
        if total_portfolio_value <= 0:
            return {"status": "ERROR", "reason": "Zero or negative portfolio valuation."}

        current_weights = {pair: val / total_portfolio_value for pair, val in holdings_val.items()}
        
        actions = {}
        max_drift = 0.0
        rebalance_required = False

        for pair, target in self.target_weights.items():
            current = current_weights.get(pair, 0.0)
            drift = current - target
            max_drift = max(max_drift, abs(drift))

            current_asset_val = holdings_val.get(pair, 0.0)
            target_asset_val = total_portfolio_value * target
            diff_val = target_asset_val - current_asset_val

            # Enforce Rebalancing Rules:
            # - Do nothing if drift < 3%[cite: 1]
            # - Sell only overweight assets (diff_val < 0)
            # - Buy only underweight assets (diff_val > 0)
            if abs(drift) >= self.drift_threshold:
                rebalance_required = True
                if diff_val < 0:
                    # Overweight -> Sell excess
                    actions[pair] = {
                        "action": "SELL",
                        "target_adjustment_val": abs(diff_val),
                        "estimated_fee": abs(diff_val) * self.fee_rate,
                        "reason": f"Drift {drift*100:.2f}% exceeds threshold."
                    }
                elif diff_val > 0:
                    # Underweight -> Buy deficit (subject to available NGN cash reserve)
                    allocatable_cash = min(diff_val, ngn_balance * 0.9) # Reserve 10% NGN cash buffer
                    if allocatable_cash > 50: # Minimum order threshold ₦50[cite: 1]
                        actions[pair] = {
                            "action": "BUY",
                            "target_adjustment_val": allocatable_cash,
                            "estimated_fee": allocatable_cash * self.fee_rate,
                            "reason": f"Drift {drift*100:.2f}% requires accumulation."
                        }
                    else:
                        actions[pair] = {"action": "HOLD", "target_adjustment_val": 0.0, "reason": "Insufficient liquid NGN cash reserve."}
            else:
                actions[pair] = {"action": "HOLD", "target_adjustment_val": 0.0, "reason": "Drift within 3% tolerance[cite: 1]."}

        return {
            "status": "SUCCESS",
            "rebalance_triggered": rebalance_required,
            "max_drift_percentage": round(max_drift * 100, 2),
            "current_weights": {k: round(v * 100, 2) for k, v in current_weights.items()},
            "target_weights": {k: round(v * 100, 2) for k, v in self.target_weights.items()},
            "execution_plan": actions
        }
