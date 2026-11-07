import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("AutonomousRebalancer")

class AutonomousRebalancer:
    """
    Enforces disciplined 50/30/20 portfolio rebalancing[cite: 1] without emotional bias.
    Applies the 3% drift rule[cite: 1], directional constraints, and automated NGN treasury cash sweeps.
    """

    def __init__(self, target_weights: dict = None, drift_threshold: float = 0.03, min_cash_reserve_pct: float = 0.10, fee_rate: float = 0.001):
        self.target_weights = target_weights or {"XBTNGN": 0.50, "ETHNGN": 0.30, "SOLNGN": 0.20}
        self.drift_threshold = drift_threshold        # 3% drift tolerance[cite: 1]
        self.min_cash_reserve_pct = min_cash_reserve_pct # Minimum 10% NGN reserve[cite: 1]
        self.fee_rate = fee_rate                      # Exchange fee simulation

    def evaluate_rebalance(self, holdings_val: dict, ngn_balance: float, total_portfolio_value: float) -> dict:
        """
        Evaluates portfolio drift, computes required buy/sell orders, 
        and enforces institutional rebalancing rules and treasury cash sweeps[cite: 1].
        """
        if total_portfolio_value <= 0:
            return {"status": "ERROR", "reason": "Zero or negative portfolio valuation."}

        current_weights = {pair: val / total_portfolio_value for pair, val in holdings_val.items()}
        current_cash_pct = ngn_balance / total_portfolio_value
        
        actions = {}
        max_drift = 0.0
        rebalance_required = False
        treasury_sweep_action = None

        # 1. Treasury NGN Cash Reserve Check (ICIE Section 5)
        if current_cash_pct < self.min_cash_reserve_pct:
            deficit_val = (self.min_cash_reserve_pct * total_portfolio_value) - ngn_balance
            treasury_sweep_action = {
                "sweep_type": "RECAPITALIZE_CASH",
                "required_cash_injection": round(deficit_val, 2),
                "reason": f"NGN cash reserve ({current_cash_pct*100:.1f}%) is below the minimum threshold ({self.min_cash_reserve_pct*100}%). Harvesting profits from overweight assets."
            }
            rebalance_required = True

        for pair, target in self.target_weights.items():
            current = current_weights.get(pair, 0.0)
            drift = current - target
            max_drift = max(max_drift, abs(drift))

            current_asset_val = holdings_val.get(pair, 0.0)
            target_asset_val = total_portfolio_value * target
            diff_val = target_asset_val - current_asset_val

            # Enforce Rebalancing Rules:
            # - Do nothing if drift < 3% and cash reserve is healthy[cite: 1]
            # - Sell only overweight assets (diff_val < 0) -> Proceeds sweep to NGN wallet[cite: 1]
            # - Buy only underweight assets (diff_val > 0) -> Funded by NGN reserve[cite: 1]
            if abs(drift) >= self.drift_threshold or treasury_sweep_action is not None:
                rebalance_required = True
                if diff_val < 0:
                    # Overweight -> Sell excess and sweep proceeds to NGN reserve
                    excess_val = abs(diff_val)
                    actions[pair] = {
                        "action": "SELL",
                        "target_adjustment_val": excess_val,
                        "estimated_fee": excess_val * self.fee_rate,
                        "treasury_effect": "PROCEEDS_SWAPTED_TO_NGN_WALLET",
                        "reason": f"Drift {drift*100:.2f}% exceeds threshold[cite: 1]. Selling excess to recapitalize NGN."
                    }
                elif diff_val > 0:
                    # Underweight -> Buy deficit (subject to available NGN cash reserve)
                    allocatable_cash = min(diff_val, ngn_balance * (1.0 - self.min_cash_reserve_pct))
                    if allocatable_cash > 50: # Minimum order threshold ₦50[cite: 1]
                        actions[pair] = {
                            "action": "BUY",
                            "target_adjustment_val": allocatable_cash,
                            "estimated_fee": allocatable_cash * self.fee_rate,
                            "treasury_effect": "FUNDED_FROM_NGN_RESERVE",
                            "reason": f"Drift {drift*100:.2f}% requires accumulation[cite: 1]."
                        }
                    else:
                        actions[pair] = {
                            "action": "HOLD", 
                            "target_adjustment_val": 0.0, 
                            "reason": "Underweight, but restricted by minimum NGN cash reserve buffer."
                        }
            else:
                actions[pair] = {
                    "action": "HOLD", 
                    "target_adjustment_val": 0.0, 
                    "reason": "Drift within 3% tolerance[cite: 1]."
                }

        return {
            "status": "SUCCESS",
            "rebalance_triggered": rebalance_required,
            "max_drift_percentage": round(max_drift * 100, 2),
            "current_cash_percentage": round(current_cash_pct * 100, 2),
            "treasury_sweep": treasury_sweep_action,
            "current_weights": {k: round(v * 100, 2) for k, v in current_weights.items()},
            "target_weights": {k: round(v * 100, 2) for k, v in self.target_weights.items()},
            "execution_plan": actions
        }
