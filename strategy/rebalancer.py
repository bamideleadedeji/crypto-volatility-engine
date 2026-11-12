import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("AutonomousRebalancer")

class AutonomousRebalancer:
    """
    Enforces disciplined 50/30/20 portfolio rebalancing without cash injection or cash drag.
    Applies the 3% drift rule, directional constraints, and a two-phase self-funding
    asset-to-asset rotation (Sells generate internal NGN proceeds, Buys consume those proceeds),
    strictly adhering to Luno's ₦50 minimum order threshold.
    """

    def __init__(self, target_weights: dict = None, drift_threshold: float = 0.03, fee_rate: float = 0.001):
        self.target_weights = target_weights or {"XBTNGN": 0.50, "ETHNGN": 0.30, "SOLNGN": 0.20}
        self.drift_threshold = drift_threshold  # 3% drift tolerance
        self.fee_rate = fee_rate                # Exchange fee simulation buffer

    def evaluate_rebalance(self, holdings_val: dict, ngn_balance: float, total_portfolio_value: float) -> dict:
        """
        Evaluates crypto portfolio drift against 50/30/20 targets, structures a two-phase 
        cashless execution plan (Sells first, then Buys using internal proceeds), 
        and eliminates external cash dependency.
        """
        # CRITICAL INSTITUTIONAL FIX: Evaluate TPV strictly from crypto holdings 
        # to ensure 50/30/20 allocation is never distorted by stagnant NGN cash balances.
        total_crypto_val = sum(holdings_val.values())

        if total_crypto_val <= 0:
            return {"status": "ERROR", "reason": "Zero or negative crypto portfolio valuation."}

        current_weights = {pair: val / total_crypto_val for pair, val in holdings_val.items()}
        current_cash_pct = (ngn_balance / (total_crypto_val + ngn_balance)) if (total_crypto_val + ngn_balance) > 0 else 0.0
        
        sell_actions = {}
        buy_actions = {}
        hold_actions = {}
        max_drift = 0.0
        rebalance_required = False

        # First Pass: Classify Sells and Buys separately for strict two-phase sequencing
        for pair, target in self.target_weights.items():
            current = current_weights.get(pair, 0.0)
            drift = current - target
            max_drift = max(max_drift, abs(drift))

            current_asset_val = holdings_val.get(pair, 0.0)
            target_asset_val = total_crypto_val * target
            diff_val = target_asset_val - current_asset_val  # Positive = Underweight (Buy), Negative = Overweight (Sell)

            if abs(drift) >= self.drift_threshold:
                rebalance_required = True
                if diff_val < 0:
                    # Overweight -> Phase 1 SELL (Generates internal settlement proceeds)
                    excess_val = abs(diff_val)
                    if excess_val >= 50.0:  # Luno minimum order threshold ₦50
                        sell_actions[pair] = {
                            "action": "SELL",
                            "phase": 1,
                            "target_adjustment_val": excess_val,
                            "estimated_fee": excess_val * self.fee_rate,
                            "treasury_effect": "PROCEEDS_LOCKED_FOR_CASHLESS_REBALANCE",
                            "reason": f"Drift {drift*100:.2f}% exceeds 3% threshold. Liquidating overweight excess."
                        }
                    else:
                        hold_actions[pair] = {
                            "action": "HOLD",
                            "phase": 0,
                            "target_adjustment_val": 0.0,
                            "reason": "Overweight drift, but below Luno's ₦50 minimum order threshold."
                        }
                elif diff_val > 0:
                    # Underweight -> Phase 2 BUY (Funded exclusively by internal Phase 1 sell proceeds)
                    if diff_val >= 50.0:  # Luno minimum order threshold ₦50
                        buy_actions[pair] = {
                            "action": "BUY",
                            "phase": 2,
                            "target_adjustment_val": diff_val,
                            "estimated_fee": diff_val * self.fee_rate,
                            "treasury_effect": "FUNDED_VIA_INTERNAL_SELL_PROCEEDS",
                            "reason": f"Drift {drift*100:.2f}% requires cashless accumulation."
                        }
                    else:
                        hold_actions[pair] = {
                            "action": "HOLD",
                            "phase": 0,
                            "target_adjustment_val": 0.0,
                            "reason": "Underweight drift, but below Luno's ₦50 minimum order threshold."
                        }
            else:
                hold_actions[pair] = {
                    "action": "HOLD",
                    "phase": 0,
                    "target_adjustment_val": 0.0,
                    "reason": "Drift within 3% tolerance."
                }

        # Combine into a strictly ordered execution plan: Sells first, then Buys, then Holds
        execution_plan = {}
        for p, d in sell_actions.items():
            execution_plan[p] = d
        for p, d in buy_actions.items():
            execution_plan[p] = d
        for p, d in hold_actions.items():
            execution_plan[p] = d

        return {
            "status": "SUCCESS",
            "rebalance_triggered": rebalance_required,
            "max_drift_percentage": round(max_drift * 100, 2),
            "current_cash_percentage": round(current_cash_pct * 100, 2),
            "treasury_sweep": None,  # Disabled to prevent external cash injection interference
            "current_weights": {k: round(v * 100, 2) for k, v in current_weights.items()},
            "target_weights": {k: round(v * 100, 2) for k, v in self.target_weights.items()},
            "execution_plan": execution_plan
        }
