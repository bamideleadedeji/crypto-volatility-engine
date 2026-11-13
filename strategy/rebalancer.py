"""
Institutional Crypto Investment Engine (ICIE) - Autonomous Rebalancer
Target Asset Allocation: BTC 50%, ETH 30%, SOL 20%
Core Logic: Conservative accounting (balance + reserved), dynamic exchange-floor drift thresholds, 
cashless two-phase rebalancing (sell overweight -> internal NGN proceeds -> buy underweight),
and strict Luno exchange minimum base volume & fiat floor compliance gates.
"""

import logging
from decimal import Decimal, ROUND_DOWN
from execution.luno_connector import LunoConnector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ICIERebalancer")

TARGET_WEIGHTS = {
    'XBT': Decimal('0.50'),
    'ETH': Decimal('0.30'),
    'SOL': Decimal('0.20')
}

MIN_FIAT_NGN = Decimal('50.00')        # Luno absolute minimum fiat order value

# Luno hard exchange base volume floors per asset
MIN_BASE_VOLUMES = {
    'XBT': Decimal('0.000025'),
    'ETH': Decimal('0.001'),
    'SOL': Decimal('0.05')
}

class ICIERebalancer:
    def __init__(self, api_key_id: str, api_key_secret: str, dry_run: bool = True):
        self.connector = LunoConnector(api_key_id, api_key_secret)
        self.dry_run = dry_run
        self.target_weights = TARGET_WEIGHTS

    def fetch_live_portfolio(self):
        """
        Fetches live wallet balances (balance + reserved) and real-time spot prices
        to establish bonafide portfolio valuation matching Luno exchange reality.
        """
        return self.connector.get_portfolio_valuation()

    def evaluate_drift(self, portfolio_state):
        """
        Evaluates actual weights against target weights (50/30/20) using dynamic 
        exchange-floor drift thresholds to prevent sub-minimum order rejections.
        """
        total = portfolio_state['total_portfolio']
        if total <= 0:
            logger.warning("Portfolio total capital is zero or negative. Halting rebalance.")
            return False, {}

        drift_detected = False
        drift_analysis = {}

        for asset, target_weight in TARGET_WEIGHTS.items():
            val = portfolio_state['asset_values'].get(asset, Decimal('0'))
            actual_weight = val / total if total > 0 else Decimal('0')
            drift = actual_weight - target_weight
            abs_drift = abs(drift)

            # Dynamic Exchange Floor Threshold Calculation:
            price = portfolio_state['spot_prices'].get(asset, Decimal('1'))
            min_base_vol = MIN_BASE_VOLUMES.get(asset, Decimal('0'))
            min_floor_ngn = max(MIN_FIAT_NGN, min_base_vol * price)
            
            dynamic_drift_threshold = min_floor_ngn / total

            is_drifted = abs_drift >= dynamic_drift_threshold
            if is_drifted:
                drift_detected = True

            drift_analysis[asset] = {
                'actual_value': val,
                'actual_weight': float(actual_weight),
                'target_weight': float(target_weight),
                'drift': float(drift),
                'dynamic_threshold': float(dynamic_drift_threshold),
                'breached': is_drifted
            }
            logger.info(f"Asset {asset}: Actual Weight = {actual_weight*100:.2f}% | Target = {target_weight*100:.2f}% | "
                        f"Drift = {drift*100:+.2f}% | Dynamic Threshold = {dynamic_drift_threshold*100:.2f}% | Breached = {is_drifted}")

        return drift_detected, drift_analysis

    def evaluate_rebalance(self, holdings_val, ngn_balance, total_portfolio_val):
        """
        Compatibility wrapper for legacy dashboard calls.
        """
        state = self.fetch_live_portfolio()
        drift_detected, analysis = self.evaluate_drift(state)
        max_drift = max([abs(d.get('drift', 0)) for d in analysis.values()]) if analysis else 0.0
        return {
            "rebalance_triggered": drift_detected,
            "max_drift_percentage": float(max_drift * 100),
            "analysis": analysis
        }

    def execute_rebalance(self):
        """
        Executes the two-phase cashless rebalancing protocol without adding external cash:
        Phase 1: Liquidate overweight assets into internal NGN proceeds.
        Phase 2: Allocate internal NGN proceeds to underweight assets.
        Strictly governed by dynamic exchange-floor compliance gates.
        """
        logger.info("Initializing ICIE Autonomous Cashless Rebalance Cycle...")
        state = self.fetch_live_portfolio()
        drift_detected, analysis = self.evaluate_drift(state)

        if not drift_detected:
            logger.info("Portfolio drift is within dynamic exchange-floor limits. No rebalancing action required.")
            return {"status": "NO_DRIFT", "analysis": analysis}

        logger.info("Dynamic drift threshold exceeded. Commencing Cashless Two-Phase Rebalance Execution...")
        
        total = state['total_portfolio']
        overweight_actions = []
        underweight_actions = []

        for asset, data in analysis.items():
            if not data['breached']:
                continue
            target_val = total * TARGET_WEIGHTS[asset]
            diff = data['actual_value'] - target_val

            if diff > 0:
                overweight_actions.append({'asset': asset, 'excess_val': diff})
            elif diff < 0:
                underweight_actions.append({'asset': asset, 'deficit_val': abs(diff)})

        # Phase 1: Liquidation Leg (Sell Overweight into internal NGN proceeds)
        liquidated_ngn = Decimal('0')
        for item in overweight_actions:
            asset = item['asset']
            excess = item['excess_val']
            price = state['spot_prices'][asset]
            
            sell_qty = (excess / price).quantize(Decimal('0.0001'), rounding=ROUND_DOWN)
            order_val = sell_qty * price
            min_base_vol = MIN_BASE_VOLUMES.get(asset, Decimal('0'))

            # Compliance Gate: Check Fiat Floor and Hard Base Volume Floor
            if order_val < MIN_FIAT_NGN or sell_qty < min_base_vol:
                logger.info(f"[COMPLIANCE_GATE] Sell order for {asset} (Qty: {sell_qty}, Val: ₦{order_val:,.2f}) "
                            f"is below Luno minimums. Skipping sell leg.")
                continue

            pair = f"{asset}NGN"
            logger.info(f"[Phase 1] Liquidating Overweight {asset}: Selling qty {sell_qty} (~₦{order_val:,.2f})")
            
            if not self.dry_run:
                res = self.connector.post_order(pair, "SELL", sell_qty, price)
                logger.info(f"Sell order response: {res}")
            
            liquidated_ngn += order_val

        if liquidated_ngn <= 0:
            logger.info("[SETTLEMENT_GATE] No valid liquidations met Luno minimum volume and fiat floors. Halting cashless cycle.")
            return {"status": "SKIPPED_MIN_VOLUME", "liquidated_ngn": 0.0}

        # Phase 2: Reallocation Leg (Buy Underweight using strictly internal liquidated NGN proceeds)
        total_deficit = sum([item['deficit_val'] for item in underweight_actions])
        if total_deficit <= 0:
            logger.info("No underweight assets to fund.")
            return {"status": "SUCCESS_PHASE1_ONLY", "liquidated_ngn": float(liquidated_ngn)}

        for item in underweight_actions:
            asset = item['asset']
            deficit = item['deficit_val']
            
            allocation_share = deficit / total_deficit
            allocated_ngn = liquidated_ngn * allocation_share
            
            if allocated_ngn < MIN_FIAT_NGN:
                logger.info(f"[COMPLIANCE_GATE] Allocated order value ₦{allocated_ngn:,.2f} for {asset} is below ₦50. Skipping buy leg.")
                continue

            price = state['spot_prices'][asset]
            buy_qty = (allocated_ngn / price).quantize(Decimal('0.0001'), rounding=ROUND_DOWN)
            order_val = buy_qty * price
            min_base_vol = MIN_BASE_VOLUMES.get(asset, Decimal('0'))

            if order_val < MIN_FIAT_NGN or buy_qty < min_base_vol:
                logger.info(f"[COMPLIANCE_GATE] Buy order for {asset} (Qty: {buy_qty}, Val: ₦{order_val:,.2f}) "
                            f"is below Luno minimums. Skipping.")
                continue

            pair = f"{asset}NGN"
            logger.info(f"[Phase 2] Reallocating to Underweight {asset}: Buying qty {buy_qty} using internal ₦{order_val:,.2f}")

            if not self.dry_run:
                res = self.connector.post_order(pair, "BUY", buy_qty, price)
                logger.info(f"Buy order response: {res}")

        logger.info("ICIE Autonomous Cashless Rebalance Cycle Completed Successfully.")
        return {"status": "SUCCESS", "liquidated_ngn": float(liquidated_ngn)}

# Export both class names to ensure 100% import compatibility across all modules
AutonomousRebalancer = ICIERebalancer
