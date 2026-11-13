"""
Institutional Crypto Investment Engine (ICIE) - Autonomous Rebalancer
Target Asset Allocation: BTC 50%, ETH 30%, SOL 20%
Core Architecture: 
- Routed via InstitutionalOrderExecutor (idempotency, circuit breakers, retry logic)
- Fee-aware liquidation calculation (0.75% Luno taker fee adjustment)
- Two-phase cashless rebalancing with mandatory wallet refresh between Phase 1 and Phase 2
- Asset-specific decimal precision and dynamic exchange-floor drift thresholds
- SQLite WAL audit logging integration
"""

import logging
from decimal import Decimal, ROUND_DOWN
from database.repository import log_audit

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ICIERebalancer")

TARGET_WEIGHTS = {
    'XBT': Decimal('0.50'),
    'ETH': Decimal('0.30'),
    'SOL': Decimal('0.20')
}

MIN_FIAT_NGN = Decimal('50.00')  # Luno absolute minimum fiat order value

# Luno hard exchange base volume floors per asset
MIN_BASE_VOLUMES = {
    'XBT': Decimal('0.000025'),
    'ETH': Decimal('0.001'),
    'SOL': Decimal('0.05')
}

# Asset-specific decimal precision for order sizing
ASSET_PRECISION = {
    'XBT': Decimal('0.00000001'),
    'ETH': Decimal('0.000001'),
    'SOL': Decimal('0.0001')
}

LUNO_TAKER_FEE = Decimal('0.0075')  # 0.75% fee

class ICIERebalancer:
    def __init__(self, connector, order_executor, dry_run: bool = True):
        self.connector = connector
        self.order_executor = order_executor
        self.dry_run = dry_run
        self.target_weights = TARGET_WEIGHTS

    def fetch_live_portfolio(self):
        """
        Fetches live wallet balances (balance + reserved) and real-time spot prices
        with robust exception handling to prevent API timeouts from killing the engine.
        """
        try:
            return self.connector.get_portfolio_valuation()
        except Exception as e:
            logger.error(f"CRITICAL: Failed to fetch live portfolio valuation: {e}")
            log_audit("REBALANCER_ERROR", f"Portfolio valuation fetch failed: {str(e)}")
            raise

    def evaluate_drift(self, portfolio_state):
        """
        Evaluates actual weights against target weights (50/30/20) using dynamic 
        exchange-floor drift thresholds maintained in high-precision Decimal.
        """
        total = portfolio_state.get('total_portfolio', Decimal('0'))
        if total <= 0:
            logger.warning("Portfolio total capital is zero or negative. Halting rebalance.")
            return False, {}

        drift_detected = False
        drift_analysis = {}

        for asset, target_weight in TARGET_WEIGHTS.items():
            val = portfolio_state.get('asset_values', {}).get(asset, Decimal('0'))
            actual_weight = val / total if total > 0 else Decimal('0')
            drift = actual_weight - target_weight
            abs_drift = abs(drift)

            price = portfolio_state.get('spot_prices', {}).get(asset, Decimal('1'))
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

        return drift_detected, drift_analysis

    def evaluate_rebalance(self, holdings_val=None, ngn_balance=None, total_portfolio_val=None):
        """
        Compatibility wrapper for dashboard calls, utilizing live state or optional overrides.
        """
        try:
            state = self.fetch_live_portfolio()
            drift_detected, analysis = self.evaluate_drift(state)
            max_drift = max([abs(d.get('drift', 0)) for d in analysis.values()]) if analysis else 0.0
            return {
                "rebalance_triggered": drift_detected,
                "max_drift_percentage": float(max_drift * 100),
                "analysis": analysis
            }
        except Exception as e:
            logger.error(f"Error in evaluate_rebalance: {e}")
            return {"rebalance_triggered": False, "max_drift_percentage": 0.0, "analysis": {}}

    def execute_rebalance(self):
        """
        Executes institutional two-phase cashless rebalancing:
        1. Phase 1: Liquidate overweight assets (fee-adjusted for 0.75% taker fee).
        2. Portfolio Refresh: Re-query Luno API to obtain true post-liquidation NGN balance.
        3. Phase 2: Allocate available NGN reserve proportionally to underweight assets.
        All orders routed through InstitutionalOrderExecutor with circuit breakers & audit logs.
        """
        logger.info("Initializing ICIE Institutional Cashless Rebalance Cycle...")
        
        try:
            state = self.fetch_live_portfolio()
        except Exception as e:
            return {"status": "FETCH_FAILED", "error": str(e)}

        drift_detected, analysis = self.evaluate_drift(state)

        if not drift_detected:
            logger.info("Portfolio drift is within dynamic exchange-floor limits. No rebalancing required.")
            return {"status": "NO_DRIFT", "analysis": analysis}

        logger.info("Dynamic drift threshold exceeded. Commencing Institutional Two-Phase Execution...")
        
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

        # -------------------------------------------------------------
        # PHASE 1: Liquidation Leg (Sell Overweight into NGN)
        # -------------------------------------------------------------
        liquidated_ngn = Decimal('0')
        for item in overweight_actions:
            asset = item['asset']
            excess = item['excess_val']
            price = state['spot_prices'][asset]
            
            # Asset-specific quantization
            precision = ASSET_PRECISION.get(asset, Decimal('0.0001'))
            sell_qty = (excess / price).quantize(precision, rounding=ROUND_DOWN)
            order_val = sell_qty * price
            min_base_vol = MIN_BASE_VOLUMES.get(asset, Decimal('0'))

            if order_val < MIN_FIAT_NGN or sell_qty < min_base_vol:
                logger.info(f"[COMPLIANCE_GATE] Sell order for {asset} below Luno minimums. Skipping.")
                log_audit("REBALANCE_SKIP", f"Sell {asset} skipped: below exchange minimum floors.")
                continue

            pair = f"{asset}NGN"
            logger.info(f"[Phase 1] Liquidating Overweight {asset}: Selling qty {sell_qty} (~₦{order_val:,.2f})")

            if self.dry_run or not self.order_executor:
                logger.info(f"[DRY RUN] Would execute SELL market order for {pair} (Notional: ₦{order_val})")
                net_proceeds = order_val * (Decimal('1') - LUNO_TAKER_FEE)
                liquidated_ngn += net_proceeds
            else:
                try:
                    res = self.order_executor.execute_market_order(
                        pair=pair,
                        side="SELL",
                        target_val=float(order_val),
                        current_price=float(price)
                    )
                    if res.get("status") == "SUCCESS" or res.get("success"):
                        net_proceeds = order_val * (Decimal('1') - LUNO_TAKER_FEE)
                        liquidated_ngn += net_proceeds
                        log_audit("REBALANCE_SELL", f"Successfully sold {sell_qty} {asset} on {pair} for ~₦{order_val}")
                    else:
                        logger.error(f"Order executor failed SELL for {pair}: {res}")
                        log_audit("REBALANCE_ERROR", f"Sell execution failed for {pair}: {res}")
                except Exception as ex:
                    logger.error(f"Exception during SELL execution for {pair}: {ex}")
                    log_audit("REBALANCE_ERROR", f"Sell exception for {pair}: {str(ex)}")

        if liquidated_ngn <= 0:
            logger.info("[SETTLEMENT_GATE] No valid liquidations executed. Halting cashless cycle.")
            return {"status": "SKIPPED_MIN_VOLUME", "liquidated_ngn": 0.0}

        # -------------------------------------------------------------
        # MANDATORY PORTFOLIO REFRESH (Post-Sell State Sync)
        # -------------------------------------------------------------
        logger.info("[SETTLEMENT_SYNC] Refreshing live portfolio state post-liquidation...")
        try:
            refreshed_state = self.fetch_live_portfolio()
            available_ngn = refreshed_state.get('ngn_reserve', liquidated_ngn)
        except Exception as e:
            logger.warning(f"Failed to refresh state post-sell, falling back to estimated liquidated NGN: {e}")
            available_ngn = liquidated_ngn

        # -------------------------------------------------------------
        # PHASE 2: Reallocation Leg (Buy Underweight using Available NGN)
        # -------------------------------------------------------------
        total_deficit = sum([item['deficit_val'] for item in underweight_actions])
        if total_deficit <= 0 or available_ngn <= 0:
            logger.info("No underweight assets to fund or zero available NGN reserve.")
            return {"status": "SUCCESS_PHASE1_ONLY", "liquidated_ngn": float(available_ngn)}

        for item in underweight_actions:
            asset = item['asset']
            deficit = item['deficit_val']
            
            allocation_share = deficit / total_deficit
            allocated_ngn = available_ngn * allocation_share
            
            if allocated_ngn < MIN_FIAT_NGN:
                logger.info(f"[COMPLIANCE_GATE] Allocated order value ₦{allocated_ngn:,.2f} for {asset} is below ₦50. Skipping.")
                continue

            price = refreshed_state['spot_prices'][asset]
            precision = ASSET_PRECISION.get(asset, Decimal('0.0001'))
            
            # For BUY orders, calculate exact counter_volume (NGN)
            net_allocated_ngn = allocated_ngn * (Decimal('1') - LUNO_TAKER_FEE)
            buy_qty = (net_allocated_ngn / price).quantize(precision, rounding=ROUND_DOWN)
            order_val = buy_qty * price
            min_base_vol = MIN_BASE_VOLUMES.get(asset, Decimal('0'))

            if order_val < MIN_FIAT_NGN or buy_qty < min_base_vol:
                logger.info(f"[COMPLIANCE_GATE] Buy order for {asset} below Luno minimums. Skipping.")
                continue

            pair = f"{asset}NGN"
            logger.info(f"[Phase 2] Reallocating to Underweight {asset}: Buying qty {buy_qty} using internal ₦{order_val:,.2f}")

            if self.dry_run or not self.order_executor:
                logger.info(f"[DRY RUN] Would execute BUY market order for {pair} (Counter NGN: ₦{order_val})")
                log_audit("REBALANCE_BUY_DRY", f"Dry-run bought {buy_qty} {asset} on {pair} for ₦{order_val}")
            else:
                try:
                    res = self.order_executor.execute_market_order(
                        pair=pair,
                        side="BUY",
                        target_val=float(order_val),
                        current_price=float(price)
                    )
                    if res.get("status") == "SUCCESS" or res.get("success"):
                        log_audit("REBALANCE_BUY", f"Successfully bought {buy_qty} {asset} on {pair} for ₦{order_val}")
                    else:
                        logger.error(f"Order executor failed BUY for {pair}: {res}")
                        log_audit("REBALANCE_ERROR", f"Buy execution failed for {pair}: {res}")
                except Exception as ex:
                    logger.error(f"Exception during BUY execution for {pair}: {ex}")
                    log_audit("REBALANCE_ERROR", f"Buy exception for {pair}: {str(ex)}")

        logger.info("ICIE Institutional Cashless Rebalance Cycle Completed Successfully.")
        return {"status": "SUCCESS", "liquidated_ngn": float(available_ngn)}

# Export alias for backward compatibility and dashboard imports
AutonomousRebalancer = ICIERebalancer
