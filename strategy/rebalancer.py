"""
Institutional Crypto Investment Engine (ICIE) - Autonomous Rebalancer
Target Asset Allocation: BTC 50%, ETH 30%, SOL 20%
Core Architecture: 
- Routed via InstitutionalOrderExecutor (idempotency, circuit breakers, retry logic)
- Fee-aware liquidation calculation (0.75% Luno taker fee adjustment)
- Two-phase cashless rebalancing with mandatory wallet refresh between Phase 1 and Phase 2
- Asset-specific decimal precision increments (BTC 8 dec, ETH 6 dec, SOL 2 dec)
- Execution status verification and final post-rebalance state audit
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

MIN_FIAT_NGN = Decimal('50.00')

MIN_BASE_VOLUMES = {
    'XBT': Decimal('0.000025'),
    'ETH': Decimal('0.001'),
    'SOL': Decimal('0.05')
}

PAIR_PRECISION = {
    'XBT': Decimal('0.00000001'),
    'ETH': Decimal('0.000001'),
    'SOL': Decimal('0.01')
}

LUNO_TAKER_FEE = Decimal('0.0075')

class ICIERebalancer:
    """
    Autonomous Institutional Rebalancer implementing two-phase cashless rebalancing,
    dynamic exchange-floor drift thresholds, and rigorous audit compliance.
    """
    def __init__(self, connector, order_executor, dry_run: bool = True):
        self.connector = connector
        self.order_executor = order_executor
        self.dry_run = dry_run
        self.target_weights = TARGET_WEIGHTS

    def fetch_live_portfolio(self):
        try:
            return self.connector.get_portfolio_valuation()
        except Exception as e:
            logger.error(f"CRITICAL: Failed to fetch live portfolio valuation: {e}")
            log_audit(
                venue="ICIE",
                asset="SYSTEM",
                price=0.0,
                amount=0.0,
                units="N/A",
                status=f"Portfolio valuation fetch failed: {str(e)}"
            )
            raise

    def evaluate_drift(self, portfolio_state):
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

        executed_sells = 0
        for item in overweight_actions:
            asset = item['asset']
            excess = item['excess_val']
            price = state['spot_prices'][asset]
            
            precision = PAIR_PRECISION.get(asset, Decimal('0.0001'))
            sell_qty = (excess / price).quantize(precision, rounding=ROUND_DOWN)
            order_val = sell_qty * price
            min_base_vol = MIN_BASE_VOLUMES.get(asset, Decimal('0'))

            if order_val < MIN_FIAT_NGN or sell_qty < min_base_vol:
                logger.info(f"[COMPLIANCE_GATE] Sell order for {asset} below Luno minimums. Skipping.")
                log_audit(
                    venue="ICIE",
                    asset=asset,
                    price=float(price),
                    amount=float(order_val),
                    units="SKIPPED",
                    status=f"Sell {asset} skipped: below exchange minimum floors."
                )
                continue

            pair = f"{asset}NGN"
            logger.info(f"[Phase 1] Liquidating Overweight {asset}: Selling qty {sell_qty} (~₦{order_val:,.2f})")

            if self.dry_run or not self.order_executor:
                logger.info(f"[DRY RUN] Would execute SELL market order for {pair} (Notional: ₦{order_val})")
                executed_sells += 1
                log_audit(
                    venue="ICIE",
                    asset=pair,
                    price=float(price),
                    amount=float(order_val),
                    units=str(sell_qty),
                    status="DRY_RUN_SELL"
                )
            else:
                try:
                    res = self.order_executor.execute_market_order(
                        pair=pair,
                        side="SELL",
                        target_val=float(order_val),
                        current_price=float(price)
                    )
                    if res.get("status") in ["SUCCESS", "PENDING"] or res.get("success") or res.get("order_id"):
                        executed_sells += 1
                        log_audit(
                            venue="LUNO",
                            asset=pair,
                            price=float(price),
                            amount=float(order_val),
                            units=str(sell_qty),
                            status="SUCCESS"
                        )
                        logger.info(f"Order fill confirmed for {pair} SELL: {res.get('order_id', 'Simulated')}")
                    else:
                        logger.error(f"Order executor failed SELL fill confirmation for {pair}: {res}")
                        log_audit(
                            venue="LUNO",
                            asset=pair,
                            price=float(price),
                            amount=float(order_val),
                            units=str(sell_qty),
                            status=f"SELL_FAILED: {res}"
                        )
                except Exception as ex:
                    logger.error(f"Exception during SELL execution for {pair}: {ex}")
                    log_audit(
                        venue="LUNO",
                        asset=pair,
                        price=float(price),
                        amount=float(order_val),
                        units=str(sell_qty),
                        status=f"SELL_EXCEPTION: {str(ex)}"
                    )

        if executed_sells <= 0 and not self.dry_run:
            logger.info("[SETTLEMENT_GATE] No valid sell orders successfully filled. Halting cashless cycle.")
            return {"status": "SKIPPED_EXECUTION_FAILURE", "executed_sells": 0}

        logger.info("[SETTLEMENT_SYNC] Refreshing live wallet balances post-liquidation...")
        try:
            refreshed_state = self.fetch_live_portfolio()
            available_ngn = refreshed_state.get('ngn_reserve', Decimal('0'))
            logger.info(f"[SETTLEMENT_SYNC] Verified available NGN reserve post-liquidation: ₦{available_ngn:,.2f}")
            log_audit(
                venue="ICIE",
                asset="SYSTEM",
                price=0.0,
                amount=float(available_ngn),
                units="NGN",
                status="WALLET_REFRESH_SUCCESS"
            )
        except Exception as e:
            logger.error(f"CRITICAL: Failed to refresh wallet state post-sell: {e}")
            log_audit(
                venue="ICIE",
                asset="SYSTEM",
                price=0.0,
                amount=0.0,
                units="N/A",
                status=f"WALLET_REFRESH_FAILED: {str(e)}"
            )
            return {"status": "POST_SELL_REFRESH_FAILED", "error": str(e)}

        total_deficit = sum([item['deficit_val'] for item in underweight_actions])
        if total_deficit <= 0 or available_ngn <= 0:
            logger.info("No underweight assets to fund or zero verified NGN reserve available.")
            return {"status": "SUCCESS_PHASE1_ONLY", "available_ngn": float(available_ngn)}

        executed_buys = 0
        for item in underweight_actions:
            asset = item['asset']
            deficit = item['deficit_val']
            
            allocation_share = deficit / total_deficit
            allocated_ngn = available_ngn * allocation_share
            
            if allocated_ngn < MIN_FIAT_NGN:
                logger.info(f"[COMPLIANCE_GATE] Allocated order value ₦{allocated_ngn:,.2f} for {asset} is below ₦50. Skipping.")
                continue

            price = refreshed_state['spot_prices'][asset]
            precision = PAIR_PRECISION.get(asset, Decimal('0.0001'))
            
            net_allocated_ngn = allocated_ngn * (Decimal('1') - LUNO_TAKER_FEE)
            buy_qty = (net_allocated_ngn / price).quantize(precision, rounding=ROUND_DOWN)
            order_val = buy_qty * price
            min_base_vol = MIN_BASE_VOLUMES.get(asset, Decimal('0'))

            if order_val < MIN_FIAT_NGN or buy_qty < min_base_vol:
                logger.info(f"[COMPLIANCE_GATE] Buy order for {asset} below Luno minimums. Skipping.")
                continue

            pair = f"{asset}NGN"
            logger.info(f"[Phase 2] Reallocating to Underweight {asset}: Buying qty {buy_qty} using verified ₦{order_val:,.2f}")

            if self.dry_run or not self.order_executor:
                logger.info(f"[DRY RUN] Would execute BUY market order for {pair} (Counter NGN: ₦{order_val})")
                executed_buys += 1
                log_audit(
                    venue="ICIE",
                    asset=pair,
                    price=float(price),
                    amount=float(order_val),
                    units=str(buy_qty),
                    status="DRY_RUN_BUY"
                )
            else:
                try:
                    res = self.order_executor.execute_market_order(
                        pair=pair,
                        side="BUY",
                        target_val=float(order_val),
                        current_price=float(price)
                    )
                    if res.get("status") in ["SUCCESS", "PENDING"] or res.get("success") or res.get("order_id"):
                        executed_buys += 1
                        log_audit(
                            venue="LUNO",
                            asset=pair,
                            price=float(price),
                            amount=float(order_val),
                            units=str(buy_qty),
                            status="SUCCESS"
                        )
                        logger.info(f"Order fill confirmed for {pair} BUY: {res.get('order_id', 'Simulated')}")
                    else:
                        logger.error(f"Order executor failed BUY fill confirmation for {pair}: {res}")
                        log_audit(
                            venue="LUNO",
                            asset=pair,
                            price=float(price),
                            amount=float(order_val),
                            units=str(buy_qty),
                            status=f"BUY_FAILED: {res}"
                        )
                except Exception as ex:
                    logger.error(f"Exception during BUY execution for {pair}: {ex}")
                    log_audit(
                        venue="LUNO",
                        asset=pair,
                        price=float(price),
                        amount=float(order_val),
                        units=str(buy_qty),
                        status=f"BUY_EXCEPTION: {str(ex)}"
                    )

        logger.info("[FINAL_AUDIT] Performing post-rebalance drift verification check...")
        try:
            final_state = self.fetch_live_portfolio()
            _, final_analysis = self.evaluate_drift(final_state)
            max_final_drift = max([abs(d.get('drift', 0)) for d in final_analysis.values()]) if final_analysis else 0.0
            logger.info(f"[FINAL_AUDIT] Rebalance cycle completed. Maximum post-rebalance asset drift: {max_final_drift*100:.2f}%")
            log_audit(
                venue="ICIE",
                asset="PORTFOLIO",
                price=0.0,
                amount=0.0,
                units="N/A",
                status=f"COMPLETE: Final max drift {max_final_drift*100:.2f}%"
            )
        except Exception as audit_err:
            logger.warning(f"Final post-rebalance audit check encountered an error: {audit_err}")

        return {
            "status": "SUCCESS", 
            "executed_sells": executed_sells, 
            "executed_buys": executed_buys,
            "final_max_drift": float(max_final_drift) if 'max_final_drift' in locals() else 0.0
        }

AutonomousRebalancer = ICIERebalancer
