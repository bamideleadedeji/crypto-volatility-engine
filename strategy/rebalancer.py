"""
Institutional Crypto Investment Engine (ICIE) - Smart Mid-Spread Cashless Rebalancer
Target Allocation: BTC 50%, ETH 30%, SOL 20% | Maker-First Spread Capture & Volatility Regime Gate
"""

import os
import time
import logging
import pandas as pd
from datetime import datetime
from decimal import Decimal, ROUND_DOWN
from risk.volatility_gate import VolatilityRegimeGate

TARGET_WEIGHTS = {'XBT': Decimal('0.50'), 'ETH': Decimal('0.30'), 'SOL': Decimal('0.20')}
MAX_WEIGHTS = {'XBT': Decimal('0.60'), 'ETH': Decimal('0.40'), 'SOL': Decimal('0.30')}
MIN_BASE_VOLUMES = {'XBT': Decimal('0.000025'), 'ETH': Decimal('0.001'), 'SOL': Decimal('0.05')}
PAIR_PRECISION = {'XBT': Decimal('0.00000001'), 'ETH': Decimal('0.000001'), 'SOL': Decimal('0.01')}
MIN_FIAT_NGN = Decimal('50.00')
EST_FEE_BUFFER = Decimal('0.0025')
MAX_SPREAD = Decimal('0.12')

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ICIERebalancer")


def _record_audit(venue: str, asset: str, side: str, vol: str, price: float, val: float, status: str):
    try:
        from database.repository import log_audit
        log_audit(venue, asset, price, val, vol, status)
    except Exception:
        pass
    try:
        row = pd.DataFrame([{"timestamp": datetime.utcnow().isoformat(), "asset": asset, "side": side,
                             "volume": vol, "execution_price": round(price, 2),
                             "total_value_ngn": round(val, 2), "status": status}])
        row.to_csv("trade_log.csv", mode="a", header=not os.path.exists("trade_log.csv"), index=False)
    except Exception:
        pass


class ICIERebalancer:
    def __init__(self, connector, order_executor, target_weights=None, **kwargs):
        if connector is None or order_executor is None:
            raise ValueError("ICIERebalancer requires live LunoConnector and InstitutionalOrderExecutor.")
        self.connector = connector
        self.order_executor = order_executor
        self.volatility_gate = VolatilityRegimeGate()
        self.target_weights = {
            k.replace("NGN", "").upper(): Decimal(str(v))
            for k, v in (target_weights or TARGET_WEIGHTS).items()
        }

    @staticmethod
    def _maker_price(ticker: dict, fallback: Decimal, side: str) -> Decimal:
        bid, ask = ticker.get('bid', Decimal('0')), ticker.get('ask', Decimal('0'))
        if bid > 0 and ask > bid:
            spread = ask - bid
            return bid + (spread * (Decimal('0.65') if side == "SELL" else Decimal('0.35')))
        return fallback

    def evaluate_drift(self, portfolio_state: dict):
        total = portfolio_state.get('total_portfolio', Decimal('0'))
        if total <= 0:
            return False, {}

        drift_detected, analysis = False, {}
        for coin, target_w in self.target_weights.items():
            val = portfolio_state.get('asset_values', {}).get(coin, Decimal('0'))
            actual_w = val / total
            drift = actual_w - target_w
            price = portfolio_state.get('spot_prices', {}).get(coin, Decimal('1'))
            min_ngn = max(MIN_FIAT_NGN, MIN_BASE_VOLUMES[coin] * price)
            threshold = max(Decimal('0.03'), min_ngn / total)
            breached = (abs(drift) >= threshold) or (actual_w > MAX_WEIGHTS[coin])
            if breached:
                drift_detected = True

            analysis[coin] = {
                'asset': f"{coin}NGN", 'actual_value': val, 'actual_weight': float(actual_w),
                'current_weight': float(actual_w), 'target_weight': float(target_w),
                'drift': float(drift), 'dynamic_threshold': float(threshold), 'breached': breached
            }
        return drift_detected, analysis
    def _preflight_gate(self, state: dict, sells: list, buys: list) -> tuple[bool, str]:
        reserved_total = sum(state.get('reserved_quantities', {}).values())
        if reserved_total > Decimal('0'):
            return False, "Active open order detected in reserved wallet. Waiting for existing maker order to fill."

        proj_ngn = state.get('ngn_reserve', Decimal('0'))
        tickers = state.get('market_tickers', {})
        for s in sells:
            t = tickers.get(s['coin'], {})
            if t.get('spread_pct', Decimal('0')) > MAX_SPREAD:
                return False, f"Spread too wide on {s['coin']}NGN ({t['spread_pct']*100:.2f}% > 12.0%)."
            sell_px = self._maker_price(t, state['spot_prices'][s['coin']], "SELL")
            qty = (s['val'] / sell_px).quantize(PAIR_PRECISION[s['coin']], rounding=ROUND_DOWN)
            if qty >= MIN_BASE_VOLUMES[s['coin']] and (qty * sell_px) >= MIN_FIAT_NGN:
                proj_ngn += (qty * sell_px) * (Decimal('1') - EST_FEE_BUFFER)

        total_def = sum(b['val'] for b in buys)
        if proj_ngn < MIN_FIAT_NGN or total_def <= 0:
            return False, f"Insufficient projected NGN (₦{proj_ngn:,.2f}) or zero underweight deficit."

        deployable = min(proj_ngn, total_def)
        for b in buys:
            t = tickers.get(b['coin'], {})
            if t.get('spread_pct', Decimal('0')) > MAX_SPREAD:
                return False, f"Spread too wide on {b['coin']}NGN ({t['spread_pct']*100:.2f}% > 12.0%)."
            buy_px = self._maker_price(t, state['spot_prices'][b['coin']], "BUY")
            alloc = deployable * (b['val'] / total_def) * (Decimal('1') - EST_FEE_BUFFER)
            qty = (alloc / buy_px).quantize(PAIR_PRECISION[b['coin']], rounding=ROUND_DOWN)
            if qty >= MIN_BASE_VOLUMES[b['coin']] and (qty * buy_px) >= MIN_FIAT_NGN:
                return True, "Pre-flight maker spread and disbalance checks passed."
        return False, f"Projected NGN (₦{proj_ngn:,.2f}) below Luno minimum lot sizes for underweight buys."

    def _wait_for_ngn_settlement(self, baseline_ngn: Decimal, expected_gain: Decimal) -> dict:
        latest = None
        for _ in range(6):
            time.sleep(3.0)
            latest = self.connector.get_portfolio_valuation()
            reserved = sum(latest.get('reserved_quantities', {}).get(c, Decimal('0')) for c in ['XBT', 'ETH', 'SOL'])
            if reserved == Decimal('0') and latest.get('ngn_reserve', Decimal('0')) >= baseline_ngn + expected_gain * Decimal('0.90'):
                return latest
        return latest or self.connector.get_portfolio_valuation()

    def execute_rebalance(self, initial_state: dict = None) -> dict:
        state = initial_state or self.connector.get_portfolio_valuation()
        drift_detected, analysis = self.evaluate_drift(state)
        if not drift_detected:
            return {"status": "NO_DRIFT", "analysis": analysis}

        total = state['total_portfolio']
        sells, buys = [], []
        for coin, d in analysis.items():
            diff = d['actual_value'] - (total * self.target_weights[coin])
            if diff > 0 and d['breached']:
                sells.append({'coin': coin, 'val': diff})
            elif diff < 0:
                buys.append({'coin': coin, 'val': abs(diff)})

        ok, msg = self._preflight_gate(state, sells, buys)
        if not ok:
            logger.warning(f"[DISBALANCE_GATE] {msg}")
            _record_audit("ICIE", "PORTFOLIO", "HOLD", "0", 0.0, float(total), f"HALTED: {msg}")
            return {"status": "HALTED_DISBALANCE_GATE", "reason": msg}

        baseline_ngn, expected_gain = state.get('ngn_reserve', Decimal('0')), Decimal('0')
        executed_sells, executed_buys, trades = 0, 0, 0

        for s in sells:
            if trades >= 4:
                break
            coin, pair = s['coin'], f"{s['coin']}NGN"
            ticker = state.get('market_tickers', {}).get(coin, {})
            sell_px = self._maker_price(ticker, state['spot_prices'][coin], "SELL")
            free_qty = state.get('free_quantities', {}).get(coin, Decimal('0'))
            qty = min((s['val'] / sell_px).quantize(PAIR_PRECISION[coin], rounding=ROUND_DOWN), free_qty)
            notional = qty * sell_px
            if qty < MIN_BASE_VOLUMES[coin] or notional < MIN_FIAT_NGN:
                continue

            res = self.order_executor.execute_limit_order(
                pair=pair, side="SELL", target_val=float(notional),
                current_price=float(sell_px), volume=str(qty), custom_limit_price=float(sell_px)
            )
            if res.get("status") == "SUCCESS":
                executed_sells += 1
                trades += 1
                expected_gain += notional * (Decimal('1') - EST_FEE_BUFFER)
                _record_audit("LUNO", pair, "SELL", str(qty), float(sell_px), float(notional), f"PLACED:{res.get('order_id')}")

        if executed_sells == 0 and baseline_ngn < MIN_FIAT_NGN:
            return {"status": "FAILED_PHASE1_LIQUIDATION", "executed_sells": 0, "executed_buys": 0}

        refreshed = self._wait_for_ngn_settlement(baseline_ngn, expected_gain)
        avail_ngn = refreshed.get('ngn_reserve', Decimal('0'))
        total_def = sum(b['val'] for b in buys)
        if avail_ngn < MIN_FIAT_NGN or total_def <= 0:
            return {"status": "MAKER_SELL_RESTING_AWAITING_FILL", "executed_sells": executed_sells, "executed_buys": 0}

        deployable = min(avail_ngn, total_def)
        for b in buys:
            if trades >= 4:
                break
            coin, pair = b['coin'], f"{b['coin']}NGN"

            # Phase B Volatility & Trend Gate: Hold cleared NGN cash during active crash regimes
            regime = self.volatility_gate.evaluate_asset_regime(coin)
            logger.info(f"[VOLATILITY_GATE] {regime['reason']}")
            if not regime["safe_to_buy"]:
                _record_audit("ICIE", pair, "HOLD_CASH", "0", 0.0, float(deployable), f"VOL_GATE_HOLD: {regime['reason']}")
                continue

            ticker = refreshed.get('market_tickers', {}).get(coin, {})
            buy_px = self._maker_price(ticker, refreshed['spot_prices'][coin], "BUY")
            alloc = deployable * (b['val'] / total_def)
            qty = ((alloc / (Decimal('1') + EST_FEE_BUFFER)) / buy_px).quantize(PAIR_PRECISION[coin], rounding=ROUND_DOWN)
            notional = qty * buy_px
            if qty < MIN_BASE_VOLUMES[coin] or notional < MIN_FIAT_NGN:
                continue

            res = self.order_executor.execute_limit_order(
                pair=pair, side="BUY", target_val=float(notional),
                current_price=float(buy_px), volume=str(qty), custom_limit_price=float(buy_px)
            )
            if res.get("status") == "SUCCESS":
                executed_buys += 1
                trades += 1
                _record_audit("LUNO", pair, "BUY", str(qty), float(buy_px), float(notional), f"PLACED:{res.get('order_id')}")

        return {"status": "SUCCESS", "executed_sells": executed_sells, "executed_buys": executed_buys}


AutonomousRebalancer = ICIERebalancer
