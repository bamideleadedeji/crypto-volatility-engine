"""
Institutional Crypto Investment Engine (ICIE) - Smart Mid-Spread Cashless Rebalancer
Target Allocation: BTC 50%, ETH 30%, SOL 20% | Maker-First Spread Capture, Volatility Gate & Phase C HWM Guard
"""

import os
import json
import time
import sqlite3
import logging
import pandas as pd
from datetime import datetime
from decimal import Decimal, ROUND_DOWN
from risk.volatility_gate import VolatilityRegimeGate


def evaluate_high_water_mark(current_portfolio_val: float, max_drawdown_pct: float = 0.10) -> dict:
    db_name = "luno_institutional_engine.db"
    state_file = "hwm_state.json"
    initial_invested = 95000.00
    seed_peak = 59848.50
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    current_val = float(current_portfolio_val)

    with sqlite3.connect(db_name, timeout=30) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS high_water_mark (
                id INTEGER PRIMARY KEY,
                initial_invested REAL,
                peak_value REAL,
                updated_at TEXT
            )
        ''')
        cursor.execute("SELECT initial_invested, peak_value FROM high_water_mark WHERE id = 1")
        row = cursor.fetchone()
        if not row:
            if os.path.exists(state_file):
                try:
                    with open(state_file, "r", encoding="utf-8") as f:
                        saved = json.load(f)
                        seed_peak = max(seed_peak, float(saved.get("peak_value", seed_peak)))
                except Exception:
                    pass
            cursor.execute(
                "INSERT INTO high_water_mark (id, initial_invested, peak_value, updated_at) VALUES (1, ?, ?, ?)",
                (initial_invested, seed_peak, now_str)
            )
            peak_value = seed_peak
        else:
            initial_invested = float(row)
            peak_value = float(row)

        if current_val > peak_value:
            peak_value = current_val
            cursor.execute(
                "UPDATE high_water_mark SET peak_value = ?, updated_at = ? WHERE id = 1",
                (peak_value, now_str)
            )
        conn.commit()

    try:
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump({
                "initial_invested": initial_invested,
                "peak_value": peak_value,
                "last_recorded_value": current_val,
                "updated_at": now_str
            }, f, indent=2)
    except Exception:
        pass

    active_drawdown = ((peak_value - current_val) / peak_value) if (peak_value > 0 and current_val > 0) else 0.0
    breached = active_drawdown >= max_drawdown_pct
    treasury_reserve_active = current_val >= 90000.0

    return {
        "initial_invested": initial_invested,
        "peak_value": peak_value,
        "current_value": current_val,
        "active_drawdown_pct": active_drawdown,
        "max_drawdown_pct": max_drawdown_pct,
        "drawdown_breached": breached,
        "treasury_reserve_active": treasury_reserve_active,
        "reason": (
            f"PHASE C HWM GUARD TRIGGERED: Active drawdown {active_drawdown*100:.2f}% from peak ₦{peak_value:,.2f} "
            f"(>= {max_drawdown_pct*100:.0f}% limit). Holding cleared NGN cash."
            if breached else
            f"HWM Guard Healthy: Current ₦{current_val:,.2f} vs Peak ₦{peak_value:,.2f} "
            f"(Drawdown: {active_drawdown*100:.2f}% < {max_drawdown_pct*100:.0f}% limit)."
        )
    }


TARGET_WEIGHTS = {'XBT': Decimal('0.50'), 'ETH': Decimal('0.30'), 'SOL': Decimal('0.20')}
MAX_WEIGHTS = {'XBT': Decimal('0.60'), 'ETH': Decimal('0.40'), 'SOL': Decimal('0.30')}
MIN_BASE_VOLUMES = {'XBT': Decimal('0.000025'), 'ETH': Decimal('0.001'), 'SOL': Decimal('0.05')}
PAIR_PRECISION = {'XBT': Decimal('0.00000001'), 'ETH': Decimal('0.000001'), 'SOL': Decimal('0.01')}
MIN_FIAT_NGN = Decimal('50.00')
EST_FEE_BUFFER = Decimal('0.0025')
MAX_SPREAD = Decimal('0.12')
CROSS_PAIR_SPECS = {
    'ETHXBT': {'min_base': Decimal('0.01'), 'vol_tick': Decimal('0.01'), 'price_tick': Decimal('0.000001')}
}

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
                             "volume": vol, "execution_price": round(price, 2) if price >= 10 else round(price, 6),
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

    def _effective_weights(self, total_portfolio: Decimal) -> dict:
        # Phase C: Activate 10% NGN Treasury Reserve Floor once portfolio >= 90,000 NGN
        if total_portfolio >= Decimal('90000'):
            return {coin: w * Decimal('0.90') for coin, w in self.target_weights.items()}
        return self.target_weights

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

        eff_weights = self._effective_weights(total)
        drift_detected, analysis = False, {}
        for coin, target_w in eff_weights.items():
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

    def _resolve_ethxbt_route(self, state: dict, sells: list, buys: list) -> tuple[bool, dict]:
        eth_sell = next((s for s in sells if s['coin'] == 'ETH'), None)
        xbt_buy = next((b for b in buys if b['coin'] == 'XBT'), None)
        if not eth_sell or not xbt_buy:
            return False, {}

        try:
            ethxbt = self.connector.get_pair_ticker("ETHXBT")
        except Exception as exc:
            logger.warning(f"[CROSS_ROUTE] ETHXBT ticker unavailable: {exc}")
            return False, {}

        if ethxbt['bid'] <= 0 or ethxbt['ask'] <= ethxbt['bid'] or ethxbt['spread_pct'] > MAX_SPREAD:
            return False, {}

        spec = CROSS_PAIR_SPECS['ETHXBT']
        route_ngn = min(eth_sell['val'], xbt_buy['val'])
        eth_spot_ngn = state.get('spot_prices', {}).get('ETH', Decimal('0'))
        free_eth = state.get('free_quantities', {}).get('ETH', Decimal('0'))
        total_port = state.get('total_portfolio', Decimal('0'))
        xbt_val = state.get('asset_values', {}).get('XBT', Decimal('0'))

        if eth_spot_ngn <= 0 or route_ngn < MIN_FIAT_NGN or total_port <= 0:
            return False, {}

        qty = min((route_ngn / eth_spot_ngn).quantize(spec['vol_tick'], rounding=ROUND_DOWN), free_eth)

        if qty < spec['min_base'] and free_eth >= spec['min_base']:
            min_lot_ngn = spec['min_base'] * eth_spot_ngn
            projected_xbt_weight = (xbt_val + min_lot_ngn) / total_port
            if projected_xbt_weight <= MAX_WEIGHTS['XBT']:
                qty = spec['min_base']

        if qty < spec['min_base']:
            return False, {}

        sell_px = self._maker_price(ethxbt, ethxbt['ask'], "SELL").quantize(spec['price_tick'], rounding=ROUND_DOWN)
        if sell_px <= 0:
            return False, {}

        return True, {
            'pair': 'ETHXBT',
            'qty': qty,
            'sell_px': sell_px,
            'route_ngn': qty * eth_spot_ngn,
            'spread_pct': ethxbt['spread_pct'],
        }

    def _preflight_gate(self, state: dict, sells: list, buys: list) -> tuple[bool, str]:
        reserved_total = sum(state.get('reserved_quantities', {}).values())
        if reserved_total > Decimal('0'):
            return False, "Active open order detected in reserved wallet. Waiting for existing maker order to fill."

        proj_ngn = state.get('ngn_reserve', Decimal('0'))
        tickers = state.get('market_tickers', {})
        for s in sells:
            t = tickers.get(s['coin'], {})
            if t.get('spread_pct', Decimal('0')) > MAX_SPREAD:
                if s['coin'] == 'ETH':
                    can_cross, route = self._resolve_ethxbt_route(state, sells, buys)
                    if can_cross:
                        return True, (
                            f"ETHNGN spread wide ({t['spread_pct']*100:.2f}%), "
                            f"routing {route['qty']} ETH -> XBT via ETHXBT (spread {route['spread_pct']*100:.2f}%)."
                        )
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
        total = state['total_portfolio']

        # Phase C: Evaluate High-Water Mark Drawdown Guard & Treasury Reserve Scaling
        hwm_status = evaluate_high_water_mark(float(total), max_drawdown_pct=0.10)
        logger.info(f"[PHASE_C_HWM] {hwm_status['reason']}")

        drift_detected, analysis = self.evaluate_drift(state)
        if not drift_detected:
            return {"status": "NO_DRIFT", "analysis": analysis, "hwm_status": hwm_status}

        eff_weights = self._effective_weights(total)
        sells, buys = [], []
        for coin, d in analysis.items():
            diff = d['actual_value'] - (total * eff_weights[coin])
            if diff > 0 and d['breached']:
                sells.append({'coin': coin, 'val': diff})
            elif diff < 0:
                buys.append({'coin': coin, 'val': abs(diff)})

        ok, msg = self._preflight_gate(state, sells, buys)
        if not ok:
            logger.warning(f"[DISBALANCE_GATE] {msg}")
            _record_audit("ICIE", "PORTFOLIO", "HOLD", "0", 0.0, float(total), f"HALTED: {msg}")
            return {"status": "HALTED_DISBALANCE_GATE", "reason": msg, "hwm_status": hwm_status}

        eth_ticker = state.get('market_tickers', {}).get('ETH', {})
        if eth_ticker.get('spread_pct', Decimal('0')) > MAX_SPREAD:
            can_cross, route = self._resolve_ethxbt_route(state, sells, buys)
            if can_cross:
                if hwm_status["drawdown_breached"]:
                    _record_audit("ICIE", "ETHXBT", "HOLD_CASH", "0", 0.0, float(route['route_ngn']), hwm_status["reason"])
                    return {"status": "HALTED_HWM_DRAWDOWN", "reason": hwm_status["reason"], "hwm_status": hwm_status}

                regime = self.volatility_gate.evaluate_asset_regime('XBT')
                logger.info(f"[VOLATILITY_GATE] {regime['reason']}")
                if not regime["safe_to_buy"]:
                    _record_audit("ICIE", "ETHXBT", "HOLD_CASH", "0", 0.0, float(route['route_ngn']), f"VOL_GATE_HOLD: {regime['reason']}")
                    return {"status": "HALTED_VOLATILITY_GATE", "reason": regime["reason"]}

                res = self.order_executor.execute_limit_order(
                    pair=route['pair'],
                    side="SELL",
                    target_val=float(route['route_ngn']),
                    current_price=float(route['sell_px']),
                    volume=str(route['qty']),
                    custom_limit_price=float(route['sell_px']),
                )
                if res.get("status") == "SUCCESS":
                    _record_audit("LUNO", route['pair'], "SELL", str(route['qty']), float(route['sell_px']), float(route['route_ngn']), f"PLACED:{res.get('order_id')}")
                    return {"status": "SUCCESS", "executed_sells": 1, "executed_buys": 1, "route": "ETHXBT"}
                return {"status": "FAILED_CROSS_ROUTE", "reason": res.get("error") or res.get("reason")}

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

        # Phase C: Halt Phase 2 crypto purchases if High-Water Mark drawdown >= 10%
        if hwm_status["drawdown_breached"]:
            logger.warning(f"[PHASE_C_HWM_HOLD] {hwm_status['reason']}")
            _record_audit("ICIE", "NGN_TREASURY", "HOLD_CASH", "0", 0.0, float(total), hwm_status["reason"])
            return {
                "status": "PHASE1_COMPLETED_HWM_CASH_HOLD",
                "executed_sells": executed_sells,
                "executed_buys": 0,
                "reason": hwm_status["reason"],
                "hwm_status": hwm_status
            }

        refreshed = self._wait_for_ngn_settlement(baseline_ngn, expected_gain)
        avail_ngn = refreshed.get('ngn_reserve', Decimal('0'))

        # Phase C: Hold back 10% NGN Treasury Reserve once portfolio >= 90,000 NGN
        min_ngn_floor = (total * Decimal('0.10')) if hwm_status["treasury_reserve_active"] else MIN_FIAT_NGN
        deployable_ngn = max(Decimal('0'), avail_ngn - min_ngn_floor)

        total_def = sum(b['val'] for b in buys)
        if deployable_ngn < MIN_FIAT_NGN or total_def <= 0:
            return {"status": "MAKER_SELL_RESTING_OR_TREASURY_HELD", "executed_sells": executed_sells, "executed_buys": 0}

        deployable = min(deployable_ngn, total_def)
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
                            res = self.order_executor.execute_limit_order(
                pair=pair, side="BUY", target_val=float(notional),
                current_price=float(buy_px), volume=str(qty), custom_limit_price=float(buy_px)
            )
            if res.get("status") == "SUCCESS":
                executed_buys += 1
                trades += 1
                _record_audit("LUNO", pair, "BUY", str(qty), float(buy_px), float(notional), f"PLACED:{res.get('order_id')}")

        return {"status": "SUCCESS", "executed_sells": executed_sells, "executed_buys": executed_buys, "hwm_status": hwm_status}


AutonomousRebalancer = ICIERebalancer

           
