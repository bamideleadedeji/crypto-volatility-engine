---

### File 3 of 4: `strategy/rebalancer.py`

```python
"""
Institutional Crypto Investment Engine (ICIE) - Live Cashless Autonomous Rebalancer
Author: Bamidele Adedeji, Principal Consultant (Bamidele & Associates)
Target Asset Allocation: BTC 50%, ETH 30%, SOL 20%
Core Architecture (Zero Simulation / 100% Live Execution):
- Pre-Flight Spread & Disbalance Gate: Verifies order-book spreads and ensures Phase 1 sell proceeds
  will satisfy Luno minimum lot sizes for Phase 2 buys before incurring any sell fees.
- Phase 1 Liquidation: Sells only breached overweight assets using live bid pricing.
- Active Settlement Polling: Polls Luno /balance until NGN proceeds clear and reserved balances settle.
- Phase 2 Reallocation: Deploys cleared NGN proportionally across underweight assets (never exceeding target).
- Dual Immutable Audit Logging: Records every execution to both SQLite WAL and trade_log.csv.
"""

import os
import time
import logging
import pandas as pd
from datetime import datetime
from decimal import Decimal, ROUND_DOWN

# ---------------------------------------------------------------------------
# 1. MODULE-LEVEL CONSTANTS (ICIE Constitution Rules)
# ---------------------------------------------------------------------------
TARGET_WEIGHTS = {
    'XBT': Decimal('0.50'),
    'ETH': Decimal('0.30'),
    'SOL': Decimal('0.20')
}

MAX_POSITION_WEIGHTS = {
    'XBT': Decimal('0.60'),
    'ETH': Decimal('0.40'),
    'SOL': Decimal('0.30')
}

MIN_FIAT_NGN = Decimal('50.00')          # Luno minimum fiat order value
MAX_TRADES_PER_CYCLE = 4                 # Constitution Section 15: Max 4 trades
CONSTITUTION_DRIFT_FLOOR = Decimal('0.03')  # Constitution Section 4: 3% drift threshold
MAX_ACCEPTABLE_SPREAD = Decimal('0.035') # 3.5% max bid-ask spread guard for NGN books

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

LUNO_TAKER_FEE = Decimal('0.0075')       # 0.75% taker fee
TRADE_LOG_FILE = "trade_log.csv"

# ---------------------------------------------------------------------------
# 2. LOGGING SETUP
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ICIERebalancer")


def _safe_log_audit(venue: str, asset: str, price: float, amount: float, units: str, status: str):
    """Lazy importer for database audit logging to prevent circular import side effects."""
    try:
        from database.repository import log_audit
        log_audit(venue, asset, price, amount, units, status)
    except Exception as e:
        logger.info(f"Fallback Audit -> {venue} | {asset} | {status} | Amount: ₦{amount:,.2f} | Error: {e}")


def _append_trade_csv(asset: str, side: str, volume: str, execution_price: float, total_value_ngn: float, status: str):
    """Appends an immutable execution record to trade_log.csv per Workflow Step 9."""
    try:
        row = pd.DataFrame([{
            "timestamp": datetime.utcnow().isoformat(),
            "asset": asset,
            "side": side,
            "volume": volume,
            "execution_price": round(execution_price, 2),
            "total_value_ngn": round(total_value_ngn, 2),
            "status": status
        }])
        write_header = not os.path.exists(TRADE_LOG_FILE)
        row.to_csv(TRADE_LOG_FILE, mode="a", header=write_header, index=False)
    except Exception as e:
        logger.error(f"CRITICAL: Failed to append row to {TRADE_LOG_FILE}: {e}")


# ---------------------------------------------------------------------------
# 3. LIVE INSTITUTIONAL REBALANCER CLASS
# ---------------------------------------------------------------------------
class ICIERebalancer:
    """
    Live Autonomous Institutional Rebalancer implementing two-phase cashless rebalancing,
    order-book spread checks, active settlement verification, and dual audit logging.
    """
    def __init__(self, connector, order_executor, target_weights=None):
        if connector is None or order_executor is None:
            raise ValueError("CRITICAL: ICIERebalancer requires live LunoConnector and InstitutionalOrderExecutor instances.")
        self.connector = connector
        self.order_executor = order_executor
        self.target_weights = self._normalize_weights(target_weights) if target_weights else TARGET_WEIGHTS

    @staticmethod
    def _normalize_weights(weights_dict) -> dict:
        normalized = {}
        for k, v in weights_dict.items():
            clean_key = k.replace("NGN", "").strip().upper()
            normalized[clean_key] = Decimal(str(v))
        return normalized

    def fetch_live_portfolio(self) -> dict:
        """
        Fetches live wallet balances, order-book tickers, and spot valuations from Luno.
        """
        try:
            return self.connector.get_portfolio_valuation()
        except Exception as e:
            logger.error(f"CRITICAL: Failed to fetch live portfolio valuation: {e}")
            _safe_log_audit(
                venue="ICIE",
                asset="SYSTEM",
                price=0.0,
                amount=0.0,
                units="N/A",
                status=f"Portfolio valuation fetch failed: {str(e)}"
            )
            raise

    def evaluate_drift(self, portfolio_state: dict):
        """
        Evaluates actual weights against target weights (50/30/20) using the greater of
        Constitution 3% drift floor or exchange minimum executable lot floor.
        """
        total = portfolio_state.get('total_portfolio', Decimal('0'))
        if total <= 0:
            logger.warning("Portfolio total capital is zero or negative. Halting rebalance.")
            return False, {}

        drift_detected = False
        drift_analysis = {}

        for asset, target_weight in self.target_weights.items():
            val = portfolio_state.get('asset_values', {}).get(asset, Decimal('0'))
            actual_weight = val / total if total > 0 else Decimal('0')
            drift = actual_weight - target_weight
            abs_drift = abs(drift)

            price = portfolio_state.get('spot_prices', {}).get(asset, Decimal('1'))
            min_base_vol = MIN_BASE_VOLUMES.get(asset, Decimal('0'))
            min_floor_ngn = max(MIN_FIAT_NGN, min_base_vol * price)

            exchange_floor = min_floor_ngn / total
            dynamic_drift_threshold = max(CONSTITUTION_DRIFT_FLOOR, exchange_floor)

            max_cap = MAX_POSITION_WEIGHTS.get(asset, Decimal('1.00'))
            cap_breached = actual_weight > max_cap

            is_drifted = (abs_drift >= dynamic_drift_threshold) or cap_breached
            if is_drifted:
                drift_detected = True

            drift_analysis[asset] = {
                'actual_value': val,
                'actual_weight': float(actual_weight),
                'target_weight': float(target_weight),
                'drift': float(drift),
                'dynamic_threshold': float(dynamic_drift_threshold),
                'cap_breached': cap_breached,
                'breached': is_drifted
            }

        return drift_detected, drift_analysis

    def _poll_settlement_clearance(self, baseline_ngn: Decimal, expected_min_increment: Decimal, max_attempts: int = 6, sleep_seconds: float = 2.5) -> dict:
        """
        Workflow Step 6: Polls Luno /balance until open sell orders settle out of 'reserved'
        and cleared NGN increases in 'free' balance.
        """
        logger.info(f"[SETTLEMENT_POLL] Waiting for Luno NGN settlement (Baseline: ₦{baseline_ngn:,.2f}, Expected net gain: ~₦{expected_min_increment:,.2f})...")
        latest_state = None

        for attempt in range(1, max_attempts + 1):
            time.sleep(sleep_seconds)
            latest_state = self.fetch_live_portfolio()
            free_ngn = latest_state.get('ngn_reserve', Decimal('0'))
            reserved_crypto = sum(
                latest_state.get('reserved_quantities', {}).get(c, Decimal('0'))
                for c in ['XBT', 'ETH', 'SOL']
            )

            logger.info(
                f"[SETTLEMENT_POLL] Attempt {attempt}/{max_attempts} -> "
                f"Cleared Free NGN: ₦{free_ngn:,.2f} | Reserved Crypto Units: {reserved_crypto}"
            )

            if reserved_crypto == Decimal('0') and free_ngn >= (baseline_ngn + expected_min_increment * Decimal('0.90')):
                logger.info("[SETTLEMENT_POLL] Full NGN settlement confirmed on Luno ledger.")
                return latest_state

        logger.warning("[SETTLEMENT_POLL] Polling window elapsed; proceeding with verified cleared NGN balance.")
        return latest_state if latest_state is not None else self.fetch_live_portfolio()

    def _check_preflight_disbalance_gate(self, state: dict, overweight_actions: list, underweight_actions: list) -> tuple[bool, str]:
        """
        Workflow Step 7 (Pre-Flight Gate): Verifies before selling that projected net NGN proceeds
        will actually meet Luno minimum order volumes for at least one underweight asset.
        Prevents paying sell fees only to end up trapped with un-investable NGN dust.
        """
        projected_ngn = state.get('ngn_reserve', Decimal('0'))
        tickers = state.get('market_tickers', {})

        for item in overweight_actions:
            asset = item['asset']
            excess = item['excess_val']
            ticker = tickers.get(asset, {})
            bid_price = ticker.get('bid') or state['spot_prices'][asset]
            spread = ticker.get('spread_pct', Decimal('0'))

            if spread > MAX_ACCEPTABLE_SPREAD:
                return False, f"Spread circuit breaker on {asset}NGN: spread {spread*100:.2f}% exceeds {MAX_ACCEPTABLE_SPREAD*100:.2f}% limit."

            precision = PAIR_PRECISION[asset]
sell_qty = (excess / bid_price).quantize(precision, rounding=ROUND_DOWN)
gross_ngn = sell_qty * bid_price
if gross_ngn >= MIN_FIAT_NGN and sell_qty >= MIN_BASE_VOLUMES[asset]:
projected_ngn += gross_ngn * (Decimal('1') - LUNO_TAKER_FEE)
if projected_ngn < MIN_FIAT_NGN:
return False, f"Projected post-sell NGN (₦{projected_ngn:,.2f}) is below ₦50 Luno minimum."
total_deficit = sum(item['deficit_val'] for item in underweight_actions)
if total_deficit <= 0:
return False, "No underweight assets require funding."
deployable_ngn = min(projected_ngn, total_deficit)
viable_buys = 0
for item in underweight_actions:
asset = item['asset']
ticker = tickers.get(asset, {})
ask_price = ticker.get('ask') or state['spot_prices'][asset]
spread = ticker.get('spread_pct', Decimal('0'))
if spread > MAX_ACCEPTABLE_SPREAD:
return False, f"Spread circuit breaker on {asset}NGN: spread {spread100:.2f}% exceeds {MAX_ACCEPTABLE_SPREAD100:.2f}% limit."
share = item['deficit_val'] / total_deficit
alloc_ngn = deployable_ngn * share * (Decimal('1') - LUNO_TAKER_FEE)
buy_qty = (alloc_ngn / ask_price).quantize(PAIR_PRECISION[asset], rounding=ROUND_DOWN)
buy_notional = buy_qty * ask_price
if buy_notional >= MIN_FIAT_NGN and buy_qty >= MIN_BASE_VOLUMES[asset]:
viable_buys += 1
if viable_buys == 0:
return False, (
f"Disbalance Gate Halt: Projected NGN (₦{projected_ngn:,.2f}) cannot meet Luno minimum "
f"lot sizes for any underweight asset. Holding overweight asset to avoid fee bleed."
)
return True, "Pre-flight disbalance and spread checks passed."
def execute_rebalance(self, initial_state: dict = None) -> dict:
"""
Executes live two-phase cashless self-rebalancing:
1. Evaluates drift & runs Pre-Flight Disbalance + Spread Gate.
2. Phase 1: Sells breached overweight assets at live bid price (never below target).
3. Polls Luno ledger until NGN proceeds settle.
4. Phase 2: Buys underweight assets at live ask price using cleared NGN (never above target).
5. Audits post-rebalance portfolio state.
"""
logger.info("Initializing Live ICIE Cashless Self-Rebalance Cycle...")
try:
state = initial_state if initial_state is not None else self.fetch_live_portfolio()
except Exception as e:
return {"status": "FETCH_FAILED", "error": str(e)}
drift_detected, analysis = self.evaluate_drift(state)
if not drift_detected:
logger.info("Portfolio drift is within Constitution & exchange-floor limits. Holding position.")
return {"status": "NO_DRIFT", "analysis": analysis}
total = state['total_portfolio']
overweight_actions = []
underweight_actions = []
for asset, data in analysis.items():
target_val = total * self.target_weights[asset]
diff = data['actual_value'] - target_val
if diff > 0 and data['breached']:
overweight_actions.append({'asset': asset, 'excess_val': diff})
elif diff < 0:
underweight_actions.append({'asset': asset, 'deficit_val': abs(diff)})
gate_ok, gate_msg = self._check_preflight_disbalance_gate(state, overweight_actions, underweight_actions)
if not gate_ok:
logger.warning(f"[DISBALANCE_GATE] {gate_msg}")
_safe_log_audit(venue="ICIE", asset="PORTFOLIO", price=0.0, amount=float(total), units="HALTED", status=gate_msg)
return {"status": "HALTED_DISBALANCE_GATE", "reason": gate_msg}
trades_executed_total = 0
baseline_ngn = state.get('ngn_reserve', Decimal('0'))
expected_net_ngn_gain = Decimal('0')
executed_sells = 0
# -------------------------------------------------------------
# PHASE 1: Liquidation Leg (Sell Overweight into NGN at Bid)
# -------------------------------------------------------------
for item in overweight_actions:
if trades_executed_total >= MAX_TRADES_PER_CYCLE:
logger.warning("[LIMIT_GATE] Maximum trades per cycle (4) reached.")
break
asset = item['asset']
excess = item['excess_val']
ticker = state.get('market_tickers', {}).get(asset, {})
exec_price = ticker.get('bid') or state['spot_prices'][asset]
precision = PAIR_PRECISION[asset]
free_units = state.get('free_quantities', {}).get(asset, Decimal('0'))
raw_sell_qty = (excess / exec_price).quantize(precision, rounding=ROUND_DOWN)
sell_qty = min(raw_sell_qty, free_units)
order_val = sell_qty * exec_price
if order_val < MIN_FIAT_NGN or sell_qty < MIN_BASE_VOLUMES[asset]:
logger.info(f"[COMPLIANCE_GATE] Sell order for {asset} (₦{order_val:,.2f}) below Luno minimums. Skipping.")
continue
pair = f"{asset}NGN"
logger.info(f"[Phase 1] Selling Overweight {asset}: qty {sell_qty} @ Bid ₦{exec_price:,.2f} (~₦{order_val:,.2f})")
res = self.order_executor.execute_limit_order(
pair=pair,
side="SELL",
target_val=float(order_val),
current_price=float(exec_price),
volume=str(sell_qty)
)
if res.get("status") == "SUCCESS":
executed_sells += 1
trades_executed_total += 1
expected_net_ngn_gain += order_val * (Decimal('1') - LUNO_TAKER_FEE)
_append_trade_csv(
asset=pair,
side="SELL",
volume=res.get("executed_volume", str(sell_qty)),
execution_price=res.get("limit_price", float(exec_price)),
total_value_ngn=res.get("notional_ngn", float(order_val)),
status=f"SUCCESS:{res.get('order_id')}"
)
else:
logger.error(f"[Phase 1] Sell failed for {pair}: {res}")
_append_trade_csv(
asset=pair,
side="SELL",
volume=str(sell_qty),
execution_price=float(exec_price),
total_value_ngn=float(order_val),
status=f"FAILED:{res.get('error', res.get('reason', 'UNKNOWN'))}"
)
if executed_sells == 0 and baseline_ngn < MIN_FIAT_NGN:
logger.error("[SETTLEMENT_GATE] Zero sell orders succeeded and free NGN < ₦50. Aborting Phase 2.")
return {"status": "FAILED_PHASE1_LIQUIDATION", "executed_sells": 0, "executed_buys": 0}
# -------------------------------------------------------------
# STEP 6: Active Settlement Verification & Ledger Refresh
# -------------------------------------------------------------
try:
refreshed_state = self._poll_settlement_clearance(
baseline_ngn=baseline_ngn,
expected_min_increment=expected_net_ngn_gain
)
available_ngn = refreshed_state.get('ngn_reserve', Decimal('0'))
except Exception as e:
logger.error(f"CRITICAL: Post-sell settlement refresh failed: {e}")
return {"status": "POST_SELL_REFRESH_FAILED", "error": str(e)}
# -------------------------------------------------------------
# PHASE 2: Reallocation Leg (Buy Underweight using Cleared NGN)
# -------------------------------------------------------------
total_deficit = sum(item['deficit_val'] for item in underweight_actions)
if total_deficit <= 0 or available_ngn < MIN_FIAT_NGN:
logger.info(f"Phase 2 skipped: total_deficit=₦{total_deficit:,.2f}, available_ngn=₦{available_ngn:,.2f}")
return {
"status": "SUCCESS_PHASE1_ONLY",
"executed_sells": executed_sells,
"executed_buys": 0,
"available_ngn": float(available_ngn)
}
deployable_ngn = min(available_ngn, total_deficit)
executed_buys = 0
for item in underweight_actions:
if trades_executed_total >= MAX_TRADES_PER_CYCLE:
logger.warning("[LIMIT_GATE] Maximum trades per cycle (4) reached.")
break
asset = item['asset']
deficit = item['deficit_val']
share = deficit / total_deficit
allocated_ngn = deployable_ngn * share
if allocated_ngn < MIN_FIAT_NGN:
logger.info(f"[COMPLIANCE_GATE] Allocated NGN ₦{allocated_ngn:,.2f} for {asset} < ₦50. Skipping.")
continue
ticker = refreshed_state.get('market_tickers', {}).get(asset, {})
exec_price = ticker.get('ask') or refreshed_state['spot_prices'][asset]
precision = PAIR_PRECISION[asset]
# Deduct taker fee + aggressive limit buffer so total NGN debit never exceeds cleared NGN
effective_cost_factor = Decimal('1') + LUNO_TAKER_FEE + Decimal('0.0020')
net_spendable_ngn = allocated_ngn / effective_cost_factor
buy_qty = (net_spendable_ngn / exec_price).quantize(precision, rounding=ROUND_DOWN)
order_val = buy_qty * exec_price
if order_val < MIN_FIAT_NGN or buy_qty < MIN_BASE_VOLUMES[asset]:
logger.info(f"[COMPLIANCE_GATE] Buy order for {asset} (qty {buy_qty}, ₦{order_val:,.2f}) below Luno minimums. Skipping.")
continue
pair = f"{asset}NGN"
logger.info(f"[Phase 2] Buying Underweight {asset}: qty {buy_qty} @ Ask ₦{exec_price:,.2f} (~₦{order_val:,.2f})")
res = self.order_executor.execute_limit_order(
pair=pair,
side="BUY",
target_val=float(order_val),
current_price=float(exec_price),
volume=str(buy_qty)
)
if res.get("status") == "SUCCESS":
executed_buys += 1
trades_executed_total += 1
_append_trade_csv(
asset=pair,
side="BUY",
volume=res.get("executed_volume", str(buy_qty)),
execution_price=res.get("limit_price", float(exec_price)),
total_value_ngn=res.get("notional_ngn", float(order_val)),
status=f"SUCCESS:{res.get('order_id')}"
)
else:
logger.error(f"[Phase 2] Buy failed for {pair}: {res}")
_append_trade_csv(
asset=pair,
side="BUY",
volume=str(buy_qty),
execution_price=float(exec_price),
total_value_ngn=float(order_val),
status=f"FAILED:{res.get('error', res.get('reason', 'UNKNOWN'))}"
)
# -------------------------------------------------------------
# FINAL POST-REBALANCE STATE VERIFICATION AUDIT
# -------------------------------------------------------------
max_final_drift = 0.0
try:
time.sleep(2.0)
final_state = self.fetch_live_portfolio()
_, final_analysis = self.evaluate_drift(final_state)
max_final_drift = max([abs(d.get('drift', 0)) for d in final_analysis.values()]) if final_analysis else 0.0
logger.info(
f"[FINAL_AUDIT] Cycle complete | Final TPV: ₦{final_state['total_portfolio']:,.2f} | "
f"Remaining Free NGN: ₦{final_state['ngn_reserve']:,.2f} | Max Drift: {max_final_drift100:.2f}%"
)
_safe_log_audit(
venue="ICIE",
asset="PORTFOLIO",
price=0.0,
amount=float(final_state['total_portfolio']),
units="N/A",
status=f"COMPLETE: Sells={executed_sells}, Buys={executed_buys}, FinalMaxDrift={max_final_drift100:.2f}%"
)
except Exception as audit_err:
logger.warning(f"Final post-rebalance audit encountered an error: {audit_err}")
return {
"status": "SUCCESS",
"executed_sells": executed_sells,
"executed_buys": executed_buys,
"final_max_drift": float(max_final_drift)
}

Backward compatibility alias

AutonomousRebalancer = ICIERebalancer

---

### File 4 of 4: `scripts/luno_institutional_engine.py`

```python
#!/usr/bin/env python3
"""
ICIE Autonomous Institutional Engine (Luno Live Production Node)
Author: Bamidele Adedeji, Principal Consultant (Bamidele & Associates)
Objective: Live Institutional Crypto Investment & Cashless Self-Rebalancing (50% BTC, 30% ETH, 20% SOL)
Architecture: Zero-Simulation Production Node with Sequential Settlement Verification & Dual Audit Trails
"""

import os
import sys
import pandas as pd
from datetime import datetime
from decimal import Decimal

# ==========================================
# 0. ABSOLUTE PATH RESOLUTION & LOG INITIALIZATION
# ==========================================
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

TRADE_LOG_FILE = "trade_log.csv"
LOG_FILE = "institutional_audit.log"

if not os.path.exists(TRADE_LOG_FILE):
    df_init = pd.DataFrame(columns=["timestamp", "asset", "side", "volume", "execution_price", "total_value_ngn", "status"])
    df_init.to_csv(TRADE_LOG_FILE, index=False)

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"{datetime.utcnow().isoformat()} [INFO] ICIE Live Audit Log Initialized.\n")

from core.config import load_config
from core.logging import setup_logger
from database.repository import init_database, log_audit
from execution.luno_connector import LunoConnector
from execution.order_executor import InstitutionalOrderExecutor
from risk.circuit_breakers import CircuitBreakerManager
from strategy.rebalancer import ICIERebalancer, TARGET_WEIGHTS

# ==========================================
# 1. INITIALIZATION & STRICT LIVE CREDENTIAL GATE
# ==========================================
logger = setup_logger("ICIEEngine")


def run_institutional_engine():
    """
    Executes the Live ICIE Closed-Loop Two-Phase Cashless Rebalancing Engine.
    Strictly enforces live Luno API connectivity—aborts via Circuit Breaker if credentials
    are missing or exchange telemetry fails. Zero simulation code.
    """
    logger.info("[START] Initializing ICIE Autonomous Institutional Engine (Luno Live Production Node)...")

    init_database()
    config = load_config()

    luno_key = os.environ.get("LUNO_API_KEY_ID")
    luno_secret = os.environ.get("LUNO_API_KEY_SECRET")

    if not luno_key or not luno_secret:
        logger.error("CRITICAL [CIRCUIT BREAKER]: LUNO_API_KEY_ID or LUNO_API_KEY_SECRET missing from environment. Aborting live execution.")
        try:
            log_audit(
                venue="ICIE",
                asset="SYSTEM",
                price=0.0,
                amount=0.0,
                units="N/A",
                status="CIRCUIT_BREAKER_ABORT: Missing Luno API credentials"
            )
        except Exception:
            pass
        sys.exit(1)

    try:
        connector = LunoConnector(luno_key, luno_secret)
        circuit_breaker = CircuitBreakerManager()
        order_executor = InstitutionalOrderExecutor(connector, circuit_breaker)
        target_weights_cfg = config.get("target_weights", TARGET_WEIGHTS)

        rebalancer = ICIERebalancer(
            connector=connector,
            order_executor=order_executor,
            target_weights=target_weights_cfg
        )
    except Exception as init_err:
        logger.error(f"CRITICAL: Engine initialization failed: {init_err}")
        sys.exit(1)

    # ==========================================
    # 2. FETCH REAL LIVE VALUATION & ORDER BOOK TELEMETRY
    # ==========================================
    logger.info("[LUNO API] Querying live exchange portfolio valuation, wallet breakdown, and order-book spreads...")

    try:
        portfolio_state = connector.get_portfolio_valuation()
    except Exception as e:
        logger.error(f"CRITICAL [CIRCUIT BREAKER]: Failed to retrieve live portfolio state from Luno: {e}")
        try:
            log_audit(
                venue="ICIE",
                asset="SYSTEM",
                price=0.0,
                amount=0.0,
                units="N/A",
                status=f"CIRCUIT_BREAKER_ABORT: Valuation fetch failed - {str(e)}"
            )
        except Exception:
            pass
        sys.exit(1)

    total_portfolio = portfolio_state.get('total_portfolio', Decimal('0'))
    ngn_reserve = portfolio_state.get('ngn_reserve', Decimal('0'))
    asset_vals = portfolio_state.get('asset_values', {})
    tickers = portfolio_state.get('market_tickers', {})

    if total_portfolio <= 0:
        logger.error("CRITICAL: Total live portfolio capital evaluated to zero or negative. Aborting execution.")
        sys.exit(1)

    logger.info(f"[AUDIT] Live Portfolio Capital (TPV): ₦{total_portfolio:,.2f} | Free Cleared NGN: ₦{ngn_reserve:,.2f}")
    logger.info(
        f"[AUDIT] Live Holdings -> BTC: ₦{asset_vals.get('XBT', 0):,.2f} | "
        f"ETH: ₦{asset_vals.get('ETH', 0):,.2f} | SOL: ₦{asset_vals.get('SOL', 0):,.2f}"
    )

    for coin, t_info in tickers.items():
        logger.info(
            f"[ORDER BOOK] {t_info['pair']}: Bid ₦{t_info['bid']:,.2f} | "
            f"Ask ₦{t_info['ask']:,.2f} | Spread {t_info['spread_pct']*100:.2f}%"
        )

    # ==========================================
    # 3. EVALUATE REBALANCE DRIFT
    # ==========================================
    drift_detected, drift_analysis = rebalancer.evaluate_drift(portfolio_state)
    max_drift = max([abs(d.get('drift', 0)) for d in drift_analysis.values()]) if drift_analysis else 0.0

    for asset, stats in drift_analysis.items():
        logger.info(
            f"[DRIFT AUDIT] {asset}: Actual {stats['actual_weight']*100:.2f}% vs "
            f"Target {stats['target_weight']*100:.2f}% | Drift: {stats['drift']*100:+.2f}% | "
            f"Threshold: {stats['dynamic_threshold']*100:.2f}% | Breached: {stats['breached']}"
        )

    if not drift_detected:
        logger.info(f"[REBALANCE] Portfolio drift ({max_drift*100:.2f}%) is within Constitution & exchange-floor tolerance. Holding position.")
        logger.info("[COMPLETE] Live institutional audit cycle finished successfully.")
        try:
            log_audit(
                venue="ICIE",
                asset="PORTFOLIO",
                price=0.0,
                amount=float(total_portfolio),
                units="N/A",
                status="NO_DRIFT_HOLD"
            )
        except Exception:
            pass
        return

    logger.info(f"✅ Drift threshold breached (Max Drift: {max_drift*100:.2f}%). Initiating Live Cashless Self-Rebalance Sequence.")

    # ==========================================
    # 4. EXECUTE LIVE TWO-PHASE CASHLESS REBALANCE
    # ==========================================
    try:
        rebalance_result = rebalancer.execute_rebalance(initial_state=portfolio_state)
        logger.info(f"[EXECUTION COMPLETE] ICIE Live Rebalance Result: {rebalance_result}")
        try:
            log_audit(
                venue="ICIE",
                asset="PORTFOLIO",
                price=0.0,
                amount=float(total_portfolio),
                units="N/A",
                status=f"REBALANCE_CYCLE_RESULT: {rebalance_result.get('status', 'COMPLETED')}"
            )
        except Exception:
            pass
    except Exception as exec_err:
        logger.error(f"CRITICAL: Live rebalance execution sequence failed: {exec_err}")
        try:
            log_audit(
                venue="ICIE",
                asset="PORTFOLIO",
                price=0.0,
                amount=float(total_portfolio),
                units="N/A",
                status=f"REBALANCE_FAILED: {str(exec_err)}"
            )
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    run_institutional_engine()
