"""
Institutional Crypto Investment Engine (ICIE) - Institutional Order Executor
Author: Bamidele Adedeji, Principal Consultant (Bamidele & Associates)
Executes live trades on Luno with integrated circuit breaker validation,
idempotency hash checks, exact Luno minimum base volumes & decimal precisions,
aggressive limit-pricing buffer, and strict response verification.
"""

import logging
import hashlib
from datetime import datetime
from decimal import Decimal, ROUND_DOWN
from execution.luno_connector import LunoConnector
from risk.circuit_breakers import CircuitBreakerManager
from database.repository import check_idempotency, record_order, log_audit

logger = logging.getLogger("InstitutionalOrderExecutor")

# Verified Luno Exchange Compliance Specifications
LUNO_MIN_BASE_VOLUMES = {
    "XBTNGN": Decimal("0.000025"),
    "BTCNGN": Decimal("0.000025"),
    "ETHNGN": Decimal("0.001"),
    "SOLNGN": Decimal("0.05")
}

LUNO_VOLUME_PRECISION = {
    "XBTNGN": Decimal("0.00000001"),  # 8 decimals
    "BTCNGN": Decimal("0.00000001"),
    "ETHNGN": Decimal("0.000001"),    # 6 decimals
    "SOLNGN": Decimal("0.01")         # 2 decimals
}

LUNO_PRICE_PRECISION = {
    "XBTNGN": Decimal("1"),             # Whole Naira
    "BTCNGN": Decimal("1"),
    "ETHNGN": Decimal("1"),             # Whole Naira
    "SOLNGN": Decimal("0.01")           # 2 decimals
}

MIN_ORDER_NGN = Decimal("50.00")


class InstitutionalOrderExecutor:
    """
    Executes live orders on Luno with integrated circuit breaker validation,
    idempotency hash checks, mandatory SQLite WAL audit logging, and
    aggressive limit-pricing buffer to cross the spread cleanly.
    """

    def __init__(self, connector: LunoConnector, circuit_breaker: CircuitBreakerManager):
        if connector is None:
            raise ValueError("CRITICAL: InstitutionalOrderExecutor requires a live LunoConnector.")
        self.connector = connector
        self.circuit_breaker = circuit_breaker
        self.daily_trades_count = 0
        self.aggressive_limit_buffer = Decimal("0.0015")  # 0.15% aggressive price cross

    def execute_limit_order(
        self,
        pair: str,
        side: str,
        target_val: float = 0.0,
        current_price: float = 0.0,
        daily_pnl_pct: float = 0.0,
        volume: str = None,
        counter_volume: str = None,
        amount: float = None,
        quantity: float = None,
        **kwargs
    ) -> dict:
        """
        Validates circuit breakers, checks idempotency hash, enforces Luno exchange minimums,
        applies aggressive limit pricing, dispatches via LunoConnector, and verifies Luno acceptance.
        """
        clean_pair = pair.strip().upper()
        clean_side = side.strip().upper()

        resolved_val = Decimal(str(target_val or amount or 0))
        price_dec = Decimal(str(current_price or 0))

        if resolved_val <= 0:
            if volume and price_dec > 0:
                resolved_val = Decimal(str(volume)) * price_dec
            elif counter_volume:
                resolved_val = Decimal(str(counter_volume))
            elif quantity and price_dec > 0:
                resolved_val = Decimal(str(quantity)) * price_dec

        estimated_val = float(resolved_val)
        notional_ngn = Decimal("0")
        crypto_units = Decimal("0")
        limit_price = Decimal("0")
        order_hash = "uninitialized"

        # 1. Enforce ICIE Constitution Rule 15: Minimum Order of NGN 50
        if resolved_val < MIN_ORDER_NGN:
            error_msg = f"Order value (NGN {estimated_val:,.2f}) is below strict minimum threshold of NGN {MIN_ORDER_NGN}."
            logger.warning(error_msg)
            log_audit(
                venue="LUNO",
                asset=clean_pair,
                price=float(price_dec),
                amount=estimated_val,
                units="SKIPPED",
                status="BELOW_MIN_LIMIT"
            )
            return {"status": "SKIPPED", "reason": error_msg}

        # 2. Generate Unique Order Hash for Idempotency Protection
        raw_hash_string = f"{clean_pair}-{clean_side}-{resolved_val:.2f}-{datetime.utcnow().strftime('%Y%m%d%H%M')}"
        order_hash = hashlib.sha256(raw_hash_string.encode()).hexdigest()

        if check_idempotency(order_hash):
            warning_msg = f"Duplicate order prevented by idempotency check: Hash {order_hash[:8]} for {clean_pair} {clean_side}"
            logger.warning(warning_msg)
            log_audit(
                venue="LUNO",
                asset=clean_pair,
                price=float(price_dec),
                amount=estimated_val,
                units="DUPLICATE",
                status="DUPLICATE_BLOCKED"
            )
            return {"status": "BLOCKED", "reason": warning_msg}

        # 3. Enforce Circuit Breakers & Position Limits
        passed, reason = self.circuit_breaker.validate_order(
            asset_pair=clean_pair,
            order_value=estimated_val,
            current_daily_trades=self.daily_trades_count,
            daily_pnl_pct=daily_pnl_pct
        )

        if not passed:
            logger.error(f"Order rejected by Circuit Breaker: {reason}")
            log_audit(
                venue="LUNO",
                asset=clean_pair,
                price=float(price_dec),
                amount=estimated_val,
                units="REJECTED",
                status=f"CB_REJECTED: {reason}"
            )
            return {"status": "REJECTED", "reason": reason}

        # 4. Compute Exchange-Compliant Limit Price & Base Volume
        if price_dec <= 0:
            return {"status": "FAILED", "error": f"Invalid spot price ({price_dec}) provided for order execution."}

        price_tick = LUNO_PRICE_PRECISION.get(clean_pair, Decimal("0.01"))
        vol_tick = LUNO_VOLUME_PRECISION.get(clean_pair, Decimal("0.0001"))
        min_vol_limit = LUNO_MIN_BASE_VOLUMES.get(clean_pair, Decimal("0.001"))

        if clean_side in ["BUY", "BID"]:
            raw_limit_price = price_dec * (Decimal("1") + self.aggressive_limit_buffer)
        else:
            raw_limit_price = price_dec * (Decimal("1") - self.aggressive_limit_buffer)

        limit_price = raw_limit_price.quantize(price_tick, rounding=ROUND_DOWN)
        if limit_price <= 0:
            return {"status": "FAILED", "error": f"Quantized limit price evaluated to zero for {clean_pair}."}

        if volume is not None and Decimal(str(volume)) > 0:
            crypto_units = Decimal(str(volume)).quantize(vol_tick, rounding=ROUND_DOWN)
        else:
            crypto_units = (resolved_val / limit_price).quantize(vol_tick, rounding=ROUND_DOWN)

        notional_ngn = crypto_units * limit_price

        if crypto_units < min_vol_limit or notional_ngn < MIN_ORDER_NGN:
            error_msg = (
                f"Quantized order for {clean_pair} (vol={crypto_units}, notional=NGN {notional_ngn:,.2f}) "
                f"is below Luno minimum base volume ({min_vol_limit}) or NGN 50 floor."
            )
            logger.warning(error_msg)
            log_audit(
                venue="LUNO",
                asset=clean_pair,
                price=float(limit_price),
                amount=float(notional_ngn),
                units=str(crypto_units),
                status="BELOW_BASE_VOLUME"
            )
            return {"status": "SKIPPED", "reason": error_msg}

        # 5. Dispatch Live Order via Luno Connector & Verify Exchange Response
        try:
            logger.info(
                f"Dispatching LIVE {clean_side} order on {clean_pair} | "
                f"Volume: {crypto_units} | Limit Price: NGN {limit_price:,.2f} | Notional: NGN {notional_ngn:,.2f}"
            )

            response = self.connector.post_order(
                pair=clean_pair,
                side=clean_side,
                volume=crypto_units,
                price=limit_price
            )

            if not response.get("success"):
                err_detail = response.get("error", "Unknown Luno API rejection")
                logger.error(f"Luno rejected {clean_side} order on {clean_pair}: {err_detail}")
                record_order(
                    order_hash=order_hash,
                    order_id="REJECTED",
                    asset=clean_pair,
                    side=clean_side,
                    amount=float(notional_ngn),
                    status="FAILED"
                )
                log_audit(
                    venue="LUNO",
                    asset=clean_pair,
                    price=float(limit_price),
                    amount=float(notional_ngn),
                    units=str(crypto_units),
                    status=f"LUNO_REJECTED: {err_detail}"
                )
                return {"status": "FAILED", "error": err_detail, "order_response": response}

            order_data = response.get("data", {})
            order_id = order_data.get("order_id")
            if not order_id:
                err_detail = f"Luno response lacked order_id: {response}"
                logger.error(err_detail)
                return {"status": "FAILED", "error": err_detail, "order_response": response}

            record_order(
                order_hash=order_hash,
                order_id=str(order_id),
                asset=clean_pair,
                side=clean_side,
                amount=float(notional_ngn),
                status="SUCCESS"
            )

            log_audit(
                venue="LUNO",
                asset=clean_pair,
                price=float(limit_price),
                amount=float(notional_ngn),
                units=str(crypto_units),
                status=f"FILLED_OR_PLACED: {order_id}"
            )

            self.daily_trades_count += 1
            return {
                "status": "SUCCESS",
                "order_id": str(order_id),
                "executed_volume": str(crypto_units),
                "limit_price": float(limit_price),
                "notional_ngn": float(notional_ngn),
                "order_response": response
            }

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Order execution exception on {clean_pair}: {error_msg}")
            record_order(
                order_hash=order_hash,
                order_id="EXCEPTION",
                asset=clean_pair,
                side=clean_side,
                amount=float(notional_ngn),
                status="FAILED"
            )
            log_audit(
                venue="LUNO",
                asset=clean_pair,
                price=float(limit_price),
                amount=float(notional_ngn),
                units=str(crypto_units),
                status=f"EXCEPTION: {error_msg}"
            )
            return {"status": "FAILED", "error": error_msg}

    def execute_market_order(self, **kwargs) -> dict:
        """Alias routing market order calls through the validated limit-order executor."""
        return self.execute_limit_order(**kwargs)
