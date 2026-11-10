import logging
import hashlib
from datetime import datetime
from execution.luno_connector import LunoConnector
from risk.circuit_breakers import CircuitBreakerManager
from database.repository import check_idempotency, record_order, log_audit

logger = logging.getLogger("InstitutionalOrderExecutor")

class InstitutionalOrderExecutor:
    """
    Executes trades on Luno with integrated circuit breaker validation, 
    idempotency hash checks, mandatory SQLite WAL audit logging, and 
    strict parameter mapping. Fully compatible with target_val, volume, and counter_volume.
    """

    def __init__(self, connector: LunoConnector, circuit_breaker: CircuitBreakerManager):
        self.connector = connector
        self.circuit_breaker = circuit_breaker
        self.daily_trades_count = 0

    def execute_market_order(self, pair: str, side: str, target_val: float = 0.0, current_price: float = 0.0, daily_pnl_pct: float = 0.0, volume: str = None, counter_volume: str = None, amount: float = None, quantity: float = None, **kwargs) -> dict:
        """
        Validates circuit breakers, checks idempotency hash, enforces exchange minimums,
        maps Luno volume parameters correctly, executes via Luno, and records audits to SQLite WAL.
        """
        resolved_val = target_val or amount or 0.0
        if resolved_val <= 0:
            if volume and current_price > 0:
                resolved_val = float(volume) * current_price
            elif counter_volume:
                resolved_val = float(counter_volume)
            elif quantity and current_price > 0:
                resolved_val = float(quantity) * current_price

        estimated_val = float(resolved_val) if resolved_val > 0 else 0.0

        # Enforce ICIE Blueprint Rule 15: Minimum Order of ₦50
        MIN_ORDER_NGN = 50.0
        if estimated_val < MIN_ORDER_NGN:
            error_msg = f"Order value (₦{estimated_val:,.2f}) is below the strict minimum order threshold of ₦{MIN_ORDER_NGN}."
            logger.warning(error_msg)
            log_audit(venue="LUNO", asset=pair, price=current_price, amount=estimated_val, units="SKIPPED", status="BELOW_MIN_LIMIT")
            return {"status": "SKIPPED", "reason": error_msg}

        # 1. Generate Unique Order Hash for Idempotency Protection
        raw_hash_string = f"{pair}-{side.upper()}-{estimated_val:.2f}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        order_hash = hashlib.sha256(raw_hash_string.encode()).hexdigest()

        if check_idempotency(order_hash):
            warning_msg = f"Duplicate order prevented by idempotency check: Hash {order_hash[:8]} for {pair} {side}"
            logger.warning(warning_msg)
            log_audit(venue="LUNO", asset=pair, price=current_price, amount=estimated_val, units="DUPLICATE", status="DUPLICATE_BLOCKED")
            return {"status": "BLOCKED", "reason": "Duplicate order execution attempt detected and blocked."}

        # 2. Enforce Circuit Breakers & Position Limits
        passed, reason = self.circuit_breaker.validate_order(
            asset_pair=pair,
            order_value=estimated_val,
            current_daily_trades=self.daily_trades_count,
            daily_pnl_pct=daily_pnl_pct
        )

        if not passed:
            logger.error(f"Order rejected by Circuit Breaker: {reason}")
            log_audit(venue="LUNO", asset=pair, price=current_price, amount=estimated_val, units="REJECTED", status="REJECTED")
            return {"status": "REJECTED", "reason": reason}

        # 3. Format Parameters for Luno API with Exchange Base Volume Compliance
        resolved_volume = volume
        resolved_counter_volume = counter_volume

        # Luno Exchange Minimum Base Volume Limits
        min_base_volumes = {
            "BTCNGN": 0.0005,
            "ETHNGN": 0.005,
            "SOLNGN": 0.05
        }
        min_vol_limit = min_base_volumes.get(pair, 0.001)

        if side.upper() == "BUY":
            resolved_counter_volume = str(round(estimated_val, 2))
            resolved_volume = None
        elif side.upper() == "SELL":
            if current_price <= 0:
                return {"status": "FAILED", "error": f"Invalid spot price ({current_price}) provided for SELL order base volume calculation."}
            
            # Apply fee haircut while validating against exchange base volume limits
            adjusted_val = estimated_val * 0.98
            crypto_units = adjusted_val / current_price
            
            if crypto_units < min_vol_limit:
                error_msg = f"Calculated sell volume ({crypto_units:.6f}) for {pair} is below Luno minimum base volume ({min_vol_limit})."
                logger.warning(error_msg)
                log_audit(venue="LUNO", asset=pair, price=current_price, amount=estimated_val, units="SKIPPED", status="BELOW_BASE_VOLUME")
                return {"status": "SKIPPED", "reason": error_msg}

            precision = 8 if "XBT" in pair or "BTC" in pair else 6
            resolved_volume = f"{crypto_units:.{precision}f}"
            resolved_counter_volume = None

        # 4. Execute Order via Luno Connector
        try:
            logger.info(f"Executing {side.upper()} market order for {pair} (Value: ₦{estimated_val:,.2f}, Volume Param: {resolved_volume or resolved_counter_volume})")
            response = self.connector.place_market_order(
                pair=pair, 
                side=side.upper(), 
                volume=resolved_volume, 
                counter_volume=resolved_counter_volume
            )
            
            order_id = response.get("order_id", "UNKNOWN")
            
            # 5. Record to Orders Table & Audit Ledger
            record_order(
                order_hash=order_hash,
                order_id=order_id,
                asset=pair,
                side=side.upper(),
                amount=estimated_val,
                status="SUCCESS"
            )
            
            log_audit(
                venue="LUNO",
                asset=pair,
                price=current_price,
                amount=estimated_val,
                units=resolved_volume or "COUNTER",
                status="SUCCESS"
            )
            
            self.daily_trades_count += 1
            return {"status": "SUCCESS", "order_response": response}

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Order execution failed: {error_msg}")
            
            record_order(
                order_hash=order_hash,
                order_id="FAILED",
                asset=pair,
                side=side.upper(),
                amount=estimated_val,
                status="FAILED"
            )
            
            log_audit(
                venue="LUNO",
                asset=pair,
                price=current_price,
                amount=estimated_val,
                units=resolved_volume or "COUNTER",
                status="FAILED"
            )
            
            return {"status": "FAILED", "error": error_msg}
