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
    strict parameter mapping. Uses *args and **kwargs to guarantee 
    zero signature mismatch errors across cached or multi-version deployments.
    """

    def __init__(self, connector: LunoConnector, circuit_breaker: CircuitBreakerManager):
        self.connector = connector
        self.circuit_breaker = circuit_breaker
        self.daily_trades_count = 0

    def execute_market_order(self, pair: str, side: str, *args, **kwargs) -> dict:
        """
        Validates circuit breakers, checks idempotency hash, maps Luno volume parameters correctly,
        executes via Luno, and records audits and orders to SQLite WAL tables.
        Polymorphic signature accepts any combination of target_val, current_price, volume, or counter_volume.
        """
        # 1. Extract parameters robustly regardless of how the caller invokes them
        target_val = kwargs.get("target_val", 0.0)
        if not target_val and len(args) > 0:
            target_val = args[0]

        current_price = kwargs.get("current_price", 0.0)
        if not current_price and len(args) > 1:
            current_price = args[1]

        daily_pnl_pct = kwargs.get("daily_pnl_pct", 0.0)
        explicit_volume = kwargs.get("volume", None)
        explicit_counter_volume = kwargs.get("counter_volume", None)

        # Fallback valuation logic
        if target_val <= 0:
            if explicit_volume and current_price > 0:
                target_val = float(explicit_volume) * current_price
            elif explicit_counter_volume:
                target_val = float(explicit_counter_volume)

        estimated_val = float(target_val) if target_val > 0 else 0.0

        if estimated_val <= 0:
            return {"status": "FAILED", "error": f"Invalid target value ({estimated_val}) provided for order execution."}

        # 2. Generate Unique Order Hash for Idempotency Protection
        raw_hash_string = f"{pair}-{side.upper()}-{estimated_val:.2f}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        order_hash = hashlib.sha256(raw_hash_string.encode()).hexdigest()

        if check_idempotency(order_hash):
            warning_msg = f"Duplicate order prevented by idempotency check: Hash {order_hash[:8]} for {pair} {side}"
            logger.warning(warning_msg)
            log_audit(venue="LUNO", asset=pair, price=current_price, amount=estimated_val, units="DUPLICATE", status="DUPLICATE_BLOCKED")
            return {"status": "BLOCKED", "reason": "Duplicate order execution attempt detected and blocked."}

        # 3. Enforce Circuit Breakers & Position Limits
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

        # 4. Format Parameters for Luno API (Buys use counter_volume, Sells use base_volume)
        volume = explicit_volume
        counter_volume = explicit_counter_volume

        if side.upper() == "BUY":
            counter_volume = str(round(estimated_val, 2))
            volume = None
        elif side.upper() == "SELL":
            if current_price <= 0:
                return {"status": "FAILED", "error": f"Invalid spot price ({current_price}) provided for SELL order base volume calculation."}
            
            crypto_units = estimated_val / current_price
            precision = 8 if "XBT" in pair or "BTC" in pair else 6
            volume = f"{crypto_units:.{precision}f}"
            counter_volume = None

        # 5. Execute Order via Luno Connector
        try:
            logger.info(f"Executing {side.upper()} market order for {pair} (Value: ₦{estimated_val:,.2f}, Volume Param: {volume or counter_volume})")
            response = self.connector.place_market_order(
                pair=pair, 
                side=side.upper(), 
                volume=volume, 
                counter_volume=counter_volume
            )
            
            order_id = response.get("order_id", "UNKNOWN")
            
            # 6. Record to Orders Table & Audit Ledger
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
                units=volume or "COUNTER",
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
                units=volume or "COUNTER",
                status="FAILED"
            )
            
            return {"status": "FAILED", "error": error_msg}
