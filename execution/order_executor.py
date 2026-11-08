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
    strict parameter mapping (counter_volume for BUYS, base_volume for SELLS).
    """

    def __init__(self, connector: LunoConnector, circuit_breaker: CircuitBreakerManager):
        self.connector = connector
        self.circuit_breaker = circuit_breaker
        self.daily_trades_count = 0

    def execute_market_order(self, pair: str, side: str, target_val: float, current_price: float = 0.0, daily_pnl_pct: float = 0.0) -> dict:
        """
        Validates circuit breakers, checks idempotency hash, maps Luno volume parameters correctly,
        executes via Luno, and records audits and orders to SQLite WAL tables.
        """
        estimated_val = float(target_val) if target_val else 0.0

        # 1. Generate Unique Order Hash for Idempotency Protection
        raw_hash_string = f"{pair}-{side}-{estimated_val}-{datetime.now().strftime('%Y%m%d')}"
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

        # 3. Format Parameters for Luno API (Buys use counter_volume, Sells use base_volume)
        volume = None
        counter_volume = None

        if side.upper() == "BUY":
            counter_volume = str(round(estimated_val, 2))
        elif side.upper() == "SELL":
            if current_price <= 0:
                return {"status": "FAILED", "error": f"Invalid spot price ({current_price}) provided for SELL order base volume calculation."}
            
            # Calculate exact crypto units (base volume) from target NGN value / spot price
            crypto_units = estimated_val / current_price
            precision = 8 if "XBT" in pair or "BTC" in pair else 6
            volume = f"{crypto_units:.{precision}f}"

        # 4. Execute Order via Luno Connector
        try:
            logger.info(f"Executing {side} market order for {pair} (Value: ₦{estimated_val:,.2f}, Volume Param: {volume or counter_volume})")
            response = self.connector.place_market_order(
                pair=pair, 
                side=side.upper(), 
                volume=volume, 
                counter_volume=counter_volume
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
