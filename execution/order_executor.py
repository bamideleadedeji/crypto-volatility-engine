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
    idempotency hash checks, and mandatory SQLite WAL audit logging as per ICIE Section 16[cite: 1].
    """

    def __init__(self, connector: LunoConnector, circuit_breaker: CircuitBreakerManager):
        self.connector = connector
        self.circuit_breaker = circuit_breaker
        self.daily_trades_count = 0

    def execute_market_order(self, pair: str, side: str, volume: str = None, counter_volume: str = None, daily_pnl_pct: float = 0.0) -> dict:
        """
        Validates circuit breakers, checks idempotency hash, executes via Luno, 
        and records audits and orders to SQLite WAL tables.
        """
        estimated_val = float(volume) if volume else (float(counter_volume) if counter_volume else 0.0)

        # 1. Generate Unique Order Hash for Idempotency Protection
        raw_hash_string = f"{pair}-{side}-{volume}-{counter_volume}-{datetime.now().strftime('%Y%m%d')}"
        order_hash = hashlib.sha256(raw_hash_string.encode()).hexdigest()

        if check_idempotency(order_hash):
            warning_msg = f"Duplicate order prevented by idempotency check: Hash {order_hash[:8]} for {pair} {side}"
            logger.warning(warning_msg)
            log_audit(venue="LUNO", asset=pair, price=0.0, amount=estimated_val, units=volume or "COUNTER", status="DUPLICATE_BLOCKED")
            return {"status": "BLOCKED", "reason": "Duplicate order execution attempt detected and blocked."}

        # 2. Enforce Circuit Breakers & Position Limits[cite: 1]
        passed, reason = self.circuit_breaker.validate_order(
            asset_pair=pair,
            order_value=estimated_val,
            current_daily_trades=self.daily_trades_count,
            daily_pnl_pct=daily_pnl_pct
        )

        if not passed:
            logger.error(f"Order rejected by Circuit Breaker: {reason}")
            log_audit(venue="LUNO", asset=pair, price=0.0, amount=estimated_val, units=volume or "COUNTER", status="REJECTED")
            return {"status": "REJECTED", "reason": reason}

        # 3. Execute Order via Luno Connector
        try:
            logger.info(f"Executing {side} market order for {pair} (Value/Volume: {estimated_val})")
            response = self.connector.place_market_order(pair=pair, side=side, volume=volume, counter_volume=counter_volume)
            
            order_id = response.get("order_id", "UNKNOWN")
            
            # 4. Record to Orders Table & Audit Ledger (ICIE Section 16)[cite: 1]
            record_order(
                order_hash=order_hash,
                order_id=order_id,
                asset=pair,
                side=side,
                amount=estimated_val,
                status="SUCCESS"
            )
            
            log_audit(
                venue="LUNO",
                asset=pair,
                price=0.0,
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
                side=side,
                amount=estimated_val,
                status="FAILED"
            )
            
            log_audit(
                venue="LUNO",
                asset=pair,
                price=0.0,
                amount=estimated_val,
                units=volume or "COUNTER",
                status="FAILED"
            )
            
            return {"status": "FAILED", "error": error_msg}
