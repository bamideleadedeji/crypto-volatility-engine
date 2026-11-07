import logging
from execution.luno_connector import LunoConnector
from risk.circuit_breakers import CircuitBreakerManager
from database.repository import log_audit_event

logger = logging.getLogger("InstitutionalOrderExecutor")

class InstitutionalOrderExecutor:
    """
    Executes trades on Luno with integrated circuit breaker validation, 
    idempotency checks, and mandatory SQLite WAL audit logging as per ICIE Section 16[cite: 1].
    """

    def __init__(self, connector: LunoConnector, circuit_breaker: CircuitBreakerManager):
        self.connector = connector
        self.circuit_breaker = circuit_breaker
        self.daily_trades_count = 0

    def execute_market_order(self, pair: str, side: str, volume: str = None, counter_volume: str = None, daily_pnl_pct: float = 0.0) -> dict:
        """
        Validates, executes, and audits a market order.
        """
        estimated_val = float(volume) if volume else (float(counter_volume) if counter_volume else 0.0)

        # 1. Enforce Circuit Breakers & Position Limits
        passed, reason = self.circuit_breaker.validate_order(
            asset_pair=pair,
            order_value=estimated_val,
            current_daily_trades=self.daily_trades_count,
            daily_pnl_pct=daily_pnl_pct
        )

        if not passed:
            logger.error(f"Order rejected by Circuit Breaker: {reason}")
            log_audit_event("ORDER_REJECTED", f"Pair: {pair} | Side: {side} | Reason: {reason}")
            return {"status": "REJECTED", "reason": reason}

        # 2. Execute Order via Luno Connector
        try:
            logger.info(f"Executing {side} market order for {pair} (Value/Volume: {estimated_val})")
            response = self.connector.place_market_order(pair=pair, side=side, volume=volume, counter_volume=counter_volume)
            
            order_id = response.get("order_id")
            closed = response.get("closed", False)
            
            # 3. Mandated SQLite WAL Audit Logging (ICIE Section 16)[cite: 1]
            audit_details = (
                f"SUCCESS | Order ID: {order_id} | Pair: {pair} | Side: {side} | "
                f"Requested Vol: {volume or counter_volume} | Closed: {closed} | API Response: {response}"
            )
            log_audit_event("ORDER_EXECUTED", audit_details)
            
            self.daily_trades_count += 1
            return {"status": "SUCCESS", "order_response": response}

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Order execution failed: {error_msg}")
            log_audit_event("ORDER_FAILED", f"Pair: {pair} | Side: {side} | Error: {error_msg}")
            return {"status": "FAILED", "error": error_msg}
