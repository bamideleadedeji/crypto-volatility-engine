import sqlite3
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from core.logging import setup_logger

logger = setup_logger("RiskManager")
DB_NAME = "luno_institutional_engine.db"

class RiskManager:
    def __init__(self, config: dict):
        self.config = config
        self.drift_tolerance = config.get("drift_tolerance", 0.03)
        self.max_failures = config.get("max_failures", 5)
        self.cooldown_minutes = config.get("cooldown_minutes", 15)
        self.max_single_asset_weight = config.get("max_single_asset_weight", 0.60)
        self.max_daily_loss_pct = config.get("max_daily_loss_pct", 0.05)

    def check_circuit_breaker(self) -> bool:
        with sqlite3.connect(DB_NAME, timeout=30) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT failure_count, is_tripped, last_trip_time FROM circuit_breaker_state WHERE id=1")
            row = cursor.fetchone()
        if not row:
            return True
        _, is_tripped, last_trip_time = row[0], row[1], row[2]
        if is_tripped:
            if last_trip_time:
                trip_dt = datetime.strptime(last_trip_time, "%Y-%m-%d %H:%M:%S")
                if datetime.now() - trip_dt > timedelta(minutes=self.cooldown_minutes):
                    logger.info("🔄 Circuit breaker cooldown elapsed. Automatic reset executed.")
                    self.reset_circuit_breaker()
                    return True
            logger.warning("🚨 CIRCUIT BREAKER ACTIVE: Trading halted.")
            return False
        return True

    def record_api_failure(self):
        with sqlite3.connect(DB_NAME, timeout=30) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT failure_count FROM circuit_breaker_state WHERE id=1")
            count = cursor.fetchone()[0] + 1
            is_tripped = 1 if count >= self.max_failures else 0
            trip_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S") if is_tripped else ""
            cursor.execute('''
                UPDATE circuit_breaker_state 
                SET failure_count = ?, is_tripped = ?, last_trip_time = ? 
                WHERE id = 1
            ''', (count, is_tripped, trip_time))
            conn.commit()

    def reset_circuit_breaker(self):
        with sqlite3.connect(DB_NAME, timeout=30) as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE circuit_breaker_state SET failure_count = 0, is_tripped = 0, last_trip_time = '' WHERE id=1")
            conn.commit()

    def evaluate_emergency_risk_limits(self, current_weights: dict, portfolio_val: float) -> bool:
        for asset, weight in current_weights.items():
            if weight > self.max_single_asset_weight:
                logger.error(f"🚨 EMERGENCY: {asset} weight ({weight*100:.2f}%) exceeds max limit ({self.max_single_asset_weight*100}%).")
                return False

        with sqlite3.connect(DB_NAME, timeout=30) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(portfolio_value) FROM portfolio_history")
            row = cursor.fetchone()
            peak_val = row[0] if row and row[0] is not None else portfolio_val

        if peak_val > 0:
            drawdown = (peak_val - portfolio_val) / peak_val
            if drawdown > self.max_daily_loss_pct:
                logger.error(f"🚨 EMERGENCY: Drawdown ({drawdown*100:.2f}%) exceeds daily loss limit ({self.max_daily_loss_pct*100}%).")
                return False
        return True

    def evaluate_portfolio_drift(self, current_weights: dict, target_weights: dict) -> bool:
        max_drift = max(abs(current_weights.get(a, 0.0) - t) for a, t in target_weights.items())
        logger.info(f"Portfolio Drift -> Max Drift: {max_drift*100:.2f}% (Threshold: {self.drift_tolerance*100}%)")
        return max_drift >= self.drift_tolerance
