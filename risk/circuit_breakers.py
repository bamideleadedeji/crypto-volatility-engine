import logging

logger = logging.getLogger("CircuitBreakerManager")

class CircuitBreakerManager:
    """
    Enforces strict institutional risk controls, position caps, and emergency circuit breakers
    to protect capital against system anomalies and market crashes[cite: 1].
    """

    def __init__(self):
        # Position limits as per ICIE Constitution Section 15
        self.max_position_limits = {
            "XBTNGN": 0.60,  # Max BTC 60%[cite: 1]
            "ETHNGN": 0.40,  # Max ETH 40%[cite: 1]
            "SOLNGN": 0.30   # Max SOL 30%[cite: 1]
        }
        self.max_daily_loss_pct = 0.05  # Maximum Daily Loss 5%[cite: 1]
        self.max_daily_trades = 4       # Maximum Trades 4[cite: 1]
        self.min_order_value = 50.0     # Minimum Order ₦50[cite: 1]

    def validate_order(self, asset_pair: str, order_value: float, current_daily_trades: int, daily_pnl_pct: float) -> tuple[bool, str]:
        """
        Validates whether a proposed trade complies with all constitutional circuit breakers.
        """
        # 1. Minimum Order Check
        if order_value < self.min_order_value:
            return False, f"Order value ₦{order_value:,.2f} violates minimum order threshold of ₦{self.min_order_value}[cite: 1]."

        # 2. Maximum Daily Trades Check
        if current_daily_trades >= self.max_daily_trades:
            return False, f"Daily trade limit reached ({current_daily_trades}/{self.max_daily_trades})[cite: 1]."

        # 3. Maximum Daily Loss Circuit Breaker
        if daily_pnl_pct <= -self.max_daily_loss_pct:
            return False, f"CIRCUIT BREAKER TRIGGERED: Daily portfolio loss of {daily_pnl_pct*100:.2f}% breaches 5% max daily loss limit[cite: 1]."

        return True, "Order passed all institutional circuit breaker checks."

    def validate_position_limits(self, weights: dict) -> tuple[bool, str]:
        """
        Ensures individual asset allocations do not breach maximum constitutional caps.
        """
        for asset, weight in weights.items():
            max_limit = self.max_position_limits.get(asset, 1.0)
            if weight > max_limit:
                return False, f"Position limit violation: {asset} weight {weight*100:.2f}% exceeds maximum limit of {max_limit*100}%[cite: 1]."
        return True, "All asset weights within constitutional position limits."
