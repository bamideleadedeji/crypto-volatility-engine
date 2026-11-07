import pandas as pd
import numpy as np

class MarketRegimeDetector:
    """
    Classifies market conditions (Bull, Bear, High Volatility, Crisis)
    and dynamically adjusts volatility estimation lookback windows as per ICIE Section 12[cite: 1].
    """
    def __init__(self):
        pass

    def detect_regime(self, price_series: pd.Series, rolling_window: int = 30) -> dict:
        """
        Analyzes price series to determine current market regime and optimal GARCH lookback.
        """
        if price_series.empty or len(price_series) < rolling_window:
            return {
                "regime": "SIDEWAYS / NORMAL",
                "current_annualized_volatility": 0.0,
                "current_drawdown_pct": 0.0,
                "recommended_lookback_days": 90,
                "risk_multiplier": 1.0
            }

        returns = price_series.pct_change().dropna()
        rolling_vol = returns.rolling(window=rolling_window).std() * np.sqrt(365)
        current_vol = rolling_vol.iloc[-1] if not rolling_vol.empty else 0.04

        # Calculate drawdown from rolling peak
        rolling_max = price_series.cummax()
        drawdown = (price_series - rolling_max) / rolling_max
        current_drawdown = drawdown.iloc[-1] if not drawdown.empty else 0.0

        # Calculate recent trend (30-day return)
        lookback_period = min(30, len(price_series) - 1)
        trend_return = (price_series.iloc[-1] / price_series.iloc[-lookback_period]) - 1.0

        # Regime Classification Logic & Dynamic Lookback Adjustment
        if current_drawdown <= -0.20 or current_vol > 0.80:
            regime = "CRISIS / PANIC"
            lookback = 30   # Short lookback for rapid response to tail risk
            risk_multiplier = 0.50
        elif current_vol > 0.60:
            regime = "HIGH VOLATILITY"
            lookback = 45
            risk_multiplier = 0.75
        elif trend_return > 0.05 and current_vol <= 0.50:
            regime = "BULL MARKET"
            lookback = 180  # Longer lookback for stable trend smoothing
            risk_multiplier = 1.0
        elif trend_return < -0.05:
            regime = "BEAR MARKET"
            lookback = 90
            risk_multiplier = 0.80
        else:
            regime = "SIDEWAYS / NORMAL"
            lookback = 90
            risk_multiplier = 1.0

        return {
            "regime": regime,
            "current_annualized_volatility": round(current_vol * 100, 2),
            "current_drawdown_pct": round(current_drawdown * 100, 2),
            "trend_return_pct": round(trend_return * 100, 2),
            "recommended_lookback_days": lookback,
            "risk_multiplier": risk_multiplier
        }
