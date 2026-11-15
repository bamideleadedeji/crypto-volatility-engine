"""
Institutional Crypto Investment Engine (ICIE) - Phase B Volatility & Trend Regime Gate
Fetches real hourly OHLC candles from Luno /api/1/candles to compute rolling volatility,
EWMA shock ratios, and momentum filters. Protects capital by halting Phase 2 dip-buys
during active crash regimes while permitting Phase 1 profit-taking into NGN.
"""

import math
import time
import logging
from decimal import Decimal
import requests

logger = logging.getLogger("VolatilityRegimeGate")


class VolatilityRegimeGate:
    """
    Evaluates real-time market regime on Luno NGN pairs before permitting Phase 2
    capital deployment into underweight assets.
    """

    def __init__(
        self,
        max_annualized_vol: float = 1.35,   # 135% annualized hourly vol ceiling
        max_shock_ratio: float = 2.20,      # Short-term (12h) vs baseline (72h) vol spike ceiling
        max_24h_drop_pct: float = -0.08     # Halt buying an asset if it fell >8% in 24h without bouncing
    ):
        self.base_url = "https://" + "api." + "luno.com" + "/api/1"
        self.max_annualized_vol = max_annualized_vol
        self.max_shock_ratio = max_shock_ratio
        self.max_24h_drop_pct = max_24h_drop_pct

    def _fetch_hourly_closes(self, pair: str, hours: int = 72) -> list[float]:
        since_ms = int((time.time() - (hours * 3600)) * 1000)
        endpoint = f"{self.base_url}/candles?pair={pair}&since={since_ms}&duration=3600"
        response = requests.get(endpoint, timeout=15)
        if response.status_code != 200:
            raise RuntimeError(f"Luno candles HTTP {response.status_code} for {pair}: {response.text[:200]}")

        candles = response.json().get("candles", [])
        closes = [float(c["close"]) for c in candles if float(c.get("close", 0)) > 0]
        return closes

    @staticmethod
    def _compute_log_returns(closes: list[float]) -> list[float]:
        return [
            math.log(closes[i] / closes[i - 1])
            for i in range(1, len(closes))
            if closes[i - 1] > 0 and closes[i] > 0
        ]

    @staticmethod
    def _std(values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)

    def evaluate_asset_regime(self, coin: str) -> dict:
        pair = f"{coin.strip().upper()}NGN"
        try:
            closes = self._fetch_hourly_closes(pair, hours=72)
            if len(closes) < 24:
                return {
                    "pair": pair,
                    "safe_to_buy": True,
                    "regime": "INSUFFICIENT_CANDLES_ALLOW",
                    "ann_vol": 0.0,
                    "shock_ratio": 1.0,
                    "ret_24h": 0.0,
                    "reason": "Fewer than 24 hourly candles returned; defaulting to open."
                }

            returns = self._compute_log_returns(closes)
            baseline_std = self._std(returns)
            recent_std = self._std(returns[-12:]) if len(returns) >= 12 else baseline_std

            # Annualize hourly volatility (sqrt(24 * 365) = 93.5948)
            ann_vol = baseline_std * math.sqrt(8760)
            shock_ratio = (recent_std / baseline_std) if baseline_std > 1e-9 else 1.0

            price_now = closes[-1]
            price_24h_ago = closes[-24]
            ret_24h = (price_now - price_24h_ago) / price_24h_ago
            sma_12 = sum(closes[-12:]) / len(closes[-12:])

            # Crash Detection Rules
            if ann_vol > self.max_annualized_vol:
                return {
                    "pair": pair,
                    "safe_to_buy": False,
                    "regime": "EXTREME_VOLATILITY_HALT",
                    "ann_vol": round(ann_vol, 4),
                    "shock_ratio": round(shock_ratio, 2),
                    "ret_24h": round(ret_24h, 4),
                    "reason": f"{pair} annualized volatility ({ann_vol*100:.1f}%) exceeds {self.max_annualized_vol*100:.0f}% ceiling."
                }

            if shock_ratio > self.max_shock_ratio and ret_24h < -0.03:
                return {
                    "pair": pair,
                    "safe_to_buy": False,
                    "regime": "VOLATILITY_SHOCK_DOWNTREND",
                    "ann_vol": round(ann_vol, 4),
                    "shock_ratio": round(shock_ratio, 2),
                    "ret_24h": round(ret_24h, 4),
                    "reason": f"{pair} 12h volatility shock ({shock_ratio:.2f}x) during downward slide ({ret_24h*100:.2f}%)."
                }

            if ret_24h < self.max_24h_drop_pct and price_now < sma_12:
                return {
                    "pair": pair,
                    "safe_to_buy": False,
                    "regime": "FALLING_KNIFE_GUARD",
                    "ann_vol": round(ann_vol, 4),
                    "shock_ratio": round(shock_ratio, 2),
                    "ret_24h": round(ret_24h, 4),
                    "reason": f"{pair} dropped {ret_24h*100:.2f}% in 24h and remains below 12h SMA (₦{sma_12:,.0f})."
                }

            return {
                "pair": pair,
                "safe_to_buy": True,
                "regime": "STABLE_OR_RECOVERY",
                "ann_vol": round(ann_vol, 4),
                "shock_ratio": round(shock_ratio, 2),
                "ret_24h": round(ret_24h, 4),
                "reason": f"{pair} regime healthy (Vol: {ann_vol*100:.1f}%, 24h: {ret_24h*100:+.2f}%)."
            }

        except Exception as e:
            logger.warning(f"Volatility gate telemetry warning on {pair}: {e}. Defaulting to cautious allow.")
            return {
                "pair": pair,
                "safe_to_buy": True,
                "regime": "TELEMETRY_FALLBACK",
                "ann_vol": 0.0,
                "shock_ratio": 1.0,
                "ret_24h": 0.0,
                "reason": f"Candle fetch fallback: {str(e)}"
            }
