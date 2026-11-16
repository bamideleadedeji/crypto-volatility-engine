"""
Institutional Crypto Investment Engine (ICIE) - Phase B Volatility & Trend Regime Gate
Fetches real hourly OHLC candles from Luno /api/exchange/1/candles to compute rolling volatility,
EWMA shock ratios, and momentum filters. Protects capital by halting Phase 2 dip-buys
during active crash regimes while permitting Phase 1 profit-taking into NGN.
"""

import os
import math
import time
import logging
import requests

logger = logging.getLogger("VolatilityRegimeGate")


def _resolve_luno_auth() -> tuple[str, str] | None:
    key_id = os.environ.get("LUNO_API_KEY_ID", "").strip()
    key_secret = os.environ.get("LUNO_API_KEY_SECRET", "").strip()
    if not key_id or not key_secret:
        try:
            import streamlit as st
            luno_sec = st.secrets.get("luno", {}) if hasattr(st.secrets, "get") else {}
            key_id = str(st.secrets.get("LUNO_API_KEY_ID", luno_sec.get("LUNO_API_KEY_ID", ""))).strip()
            key_secret = str(st.secrets.get("LUNO_API_KEY_SECRET", luno_sec.get("LUNO_API_KEY_SECRET", ""))).strip()
        except Exception:
            pass
    return (key_id, key_secret) if (key_id and key_secret) else None


class VolatilityRegimeGate:
    """
    Evaluates real-time market regime on Luno NGN pairs before permitting Phase 2
    capital deployment into underweight assets.
    """

    def __init__(
        self,
        max_annualized_vol: float = 1.35,   # 135% annualized hourly vol ceiling
        max_shock_ratio: float = 2.20,      # Short-term (12h) vs baseline (72h) vol spike ceiling
        max_24h_drop_pct: float = -0.08,    # Halt buying an asset if it fell >8% in 24h without bouncing
        api_key_id: str = "",
        api_key_secret: str = ""
    ):
        self.base_url = "https://" + "api." + "luno.com" + "/api/exchange/1"
        self.max_annualized_vol = max_annualized_vol
        self.max_shock_ratio = max_shock_ratio
        self.max_24h_drop_pct = max_24h_drop_pct
        self.headers = {
            "Accept": "application/json",
            "User-Agent": "ICIE-Portfolio-Engine/1.0",
        }
        if api_key_id and api_key_secret:
            self.auth = (api_key_id.strip(), api_key_secret.strip())
        else:
            self.auth = _resolve_luno_auth()

    def _fetch_hourly_closes(self, pair: str, hours: int = 72) -> list[float]:
        since_ms = int((time.time() - (hours * 3600)) * 1000)
        endpoint = f"{self.base_url}/candles?pair={pair}&since={since_ms}&duration=3600"

        last_err = None
        for attempt in range(3):
            try:
                response = requests.get(
                    endpoint,
                    auth=self.auth,
                    headers=self.headers,
                    timeout=15
                )
                if response.status_code == 200:
                    try:
                        payload = response.json()
                    except ValueError:
                        preview = response.text[:150] if response.text else "<EMPTY BODY>"
                        raise RuntimeError(f"Non-JSON response on {pair}: {preview}")

                    candles = payload.get("candles", [])
                    closes = [float(c["close"]) for c in candles if float(c.get("close", 0)) > 0]
                    return closes

                if response.status_code in (429, 500, 502, 503, 504):
                    time.sleep(2 ** attempt)
                    continue

                raise RuntimeError(f"Luno candles HTTP {response.status_code} for {pair}: {response.text[:200]}")
            except (requests.RequestException, RuntimeError) as exc:
                last_err = exc
                time.sleep(2 ** attempt)

        raise RuntimeError(f"Candle fetch failed after retries for {pair}: {last_err}")

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
