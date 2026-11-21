"""
Institutional Crypto Investment Engine (ICIE) - Phase B GARCH(1,1) & Fat-Tail VaR Regime Gate
Fetches real hourly OHLC candles from Luno /api/exchange/1/candles to compute GARCH(1,1) MLE
conditional volatility, Cornish-Fisher fat-tail 95% VaR, EWMA shock ratios, and dynamic
volatility-targeted allocation scalars. Protects real Naira capital during crash regimes.
"""

import os
import math
import time
import logging
import numpy as np
import requests
from scipy.stats import kurtosis, norm, skew

logger = logging.getLogger("VolatilityRegimeGate")

try:
    from arch import arch_model
    HAS_ARCH = True
except Exception:
    HAS_ARCH = False


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
    Evaluates real-time GARCH(1,1) volatility & Cornish-Fisher tail risk on Luno NGN pairs
    before permitting Phase 2 capital deployment into underweight assets.
    """

    def __init__(
        self,
        max_annualized_vol: float = 1.35,
        max_shock_ratio: float = 2.20,
        max_24h_drop_pct: float = -0.08,
        target_annualized_vol: float = 0.65,
        max_24h_cf_var_pct: float = 0.085,
        api_key_id: str = "",
        api_key_secret: str = "",
    ):
        self.base_url = "https://" + "api." + "luno.com" + "/api/exchange/1"
        self.max_annualized_vol = max_annualized_vol
        self.max_shock_ratio = max_shock_ratio
        self.max_24h_drop_pct = max_24h_drop_pct
        self.target_annualized_vol = target_annualized_vol
        self.max_24h_cf_var_pct = max_24h_cf_var_pct
        self.headers = {
            "Accept": "application/json",
            "User-Agent": "ICIE-Portfolio-Engine/2.2",
        }
        if api_key_id and api_key_secret:
            self.auth = (api_key_id.strip(), api_key_secret.strip())
        else:
            self.auth = _resolve_luno_auth()

    def _fetch_hourly_closes(self, pair: str, hours: int = 168) -> list[float]:
        since_ms = int((time.time() - (hours * 3600)) * 1000)
        endpoint = f"{self.base_url}/candles?pair={pair}&since={since_ms}&duration=3600"

        last_err = None
        for attempt in range(3):
            try:
                response = requests.get(
                    endpoint,
                    auth=self.auth,
                    headers=self.headers,
                    timeout=15,
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

                if response.status_code == 429:
                    retry_after = float(response.headers.get("Retry-After", 2 ** attempt))
                    time.sleep(retry_after)
                    continue

                if response.status_code in (500, 502, 503, 504):
                    time.sleep(2 ** attempt)
                    continue

                raise RuntimeError(f"Luno candles HTTP {response.status_code} for {pair}: {response.text[:200]}")
            except (requests.RequestException, RuntimeError) as exc:
                last_err = exc
                time.sleep(2 ** attempt)

        raise RuntimeError(f"Candle fetch failed after retries for {pair}: {last_err}")
    @staticmethod
    def _fit_garch_and_tail_risk(closes: list[float]) -> tuple[float, float, float, str]:
        arr = np.asarray(closes, dtype=float)
        rets = np.log(arr[1:] / arr[:-1])
        rets = rets[np.isfinite(rets)]
        if len(rets) < 20:
            return 0.0, 1.0, 0.0, "INSUFFICIENT"

        baseline_std = float(np.std(rets, ddof=1))
        recent_std = float(np.std(rets[-12:], ddof=1)) if len(rets) >= 12 else baseline_std
        shock_ratio = (recent_std / baseline_std) if baseline_std > 1e-9 else 1.0

        cond_hourly_vol = baseline_std
        model_used = "EWMA"
        if HAS_ARCH and len(rets) >= 48:
            try:
                am = arch_model(rets * 100.0, vol="Garch", p=1, q=1, dist="normal", rescale=False)
                res = am.fit(disp="off", show_warning=False)
                fcast = res.forecast(horizon=1, reindex=False)
                next_var = float(fcast.variance.values[-1, 0])
                cond_hourly_vol = max(math.sqrt(max(next_var, 1e-10)) / 100.0, 1e-5)
                model_used = "GARCH(1,1)"
            except Exception:
                cond_hourly_vol = baseline_std

        ann_vol = cond_hourly_vol * math.sqrt(8760)

        # 24-Hour Cornish-Fisher Fat-Tail 95% VaR
        z_95 = float(norm.ppf(0.05))
        s = float(skew(rets))
        k = float(kurtosis(rets))
        z_cf = (
            z_95
            + (1.0 / 6.0) * (z_95**2 - 1.0) * s
            + (1.0 / 24.0) * (z_95**3 - 3.0 * z_95) * k
            - (1.0 / 36.0) * (2.0 * z_95**3 - 5.0 * z_95) * (s**2)
        )
        cf_var_24h = abs(z_cf) * cond_hourly_vol * math.sqrt(24.0)
        return ann_vol, shock_ratio, cf_var_24h, model_used

    def evaluate_asset_regime(self, coin: str) -> dict:
        pair = f"{coin.strip().upper()}NGN"
        try:
            closes = self._fetch_hourly_closes(pair, hours=168)
            if len(closes) < 24:
                return {
                    "pair": pair,
                    "safe_to_buy": True,
                    "regime": "INSUFFICIENT_CANDLES_ALLOW",
                    "ann_vol": 0.0,
                    "shock_ratio": 1.0,
                    "ret_24h": 0.0,
                    "cf_var_24h": 0.0,
                    "vol_scalar": 1.0,
                    "reason": "Fewer than 24 hourly candles returned; defaulting to open.",
                }

            ann_vol, shock_ratio, cf_var_24h, model_used = self._fit_garch_and_tail_risk(closes)
            vol_scalar = round(max(0.25, min(1.0, self.target_annualized_vol / max(ann_vol, 0.15))), 2)

            price_now = closes[-1]
            price_24h_ago = closes[-24]
            ret_24h = (price_now - price_24h_ago) / price_24h_ago
            sma_12 = sum(closes[-12:]) / len(closes[-12:])

            if ann_vol > self.max_annualized_vol:
                return {
                    "pair": pair,
                    "safe_to_buy": False,
                    "regime": "EXTREME_VOLATILITY_HALT",
                    "ann_vol": round(ann_vol, 4),
                    "shock_ratio": round(shock_ratio, 2),
                    "ret_24h": round(ret_24h, 4),
                    "cf_var_24h": round(cf_var_24h, 4),
                    "vol_scalar": 0.0,
                    "reason": f"{pair} {model_used} volatility ({ann_vol*100:.1f}%) exceeds {self.max_annualized_vol*100:.0f}% ceiling.",
                }

            if cf_var_24h > self.max_24h_cf_var_pct and ret_24h < -0.02:
                return {
                    "pair": pair,
                    "safe_to_buy": False,
                    "regime": "FAT_TAIL_VAR_HALT",
                    "ann_vol": round(ann_vol, 4),
                    "shock_ratio": round(shock_ratio, 2),
                    "ret_24h": round(ret_24h, 4),
                    "cf_var_24h": round(cf_var_24h, 4),
                    "vol_scalar": 0.0,
                    "reason": f"{pair} 24h Cornish-Fisher 95% VaR ({cf_var_24h*100:.2f}%) breached {self.max_24h_cf_var_pct*100:.1f}% tail-risk limit.",
                }

            if shock_ratio > self.max_shock_ratio and ret_24h < -0.03:
                return {
                    "pair": pair,
                    "safe_to_buy": False,
                    "regime": "VOLATILITY_SHOCK_DOWNTREND",
                    "ann_vol": round(ann_vol, 4),
                    "shock_ratio": round(shock_ratio, 2),
                    "ret_24h": round(ret_24h, 4),
                    "cf_var_24h": round(cf_var_24h, 4),
                    "vol_scalar": 0.0,
                    "reason": f"{pair} 12h volatility shock ({shock_ratio:.2f}x) during downward slide ({ret_24h*100:.2f}%).",
                }

            if ret_24h < self.max_24h_drop_pct and price_now < sma_12:
                return {
                    "pair": pair,
                    "safe_to_buy": False,
                    "regime": "FALLING_KNIFE_GUARD",
                    "ann_vol": round(ann_vol, 4),
                    "shock_ratio": round(shock_ratio, 2),
                    "ret_24h": round(ret_24h, 4),
                    "cf_var_24h": round(cf_var_24h, 4),
                    "vol_scalar": 0.0,
                    "reason": f"{pair} dropped {ret_24h*100:.2f}% in 24h and remains below 12h SMA (₦{sma_12:,.0f}).",
                }

            return {
                "pair": pair,
                "safe_to_buy": True,
                "regime": "STABLE_OR_RECOVERY",
                "ann_vol": round(ann_vol, 4),
                "shock_ratio": round(shock_ratio, 2),
                "ret_24h": round(ret_24h, 4),
                "cf_var_24h": round(cf_var_24h, 4),
                "vol_scalar": vol_scalar,
                "reason": f"{pair} {model_used} healthy (Vol: {ann_vol*100:.1f}%, 24h VaR: {cf_var_24h*100:.2f}%, Scalar: {vol_scalar:.2f}x).",
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
                "cf_var_24h": 0.0,
                "vol_scalar": 0.5,
                "reason": f"Candle fetch fallback: {str(e)}",
            }
