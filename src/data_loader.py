import pandas as pd
import numpy as np
import ccxt
import requests
import streamlit as st
import time
from requests.adapters import HTTPAdapter
from urllib3.util import Retry


def get_resilient_session():
    """Configures a requests session with exponential backoff retries for status code 429."""
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=2,  # Retries after 2s, 4s, 8s
        status_forcelist=[429, 500, 502, 503, 504],
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    return session


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_crypto_data(symbol="BTC-USD", period="1y"):
    """
    Fetches real-time market data from exchange APIs:
    Tier 1: Coinbase Public API (CCXT)
    Tier 2: Kraken Public Spot API (CCXT Fallback)
    Tier 3: CoinGecko Public REST API (Resilient Session Fallback)

    Memory-Optimized for Render Free Tier (512MB RAM Limit).
    """
    df = pd.DataFrame()

    # Normalize Pair Syntax (e.g., BTC-USD -> BTC/USD)
    base_asset = symbol.split("-")[0] if "-" in symbol else symbol.replace("USD", "")
    ccxt_symbol = f"{base_asset}/USD"

    # Limit to 365 days (sufficient statistical depth while saving 50% RAM)
    historical_limit = 365

    # ==========================================
    # Tier 1: Coinbase Public API via CCXT
    # ==========================================
    try:
        exchange_cb = ccxt.coinbase({
            "enableRateLimit": True,
            "timeout": 10000,
        })
        ohlcv = exchange_cb.fetch_ohlcv(ccxt_symbol, timeframe="1d", limit=historical_limit)

        if ohlcv and len(ohlcv) > 30:
            df = pd.DataFrame(
                ohlcv, 
                columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"]
            )
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms")
            df.set_index("Timestamp", inplace=True)
    except Exception as e:
        print(f"Coinbase API Fetch Exception for {ccxt_symbol}: {e}")
        df = pd.DataFrame()

    # ==========================================
    # Tier 2: Kraken Public API Fallback
    # ==========================================
    if df is None or df.empty or len(df) < 30:
        print(f"Coinbase feed offline. Attempting Kraken API for {ccxt_symbol}...")
        try:
            exchange_kraken = ccxt.kraken({
                "enableRateLimit": True,
                "timeout": 10000,
            })
            ohlcv_k = exchange_kraken.fetch_ohlcv(ccxt_symbol, timeframe="1d", limit=historical_limit)

            if ohlcv_k and len(ohlcv_k) > 30:
                df = pd.DataFrame(
                    ohlcv_k, 
                    columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"]
                )
                df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms")
                df.set_index("Timestamp", inplace=True)
        except Exception as ex:
            print(f"Kraken API Fetch Exception: {ex}")
            df = pd.DataFrame()

    # ==========================================
    # Tier 3: CoinGecko REST Fallback (Resilient Session)
    # ==========================================
    if df is None or df.empty or len(df) < 30:
        print(f"CCXT feeds offline. Attempting CoinGecko REST Fallback for {base_asset}...")
        try:
            coin_map = {
                "BTC": "bitcoin",
                "ETH": "ethereum",
                "SOL": "solana",
                "BNB": "binancecoin",
                "XRP": "ripple",
            }
            cg_id = coin_map.get(base_asset.upper(), "bitcoin")
            url = f"https://api.coingecko.com/api/v3/coins/{cg_id}/market_chart?vs_currency=usd&days=365&interval=daily"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            
            session = get_resilient_session()
            res = session.get(url, headers=headers, timeout=12)

            if res.status_code == 200:
                data = res.json()
                prices = data.get("prices", [])
                if len(prices) > 30:
                    df = pd.DataFrame(prices, columns=["Timestamp", "Close"])
                    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms")
                    df.set_index("Timestamp", inplace=True)
        except Exception as cg_err:
            print(f"CoinGecko API Fetch Exception: {cg_err}")
            df = pd.DataFrame()

    # ==========================================
    # Final Validation & Downcasting Optimization
    # ==========================================
    if df.empty or "Close" not in df.columns or len(df) < 30:
        st.error(f"Error: Live market feeds for {symbol} are temporarily unreachable. Please retry in a few moments.")
        return pd.DataFrame()

    # Downcast floats to float32 to enforce strict RAM ceiling
    df["Close"] = df["Close"].astype("float32")
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1)).astype("float32")
    df.dropna(subset=["Log_Return"], inplace=True)

    return df
