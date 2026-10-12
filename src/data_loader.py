import pandas as pd
import numpy as np
import requests
import streamlit as st
import time
from requests.adapters import HTTPAdapter
from urllib3.util import Retry


def get_resilient_session():
    """Configures an HTTP session with retry backoff and custom headers to bypass cloud IP blocks."""
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json"
    })
    return session


@st.cache_data(ttl=900, show_spinner=False)
def fetch_crypto_data(symbol="BTC-USD", period="1y"):
    """
    Fetches spot market data with robust fallbacks:
    Tier 1: CoinGecko API (Resilient Session)
    Tier 2: Yahoo Finance REST API Direct Fallback
    Tier 3: Coinbase Public REST API Fallback

    Optimized for Render Cloud IPs & 512MB RAM Limit.
    """
    df = pd.DataFrame()
    session = get_resilient_session()

    base_asset = symbol.split("-")[0] if "-" in symbol else symbol.replace("USD", "")
    
    # Mapping for CoinGecko
    coin_map = {
        "BTC": "bitcoin",
        "ETH": "ethereum",
        "SOL": "solana",
        "BNB": "binancecoin",
        "XRP": "ripple",
    }
    cg_id = coin_map.get(base_asset.upper(), "bitcoin")

    # ==========================================
    # Tier 1: CoinGecko Public Market Chart REST API
    # ==========================================
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{cg_id}/market_chart?vs_currency=usd&days=365&interval=daily"
        res = session.get(url, timeout=10)

        if res.status_code == 200:
            data = res.json()
            prices = data.get("prices", [])
            if len(prices) > 30:
                df = pd.DataFrame(prices, columns=["Timestamp", "Close"])
                df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms")
                df.set_index("Timestamp", inplace=True)
    except Exception as e:
        print(f"Tier 1 CoinGecko Fetch Exception: {e}")
        df = pd.DataFrame()

    # ==========================================
    # Tier 2: Yahoo Finance Direct REST Fallback
    # ==========================================
    if df is None or df.empty or len(df) < 30:
        print(f"Tier 1 failed. Attempting Tier 2 Yahoo Finance REST API for {symbol}...")
        try:
            yf_symbol = f"{base_asset}-USD"
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yf_symbol}?range=1y&interval=1d"
            res = session.get(url, timeout=10)
            
            if res.status_code == 200:
                json_data = res.json()
                result = json_data.get("chart", {}).get("result", [])
                if result:
                    timestamps = result[0].get("timestamp", [])
                    quote = result[0].get("indicators", {}).get("quote", [{}])[0]
                    closes = quote.get("close", [])
                    
                    if len(closes) > 30:
                        df = pd.DataFrame({"Timestamp": timestamps, "Close": closes})
                        df.dropna(subset=["Close"], inplace=True)
                        df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="s")
                        df.set_index("Timestamp", inplace=True)
        except Exception as ex:
            print(f"Tier 2 Yahoo Finance Exception: {ex}")
            df = pd.DataFrame()

    # ==========================================
    # Tier 3: Coinbase Direct Spot API Fallback
    # ==========================================
    if df is None or df.empty or len(df) < 30:
        print(f"Tier 2 failed. Attempting Tier 3 Coinbase Direct REST API for {base_asset}-USD...")
        try:
            cb_pair = f"{base_asset}-USD"
            url = f"https://api.exchange.coinbase.com/products/{cb_pair}/candles?granularity=86400"
            res = session.get(url, timeout=10)
            
            if res.status_code == 200:
                candles = res.json()
                if isinstance(candles, list) and len(candles) > 30:
                    df = pd.DataFrame(candles, columns=["time", "low", "high", "open", "close", "volume"])
                    df["Timestamp"] = pd.to_datetime(df["time"], unit="s")
                    df.rename(columns={"close": "Close"}, inplace=True)
                    df.set_index("Timestamp", inplace=True)
                    df.sort_index(inplace=True)
        except Exception as cb_err:
            print(f"Tier 3 Coinbase Exception: {cb_err}")
            df = pd.DataFrame()

    # ==========================================
    # Validation & Memory Optimization
    # ==========================================
    if df.empty or "Close" not in df.columns or len(df) < 30:
        st.error(f"Error: Unable to pull market data for {symbol}. Public exchange feeds blocked the cloud IP.")
        return pd.DataFrame()

    # Memory downcasting to float32
    df["Close"] = df["Close"].astype("float32")
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1)).astype("float32")
    df.dropna(subset=["Log_Return"], inplace=True)

    return df
