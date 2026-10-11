import pandas as pd
import numpy as np
import ccxt
import streamlit as st


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_crypto_data(symbol="BTC-USD", period="2y"):
    """
    Fetches real-time OHLCV market data directly from Bybit's Public Spot API.
    Configured specifically for cloud proxies (Render/Streamlit Cloud) with 
    explicit REST endpoints and timeout controls.
    """
    # 1. Map symbols to Bybit Spot format (BTC-USD -> BTC/USDT)
    ccxt_symbol = symbol.replace("-USD", "/USDT")
    if "/" not in ccxt_symbol:
        ccxt_symbol = f"{symbol}/USDT"

    df = pd.DataFrame()

    # 2. Configure CCXT Bybit with direct REST endpoint parameters
    try:
        exchange = ccxt.bybit({
            'enableRateLimit': True,
            'timeout': 20000,
            'options': {
                'defaultType': 'spot',
                'adjustForTimeDifference': True,
            },
            'headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
                'Accept': 'application/json',
            }
        })

        # Fetch up to 730 daily candles directly from Bybit
        ohlcv = exchange.fetch_ohlcv(ccxt_symbol, timeframe="1d", limit=730)

        if ohlcv and len(ohlcv) > 0:
            df = pd.DataFrame(
                ohlcv, 
                columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"]
            )
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms")
            df.set_index("Timestamp", inplace=True)

    except Exception as e:
        print(f"Bybit API Live Fetch Exception for {ccxt_symbol}: {e}")
        return pd.DataFrame()

    # 3. Validate Data Extraction
    if df.empty or "Close" not in df.columns or len(df) < 30:
        return pd.DataFrame()

    # 4. Clean Data & Compute Log Returns
    df["Close"] = df["Close"].astype(float)
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
    df.dropna(subset=["Log_Return"], inplace=True)

    return df
