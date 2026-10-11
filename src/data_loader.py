import pandas as pd
import numpy as np
import yfinance as yf
import ccxt
import requests
import streamlit as st


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_crypto_data(symbol="BTC-USD", period="2y"):
    """
    Fetches real-time, institutional-grade market data from live exchange APIs:
    Tier 1: Yahoo Finance (Live Feed with MultiIndex column handling)
    Tier 2: Bybit Spot Public API via CCXT (Live Fallback)
    
    Cached for 30 minutes (ttl=1800) to prevent IP rate-limit bans while
    guaranteeing authentic market prices.
    """
    df = pd.DataFrame()

    # Tier 1: Yahoo Finance with Custom User-Agent & MultiIndex Flattening
    try:
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        })
        ticker = yf.Ticker(symbol, session=session)
        df = ticker.history(period=period)
        
        # FIX: Flatten yfinance MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
    except Exception as e:
        print(f"yfinance live fetch exception: {e}")
        df = pd.DataFrame()

    # Tier 2: Bybit Live Spot API (CCXT Fallback)
    if df is None or df.empty or len(df) < 30:
        print(f"yfinance empty for {symbol}. Fetching live market data from Bybit...")
        try:
            exchange = ccxt.bybit({
                "enableRateLimit": True,
                "timeout": 10000,
                "headers": {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
                }
            })
            ccxt_symbol = symbol.replace("-USD", "/USDT")
            ohlcv = exchange.fetch_ohlcv(ccxt_symbol, timeframe="1d", limit=365)

            if ohlcv and len(ohlcv) > 0:
                df = pd.DataFrame(
                    ohlcv, 
                    columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"]
                )
                df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms")
                df.set_index("Timestamp", inplace=True)
        except Exception as ex:
            print(f"Bybit CCXT live fetch exception: {ex}")
            df = pd.DataFrame()

    # Validation: Ensure live dataset returned cleanly
    if df is None or df.empty or "Close" not in df.columns:
        raise ValueError(f"Unable to establish live market data connection for {symbol}.")

    # Compute Log Returns on Live Prices
    df["Close"] = df["Close"].astype(float)
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
    df.dropna(subset=["Log_Return"], inplace=True)

    return df
