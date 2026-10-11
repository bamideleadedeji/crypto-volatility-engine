import pandas as pd
import numpy as np
import ccxt
import streamlit as st


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_crypto_data(symbol="BTC-USD", period="2y"):
    """
    Fetches real-time market data directly from Bybit's Spot API via CCXT.
    Cached for 30 minutes (ttl=1800) to prevent rate limits while maintaining 
    authentic live price feeds.
    """
    # 1. Map symbols to Bybit Spot format (e.g., BTC-USD -> BTC/USDT)
    ccxt_symbol = symbol.replace("-USD", "/USDT")
    
    df = pd.DataFrame()
    
    # 2. Fetch Live Market Data from Bybit
    try:
        exchange = ccxt.bybit({
            "enableRateLimit": True,
            "timeout": 15000,
            "headers": {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
        })
        
        # Fetch up to 1000 daily candles directly from Bybit
        ohlcv = exchange.fetch_ohlcv(ccxt_symbol, timeframe="1d", limit=1000)

        if ohlcv and len(ohlcv) > 0:
            df = pd.DataFrame(
                ohlcv, 
                columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"]
            )
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms")
            df.set_index("Timestamp", inplace=True)
            
    except Exception as e:
        st.error(f"Bybit API Live Fetch Error for {ccxt_symbol}: {e}")
        return pd.DataFrame()

    # 3. Validate Data
    if df.empty or "Close" not in df.columns or len(df) < 30:
        st.warning(f"Could not retrieve sufficient live market data for {symbol} from Bybit.")
        return pd.DataFrame()

    # 4. Compute Log Returns on Live Prices
    df["Close"] = df["Close"].astype(float)
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
    df.dropna(subset=["Log_Return"], inplace=True)

    return df
