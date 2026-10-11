import pandas as pd
import numpy as np
import ccxt
import streamlit as st


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_crypto_data(symbol="BTC-USD", period="2y"):
    """
    Fetches real-time, institutional-grade market data from live exchange APIs:
    Primary: Binance Public Spot API (Cloud-friendly REST Endpoint)
    Secondary: Kraken Public Spot API (Cloud-friendly Fallback)
    
    Cached for 30 minutes (ttl=1800) to optimize bandwidth and maintain sub-second response times.
    """
    # Map symbols to standard Spot format (e.g., BTC-USD -> BTC/USDT)
    ccxt_symbol = symbol.replace("-USD", "/USDT")
    if "/" not in ccxt_symbol:
        ccxt_symbol = f"{symbol}/USDT"

    df = pd.DataFrame()

    # Tier 1: Binance Public Spot API (No CloudFront IP Restrictions)
    try:
        exchange = ccxt.binance({
            "enableRateLimit": True,
            "timeout": 15000,
            "options": {"defaultType": "spot"}
        })
        
        # Fetch up to 730 daily candles (2 years of historical daily data)
        ohlcv = exchange.fetch_ohlcv(ccxt_symbol, timeframe="1d", limit=730)

        if ohlcv and len(ohlcv) > 0:
            df = pd.DataFrame(
                ohlcv, 
                columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"]
            )
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms")
            df.set_index("Timestamp", inplace=True)

    except Exception as e:
        print(f"Binance API fetch exception: {e}")
        df = pd.DataFrame()

    # Tier 2: Kraken Public API Fallback (Bypasses Cloud-Hosting Firewalls)
    if df is None or df.empty or len(df) < 30:
        print(f"Binance unavailable for {symbol}. Trying Kraken Fallback...")
        try:
            kraken_symbol = symbol.replace("-USD", "/USD")
            exchange_kraken = ccxt.kraken({
                "enableRateLimit": True,
                "timeout": 15000
            })
            ohlcv_kraken = exchange_kraken.fetch_ohlcv(kraken_symbol, timeframe="1d", limit=730)

            if ohlcv_kraken and len(ohlcv_kraken) > 0:
                df = pd.DataFrame(
                    ohlcv_kraken, 
                    columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"]
                )
                df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms")
                df.set_index("Timestamp", inplace=True)
        except Exception as ex:
            print(f"Kraken API fetch exception: {ex}")
            df = pd.DataFrame()

    # Validation: Ensure clean price dataset was retrieved
    if df.empty or "Close" not in df.columns or len(df) < 30:
        st.error(f"Error: Unable to retrieve live market data for {symbol} across public feeds.")
        return pd.DataFrame()

    # Compute Log Returns on Authentic Market Prices
    df["Close"] = df["Close"].astype(float)
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
    df.dropna(subset=["Log_Return"], inplace=True)

    return df
