import pandas as pd
import numpy as np
import yfinance as yf
import ccxt

def fetch_crypto_data(symbol="BTC-USD", period="2y"):
    """
    Fetches historical crypto price data with multi-tier fallback:
    1. Yahoo Finance (yfinance)
    2. CCXT / Bybit Spot API
    3. Synthetic Log-Return Generator (Guarantees zero app crashes)
    """
    df = pd.DataFrame()
    
    # Tier 1: Yahoo Finance
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
    except Exception as e:
        print(f"yfinance download exception: {e}")

    # Tier 2: CCXT / Bybit Public Spot API
    if df is None or df.empty:
        print(f"yfinance returned empty data for {symbol}. Trying CCXT Bybit...")
        try:
            exchange = ccxt.bybit({"enableRateLimit": True, "timeout": 10000})
            ccxt_symbol = symbol.replace("-USD", "/USDT")
            ohlcv = exchange.fetch_ohlcv(ccxt_symbol, timeframe="1d", limit=365)
            
            if ohlcv:
                df = pd.DataFrame(
                    ohlcv, 
                    columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"]
                )
                df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms")
                df.set_index("Timestamp", inplace=True)
        except Exception as ex:
            print(f"CCXT fallback exception: {ex}")

    # Tier 3: Synthetic Data Fallback (Prevents Streamlit Cloud rate-limit crashes)
    if df is None or df.empty or len(df) < 30:
        print(f"APIs unavailable. Generating synthetic market data for {symbol}...")
        dates = pd.date_range(end=pd.Timestamp.now(), periods=365, freq="D")
        np.random.seed(42)
        returns = np.random.normal(loc=0.001, scale=0.03, size=365)
        price_path = 60000.0 * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            "Close": price_path
        }, index=dates)

    # Compute Log Returns
    df["Close"] = df["Close"].astype(float)
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
    df.dropna(subset=["Log_Return"], inplace=True)
    
    return df
