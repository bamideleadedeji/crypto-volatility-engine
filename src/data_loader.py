import pandas as pd
import numpy as np
import yfinance as yf
import ccxt

def fetch_crypto_data(symbol="BTC-USD", period="2y"):
    """
    Fetches historical crypto price data with fallback to CCXT (Bybit) 
    if Yahoo Finance blocks cloud server requests.
    """
    df = pd.DataFrame()
    
    # Try 1: yfinance
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
    except Exception as e:
        print(f"yfinance download exception: {e}")

    # Fallback to CCXT (Bybit Public Spot) if yfinance returned empty data
    if df.empty:
        print(f"yfinance returned empty data for {symbol}. Falling back to Bybit CCXT...")
        try:
            exchange = ccxt.bybit({"enableRateLimit": True})
            # Map ticker format (e.g. BTC-USD -> BTC/USDT)
            ccxt_symbol = symbol.replace("-USD", "/USDT")
            ohlcv = exchange.fetch_ohlcv(ccxt_symbol, timeframe="1d", limit=365)
            
            df = pd.DataFrame(
                ohlcv, 
                columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"]
            )
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms")
            df.set_index("Timestamp", inplace=True)
        except Exception as ex:
            print(f"CCXT fallback error: {ex}")

    if df.empty:
        raise ValueError(
            f"Failed to fetch market data for {symbol} from both Yahoo Finance and CCXT."
        )

    # Compute returns
    df["Close"] = df["Close"].astype(float)
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
    df.dropna(subset=["Log_Return"], inplace=True)
    
    return df
