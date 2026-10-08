import numpy as np
import pandas as pd
import yfinance as yf


def fetch_crypto_data(
    symbol='BTC-USD', period='2y', interval='1d'
) -> pd.DataFrame:
  """Fetches historical OHLCV data from Yahoo Finance and computes daily log returns."""
  print(f'Fetching data for {symbol}...')
  df = yf.download(symbol, period=period, interval=interval, progress=False)

  # Handle MultiIndex columns if present
  if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

  # Calculate daily log returns: r_t = ln(P_t / P_{t-1})
  df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

  # Calculate annualized realized volatility over a 21-day rolling window
  df['Realized_Vol_21d'] = df['Log_Return'].rolling(window=21).std() * np.sqrt(
      365
  )

  df = df.dropna()
  return df


if __name__ == '__main__':
  data = fetch_crypto_data()
  print("Data head with Log Returns:")
  print(data[['Close', 'Log_Return', 'Realized_Vol_21d']].head())
