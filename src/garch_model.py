from arch import arch_model
import numpy as np
import pandas as pd


def fit_garch_model(
    returns: pd.Series, p=1, q=1
) -> tuple[arch_model, pd.Series]:
  """Fits a GARCH(p, q) model to log returns and returns forecasted daily volatility."""
  # Scale returns by 100 for numerical stability during optimization
  scaled_returns = returns * 100

  # Define GARCH(1,1) with Constant Mean
  model = arch_model(scaled_returns, vol='Garch', p=p, q=q, dist='normal')
  results = model.fit(disp='off')

  # Extract conditional volatility (and rescaled back)
  cond_volatility = results.conditional_volatility / 100

  return results, cond_volatility


if __name__ == '__main__':
  from data_loader import fetch_crypto_data

  df = fetch_crypto_data()
  results, vol = fit_garch_model(df['Log_Return'])

  print("\n=== GARCH Model Summary ===")
  print(results.summary())
