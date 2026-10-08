import numpy as np


def calculate_position_size(
    account_balance: float,
    predicted_volatility: float,
    max_portfolio_risk_pct=0.02,
    confidence_level=0.95,
) -> dict:
  """Calculates parametric Value at Risk (VaR) and scales trade position size

  inversely with predicted volatility.
  """
  # Z-score for 95% confidence (1.645)
  z_score = 1.645 if confidence_level == 0.95 else 2.326

  # 1-day Value at Risk (percentage)
  one_day_var_pct = z_score * predicted_volatility

  # Maximum dollar risk allowed
  max_dollar_risk = account_balance * max_portfolio_risk_pct

  # Volatility-Targeted Position Sizing: Position = Max Risk / VaR
  recommended_position_usd = max_dollar_risk / one_day_var_pct

  return {
      'predicted_daily_volatility': f'{predicted_volatility * 100:.2f}%',
      'one_day_95pct_VaR': f'{one_day_var_pct * 100:.2f}%',
      'max_allowed_risk_usd': f'${max_dollar_risk:,.2f}',
      'recommended_position_usd': f'${recommended_position_usd:,.2f}',
  }


if __name__ == '__main__':
  # Example: $10,000 account, predicted daily volatility of 4%
  risk_summary = calculate_position_size(
      account_balance=10000.0, predicted_volatility=0.04
  )
  print("\n=== RISK ENGINE OUTPUT ===")
  for key, val in risk_summary.items():
    print(f"{key}: {val}")
