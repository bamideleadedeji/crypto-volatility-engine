import time
from src.data_loader import fetch_crypto_data
from src.garch_model import fit_garch
from src.risk_engine import compute_var_and_position


def run_pipeline():
  print("=== Executing Crypto Volatility Engine Pipeline ===")
  # Fetch latest market data
  df = fetch_crypto_data(ticker="BTC-USD", period="1y")

  # Fit GARCH Model
  garch_vol = fit_garch(df["log_return"])
  latest_vol = garch_vol.iloc[-1]

  # Calculate Risk Metrics
  var_95, pos_size = compute_var_and_position(
      predicted_vol=latest_vol, max_risk_dollars=200.0
  )

  print(f"Latest Predicted Volatility: {latest_vol * 100:.2f}%")
  print(f"1-Day 95% VaR:               {var_95 * 100:.2f}%")
  print(f"Recommended Position Size:   ${pos_size:,.2f}")


if __name__ == "__main__":
  run_pipeline()
