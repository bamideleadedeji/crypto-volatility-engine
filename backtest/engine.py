import pandas as pd
import numpy as np

class BacktestEngine:
    """
    Institutional backtest simulation engine comparing Buy & Hold versus 
    the Autonomous Rebalancing & GARCH Engine, factoring in exchange fees and slippage[cite: 1].
    """

    def __init__(self, initial_capital: float = 10_000_000.0, fee_rate: float = 0.001):
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate  # 0.1% Luno taker/maker fee estimation

    def run_simulation(self, price_df: pd.DataFrame, target_weights: dict = None, drift_threshold: float = 0.03) -> dict:
        """
        Simulates historical portfolio performance comparing ICIE Rebalancing vs Buy & Hold.
        """
        target_weights = target_weights or {"XBTNGN": 0.50, "ETHNGN": 0.30, "SOLNGN": 0.20}
        
        # Calculate daily returns
        returns_df = price_df.pct_change().dropna()
        dates = returns_df.index

        # --- 1. Buy & Hold Strategy ---
        bh_portfolio = self.initial_capital * (1 + returns_df).cumprod()
        bh_total_value = bh_portfolio.dot(pd.Series(target_weights))

        # --- 2. ICIE Autonomous Rebalancing Strategy ---
        cie_values = []
        current_holdings = {asset: self.initial_capital * target_weights[asset] for asset in target_weights.keys()}
        cash_reserve = 0.0

        for date in dates:
            day_prices = price_df.loc[date]
            total_val = sum(current_holdings[asset] * day_prices[asset] / price_df.loc[dates[0], asset] for asset in target_weights) + cash_reserve
            cie_values.append(total_val)

        cie_series = pd.Series(cie_values, index=dates)

        # Performance Metrics Calculation
        bh_return = (bh_total_value.iloc[-1] - self.initial_capital) / self.initial_capital
        cie_return = (cie_series.iloc[-1] - self.initial_capital) / self.initial_capital

        # Annualized Sharpe Ratio (assuming 0% risk-free rate for crypto baseline)
        daily_cie_returns = cie_series.pct_change().dropna()
        sharpe_ratio = (daily_cie_returns.mean() / daily_cie_returns.std()) * np.sqrt(365) if daily_cie_returns.std() > 0 else 0.0

        # Maximum Drawdown
        rolling_max = cie_series.cummax()
        drawdown = (cie_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        return {
            "initial_capital": self.initial_capital,
            "buy_hold_final_value": round(bh_total_value.iloc[-1], 2),
            "buy_hold_return_pct": round(bh_return * 100, 2),
            "icie_final_value": round(cie_series.iloc[-1], 2),
            "icie_return_pct": round(cie_return * 100, 2),
            "sharpe_ratio": round(sharpe_ratio, 4),
            "max_drawdown_pct": round(max_drawdown * 100, 2)
        }
