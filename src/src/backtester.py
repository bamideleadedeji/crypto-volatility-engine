import numpy as np
import pandas as pd

def run_var_backtest(
    df: pd.DataFrame, 
    initial_capital: float = 10000.0, 
    confidence_level: float = 0.95, 
    max_risk_pct: float = 0.02,
    lookback_window: int = 30
) -> dict:
    """
    Backtests a Parametric VaR Dynamic Position Sizing Strategy against Buy & Hold.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing 'Close' prices indexed by timestamp.
        initial_capital (float): Starting portfolio capital in USD ($10,000 default).
        confidence_level (float): VaR confidence level (e.g., 0.95 for 95% VaR).
        max_risk_pct (float): Maximum allowed portfolio risk exposure per day (e.g., 2%).
        lookback_window (int): Rolling window in days for volatility estimation.
        
    Returns:
        dict: Performance summary metrics, breach analysis, and full equity curve DataFrame.
    """
    data = df.copy()
    if 'Close' not in data.columns:
        raise ValueError("DataFrame must contain a 'Close' price column.")
        
    # Calculate daily log returns
    data['Return'] = np.log(data['Close'] / data['Close'].shift(1))
    
    # Calculate rolling daily volatility (standard deviation of returns)
    data['Rolling_Vol'] = data['Return'].rolling(window=lookback_window).std()
    
    # Parametric Z-Score (1-tailed normal distribution quantile)
    from scipy.stats import norm
    z_score = norm.ppf(confidence_level)
    
    # 1-Day Parametric VaR (%)
    data['VaR_Pct'] = data['Rolling_Vol'] * z_score
    
    # Position Sizing: Target allocation (USD) scaled to maintain daily risk within max_risk_pct
    # Allocation = (Target Dollar Risk) / VaR_Pct
    target_risk_usd = initial_capital * max_risk_pct
    data['Position_Size_USD'] = np.minimum(
        initial_capital, 
        np.where(data['VaR_Pct'] > 0, target_risk_usd / data['VaR_Pct'], initial_capital)
    )
    
    # Exposure weight (Cap weight at 1.0 / 100% unleveraged)
    data['Weight'] = np.clip(data['Position_Size_USD'] / initial_capital, 0.0, 1.0)
    data['Weight'] = data['Weight'].shift(1).fillna(0) # Prevent lookahead bias
    
    # Strategy Return
    data['Strategy_Return'] = data['Weight'] * data['Return']
    
    # Equity Curves
    data['Buy_Hold_Equity'] = initial_capital * np.exp(data['Return'].fillna(0).cumsum())
    data['Strategy_Equity'] = initial_capital * np.exp(data['Strategy_Return'].fillna(0).cumsum())
    
    # Drawdowns
    data['Strategy_Peak'] = data['Strategy_Equity'].cummax()
    data['Strategy_Drawdown'] = (data['Strategy_Equity'] - data['Strategy_Peak']) / data['Strategy_Peak']
    
    data['BH_Peak'] = data['Buy_Hold_Equity'].cummax()
    data['BH_Drawdown'] = (data['Buy_Hold_Equity'] - data['BH_Peak']) / data['BH_Peak']
    
    # VaR Breaches (Exceedances where actual daily loss exceeded predicted VaR)
    data['VaR_Breach'] = data['Return'] < -data['VaR_Pct'].shift(1)
    
    # Clean output dropna
    clean_data = data.dropna().copy()
    
    # Performance Metrics Calculation
    tot_days = len(clean_data)
    if tot_days == 0:
        return {"status": "FAILED", "message": "Insufficient data window for backtesting."}
        
    ann_factor = 365 # Crypto markets run 24/7/365
    
    strat_total_return = (clean_data['Strategy_Equity'].iloc[-1] - initial_capital) / initial_capital
    bh_total_return = (clean_data['Buy_Hold_Equity'].iloc[-1] - initial_capital) / initial_capital
    
    strat_ann_return = (1 + strat_total_return) ** (ann_factor / tot_days) - 1
    bh_ann_return = (1 + bh_total_return) ** (ann_factor / tot_days) - 1
    
    strat_vol = clean_data['Strategy_Return'].std() * np.sqrt(ann_factor)
    bh_vol = clean_data['Return'].std() * np.sqrt(ann_factor)
    
    risk_free_rate = 0.04 # 4% baseline risk-free rate assumption
    
    sharpe_strat = (strat_ann_return - risk_free_rate) / strat_vol if strat_vol > 0 else 0
    sharpe_bh = (bh_ann_return - risk_free_rate) / bh_vol if bh_vol > 0 else 0
    
    downside_returns = clean_data[clean_data['Strategy_Return'] < 0]['Strategy_Return']
    downside_std = downside_returns.std() * np.sqrt(ann_factor) if len(downside_returns) > 0 else strat_vol
    sortino_strat = (strat_ann_return - risk_free_rate) / downside_std if downside_std > 0 else 0
    
    max_dd_strat = clean_data['Strategy_Drawdown'].min()
    max_dd_bh = clean_data['BH_Drawdown'].min()
    
    breach_count = int(clean_data['VaR_Breach'].sum())
    breach_rate = breach_count / tot_days
    expected_breach_rate = 1.0 - confidence_level
    
    metrics = {
        "status": "SUCCESS",
        "initial_capital": initial_capital,
        "final_strategy_equity": round(clean_data['Strategy_Equity'].iloc[-1], 2),
        "final_bh_equity": round(clean_data['Buy_Hold_Equity'].iloc[-1], 2),
        "strategy_total_return_pct": round(strat_total_return * 100, 2),
        "buy_hold_total_return_pct": round(bh_total_return * 100, 2),
        "strategy_sharpe": round(sharpe_strat, 2),
        "buy_hold_sharpe": round(sharpe_bh, 2),
        "strategy_sortino": round(sortino_strat, 2),
        "strategy_max_drawdown_pct": round(max_dd_strat * 100, 2),
        "buy_hold_max_drawdown_pct": round(max_dd_bh * 100, 2),
        "var_breaches": breach_count,
        "total_periods_tested": tot_days,
        "actual_breach_rate_pct": round(breach_rate * 100, 2),
        "expected_breach_rate_pct": round(expected_breach_rate * 100, 2),
        "backtest_df": clean_data
    }
    
    return metrics
