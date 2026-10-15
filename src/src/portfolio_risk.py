import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, List, Tuple

def fetch_multi_asset_returns(symbols: List[str], data_fetcher_func) -> pd.DataFrame:
    """
    Fetches market data for multiple crypto assets and aligns their daily log returns.
    """
    returns_dict = {}
    for sym in symbols:
        try:
            df = data_fetcher_func(sym)
            if df is not None and not df.empty and "Log_Return" in df.columns:
                returns_dict[sym] = df["Log_Return"].dropna()
        except Exception:
            continue
            
    if not returns_dict:
        return pd.DataFrame()
        
    # Combine log returns into a single aligned DataFrame
    returns_df = pd.DataFrame(returns_dict).dropna()
    return returns_df.astype('float32')


def calculate_portfolio_var(
    returns_df: pd.DataFrame,
    weights: Dict[str, float],
    portfolio_capital: float = 10000.0,
    confidence_level: float = 0.95,
    horizon_days: int = 1
) -> Dict:
    """
    Computes Parametric Multi-Asset Portfolio VaR using the Variance-Covariance Matrix.
    """
    if returns_df.empty or len(returns_df.columns) < 2:
        return {
            "status": "FAILED",
            "message": "Insufficient multi-asset data for covariance matrix computation."
        }
        
    assets = returns_df.columns.tolist()
    
    # Normalize weight vector to sum to 1.0
    weight_vector = np.array([weights.get(asset, 1.0 / len(assets)) for asset in assets], dtype=np.float32)
    weight_vector = weight_vector / np.sum(weight_vector)
    
    # Compute Variance-Covariance Matrix (Annualized / Daily)
    cov_matrix = returns_df.cov().values.astype(np.float32)
    corr_matrix = returns_df.corr()
    
    # Individual Daily Asset Volatilities
    individual_vols = np.sqrt(np.diag(cov_matrix))
    
    # Portfolio Variance and Volatility (w^T * Cov * w)
    portfolio_variance = np.dot(weight_vector.T, np.dot(cov_matrix, weight_vector))
    portfolio_daily_vol = np.sqrt(max(portfolio_variance, 1e-8))
    
    # Normal Z-Score for specified confidence level (e.g., 1.645 for 95%)
    z_score = float(norm.ppf(confidence_level))
    
    # Portfolio 1-Day Parametric VaR
    portfolio_var_pct = portfolio_daily_vol * z_score * np.sqrt(horizon_days)
    portfolio_var_usd = portfolio_capital * portfolio_var_pct
    
    # Standalone Individual VaRs (Sum without diversification benefits)
    individual_vars_usd = individual_vols * z_score * portfolio_capital * weight_vector
    undiversified_var_usd = float(np.sum(individual_vars_usd))
    
    # Diversification Benefit ($ Saved due to cross-asset correlation < 1.0)
    diversification_benefit_usd = float(undiversified_var_usd - portfolio_var_usd)
    diversification_pct = (diversification_benefit_usd / undiversified_var_usd * 100) if undiversified_var_usd > 0 else 0.0
    
    # Asset Allocation Breakdown Table
    allocation_breakdown = []
    for i, asset in enumerate(assets):
        alloc_usd = portfolio_capital * weight_vector[i]
        asset_var_usd = alloc_usd * individual_vols[i] * z_score
        allocation_breakdown.append({
            "Asset": asset,
            "Weight (%)": round(weight_vector[i] * 100, 2),
            "Allocation ($)": round(alloc_usd, 2),
            "Daily Volatility (%)": round(individual_vols[i] * 100, 2),
            "Individual VaR ($)": round(asset_var_usd, 2)
        })

    return {
        "status": "SUCCESS",
        "portfolio_capital": portfolio_capital,
        "confidence_level_pct": confidence_level * 100,
        "portfolio_daily_vol_pct": float(portfolio_daily_vol * 100),
        "portfolio_var_pct": float(portfolio_var_pct * 100),
        "portfolio_var_usd": float(portfolio_var_usd),
        "undiversified_var_usd": float(undiversified_var_usd),
        "diversification_benefit_usd": float(diversification_benefit_usd),
        "diversification_pct": float(diversification_pct),
        "cov_matrix_df": pd.DataFrame(cov_matrix, index=assets, columns=assets),
        "corr_matrix_df": corr_matrix,
        "allocation_breakdown": pd.DataFrame(allocation_breakdown),
        "weights_used": dict(zip(assets, weight_vector))
    }
