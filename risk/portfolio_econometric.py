import numpy as np
import pandas as pd
from scipy.stats import t, norm
from sklearn.covariance import LedoitWolf

try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False


class PortfolioEconometricEngine:
    """
    Institutional econometric engine providing GARCH volatility forecasting,
    Ledoit-Wolf covariance shrinkage, and multi-asset portfolio VaR/CVaR analytics
    for the Institutional Crypto Investment Engine (ICIE).
    """

    def __init__(self, confidence_levels=[0.95, 0.975, 0.99]):
        self.confidence_levels = confidence_levels

    def fit_optimized_garch(self, returns_series: pd.Series, model_type: str = "GARCH", p: int = 1, q: int = 1):
        """
        Fits an optimized GARCH(p,q) or EGARCH model using Student's t-distribution
        to accurately account for leptokurtosis (fat tails) in cryptocurrency returns.
        """
        if not HAS_ARCH:
            raise ImportError("The 'arch' package is required for GARCH econometric modeling.")

        # Clean and scale returns for numerical optimizer stability (percentage scale)
        scaled_returns = returns_series.dropna() * 100.0

        vol_spec = "EGARCH" if model_type.upper() == "EGARCH" else "GARCH"
        
        try:
            am = arch_model(
                scaled_returns,
                mean='Zero',
                vol=vol_spec,
                p=p,
                q=q,
                dist='t'  # Student-t innovations for fat tails
            )
            res = am.fit(disp='off', show_warning=False)
            
            # Rescale conditional volatility back to decimal returns
            conditional_volatility = res.conditional_volatility / 100.0
            standardized_residuals = res.std_resid
            
            # Extract persistence (alpha + beta)
            alpha_params = [v for k, v in res.params.items() if 'alpha' in k]
            beta_params = [v for k, v in res.params.items() if 'beta' in k]
            persistence = sum(alpha_params) + sum(beta_params) if (alpha_params or beta_params) else 0.94
            
            return {
                "success": True,
                "model": res,
                "conditional_volatility": conditional_volatility,
                "standardized_residuals": standardized_residuals,
                "persistence": persistence,
                "params": res.params
            }
        except Exception as e:
            # Fallback to EWMA / rolling volatility if optimizer encounters convergence issues
            fallback_vol = returns_series.ewm(span=20).std().fillna(returns_series.std())
            return {
                "success": False,
                "error": str(e),
                "conditional_volatility": fallback_vol,
                "persistence": 0.94
            }

    def compute_robust_covariance(self, returns_df: pd.DataFrame, method: str = "ledoit_wolf") -> pd.DataFrame:
        """
        Computes a robust covariance matrix using Ledoit-Wolf shrinkage
        to prevent matrix singularity and estimation noise in multi-asset portfolios.
        """
        clean_df = returns_df.dropna()
        if clean_df.empty:
            return pd.DataFrame()

        if method.lower() == "ledoit_wolf":
            lw = LedoitWolf()
            cov_array = lw.fit(clean_df).covariance_
            cov_matrix = pd.DataFrame(cov_array, index=clean_df.columns, columns=clean_df.columns)
        else:
            cov_matrix = clean_df.cov()

        return cov_matrix

    def calculate_portfolio_var_cvar(self, weights: dict, returns_df: pd.DataFrame, covariance_matrix: pd.DataFrame, portfolio_value: float, horizon: int = 1):
        """
        Calculates Parametric and Student-t Portfolio VaR and Expected Shortfall (CVaR)
        across configured confidence levels.
        """
        tickers = list(weights.keys())
        w_vec = np.array([weights[t] for t in tickers])
        
        # Align subset covariance matrix
        sub_cov = covariance_matrix.loc[tickers, tickers].values
        
        # Portfolio Variance & Volatility (Daily)
        portfolio_variance = np.dot(w_vec.T, np.dot(sub_cov, w_vec))
        portfolio_daily_vol = np.sqrt(max(0.0, portfolio_variance))
        
        # Horizon scaling (Square root of time rule)
        horizon_vol = portfolio_daily_vol * np.sqrt(horizon)

        results = {}
        for conf in self.confidence_levels:
            # Parametric Gaussian VaR
            z_score = norm.ppf(conf)
            var_gaussian = portfolio_value * z_score * horizon_vol
            
            # Parametric Student-t VaR (assuming df=5 for cryptocurrency leptokurtosis)
            df = 5.0
            t_score = t.ppf(conf, df)
            scaling_factor = np.sqrt((df - 2) / df)
            var_student_t = portfolio_value * (t_score * scaling_factor) * horizon_vol

            # Expected Shortfall (CVaR) under normal distribution
            cvar_gaussian = portfolio_value * (norm.pdf(z_score) / (1 - conf)) * horizon_vol

            results[f"{int(conf*100)}%"] = {
                "Gaussian VaR (₦)": round(var_gaussian, 2),
                "Student-t VaR (₦)": round(var_student_t, 2),
                "Gaussian Expected Shortfall (₦)": round(cvar_gaussian, 2),
                "Daily Portfolio Volatility (%)": round(portfolio_daily_vol * 100, 4)
            }

        return results
