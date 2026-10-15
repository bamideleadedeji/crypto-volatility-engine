import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Tuple, Dict

def fit_gjr_garch(returns: pd.Series) -> Tuple[Dict[str, float], pd.Series]:
    """
    Fits a GJR-GARCH(1,1) asymmetric volatility model using Maximum Likelihood Estimation (MLE).
    Captures asymmetric leverage response to negative market shocks.
    """
    returns_clean = returns.dropna().values.astype(np.float32)
    n = len(returns_clean)
    index_dates = returns.dropna().index
    
    if n < 30:
        vol = np.full(n, np.std(returns_clean), dtype=np.float32)
        return {"omega": 0.0, "alpha": 0.0, "gamma": 0.0, "beta": 0.0, "persistence": 0.0}, pd.Series(vol, index=index_dates, dtype="float32")

    var_init = float(np.var(returns_clean))

    def negative_log_likelihood(params):
        omega, alpha, gamma, beta = params
        
        # Stationarity constraint: alpha + 0.5*gamma + beta < 1
        if alpha + 0.5 * gamma + beta >= 0.999:
            return 1e10
            
        sigma2 = np.zeros(n, dtype=np.float32)
        sigma2[0] = max(var_init, 1e-6)
        
        for t in range(1, n):
            r_prev = returns_clean[t-1]
            indicator = 1.0 if r_prev < 0 else 0.0
            sigma2[t] = omega + alpha * (r_prev**2) + gamma * indicator * (r_prev**2) + beta * sigma2[t-1]
            if sigma2[t] <= 1e-8:
                sigma2[t] = 1e-8
                
        ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + (returns_clean**2) / sigma2)
        return -ll

    initial_params = [var_init * 0.05, 0.05, 0.05, 0.85]
    bounds = [(1e-8, var_init * 0.5), (0.001, 0.4), (0.0, 0.5), (0.5, 0.98)]
    
    res = minimize(negative_log_likelihood, initial_params, bounds=bounds, method='L-BFGS-B')
    omega, alpha, gamma, beta = res.x
    
    # Reconstruct conditional volatility series
    sigma2 = np.zeros(n, dtype=np.float32)
    sigma2[0] = max(var_init, 1e-6)
    for t in range(1, n):
        r_prev = returns_clean[t-1]
        indicator = 1.0 if r_prev < 0 else 0.0
        sigma2[t] = omega + alpha * (r_prev**2) + gamma * indicator * (r_prev**2) + beta * sigma2[t-1]
        if sigma2[t] <= 1e-8:
            sigma2[t] = 1e-8
            
    cond_vol = np.sqrt(sigma2).astype(np.float32)
    vol_series = pd.Series(cond_vol, index=index_dates, dtype="float32")
    
    params_dict = {
        "omega": float(omega),
        "alpha": float(alpha),
        "gamma (leverage)": float(gamma),
        "beta": float(beta),
        "persistence": float(alpha + 0.5 * gamma + beta)
    }
    
    return params_dict, vol_series


def fit_egarch(returns: pd.Series) -> Tuple[Dict[str, float], pd.Series]:
    """
    Fits an Exponential GARCH (EGARCH(1,1)) model using Maximum Likelihood Estimation (MLE).
    Models logarithmic conditional variance to prevent non-negativity boundary violations.
    """
    returns_clean = returns.dropna().values.astype(np.float32)
    n = len(returns_clean)
    index_dates = returns.dropna().index
    
    if n < 30:
        vol = np.full(n, np.std(returns_clean), dtype=np.float32)
        return {"omega": 0.0, "alpha": 0.0, "gamma": 0.0, "beta": 0.0, "persistence": 0.0}, pd.Series(vol, index=index_dates, dtype="float32")

    var_init = float(np.var(returns_clean))
    log_var_init = np.log(max(var_init, 1e-6))
    c_norm = np.sqrt(2 / np.pi)

    def negative_log_likelihood(params):
        omega, alpha, gamma, beta = params
        if abs(beta) >= 0.999:
            return 1e10
            
        log_sigma2 = np.zeros(n, dtype=np.float32)
        log_sigma2[0] = log_var_init
        
        for t in range(1, n):
            sigma_prev = np.sqrt(np.exp(log_sigma2[t-1]))
            std_res = returns_clean[t-1] / max(sigma_prev, 1e-4)
            log_sigma2[t] = omega + alpha * (abs(std_res) - c_norm) + gamma * std_res + beta * log_sigma2[t-1]
            if log_sigma2[t] > 10.0:  # Clamp extreme upper bound
                log_sigma2[t] = 10.0
            elif log_sigma2[t] < -20.0:  # Clamp lower bound
                log_sigma2[t] = -20.0
                
        sigma2 = np.exp(log_sigma2)
        ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + (returns_clean**2) / sigma2)
        return -ll

    initial_params = [-0.1, 0.1, -0.05, 0.90]
    bounds = [(-5.0, 5.0), (0.001, 0.8), (-0.5, 0.5), (0.5, 0.98)]
    
    res = minimize(negative_log_likelihood, initial_params, bounds=bounds, method='L-BFGS-B')
    omega, alpha, gamma, beta = res.x
    
    # Reconstruct conditional volatility series
    log_sigma2 = np.zeros(n, dtype=np.float32)
    log_sigma2[0] = log_var_init
    for t in range(1, n):
        sigma_prev = np.sqrt(np.exp(log_sigma2[t-1]))
        std_res = returns_clean[t-1] / max(sigma_prev, 1e-4)
        log_sigma2[t] = omega + alpha * (abs(std_res) - c_norm) + gamma * std_res + beta * log_sigma2[t-1]
        
    cond_vol = np.sqrt(np.exp(log_sigma2)).astype(np.float32)
    vol_series = pd.Series(cond_vol, index=index_dates, dtype="float32")
    
    params_dict = {
        "omega": float(omega),
        "alpha": float(alpha),
        "gamma (leverage)": float(gamma),
        "beta": float(beta),
        "persistence": float(abs(beta))
    }
    
    return params_dict, vol_series
