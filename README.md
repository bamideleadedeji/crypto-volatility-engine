#  Hybrid Crypto Volatility Forecasting & Dynamic Risk Engine

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance quantitative financial engineering framework that benchmarks classical econometrics against deep learning architectures for cryptocurrency volatility forecasting and automated risk management.

---

##  Executive Summary

Predicting time-series volatility in crypto markets is crucial for portfolio risk management, derivatives pricing, and dynamic position sizing. Traditional statistical models like **GARCH(1,1)** capture linear volatility clustering well but struggle with non-linear regime shifts. Modern deep learning architectures, such as **LSTM** and **Temporal Transformers**—excel at learning long-range sequence dependencies and multi-horizon market shocks.

This repository provides an end-to-end benchmarking engine that:
1. Fits and evaluates **GARCH(1,1)**, **LSTM**, and **Transformer** models on live digital asset return sequences.
2. Computes **1-Day 95% Parametric Value-at-Risk (VaR)** to quantify downside tail risk.
3. Translates volatility forecasts into **Volatility-Targeted Position Sizing Algorithms** to prevent capital drawdown during high-volatility regimes.
4. Renders an interactive **Streamlit Dashboard** for real-time risk inspection.

---

##  System Architecture

```text
 ┌─────────────────────────────────────────────────────────┐
 │                   1. DATA PIPELINE                      │
 │     Fetch OHLCV & Compute Log Returns (yfinance)        │
 └────────────────────────────┬────────────────────────────┘
                              │
 ┌────────────────────────────▼────────────────────────────┐
 │               2. MODEL BENCHMARK ENGINE                 │
 │  ┌──────────────────┬──────────────────┬─────────────┐  │
 │  │    GARCH(1,1)    │      LSTM        │ Transformer │  │
 │  │ (Econometric Vol)│ (Recurrent Net)  │ (Attention) │  │
 │  └──────────────────┴──────────────────┴─────────────┘  │
 └────────────────────────────┬────────────────────────────┘
                              │
 ┌────────────────────────────▼────────────────────────────┐
 │            3. QUANTITATIVE RISK ENGINE                  │
 │   Parametric VaR (95%) & Volatility-Targeted Sizing    │
 └────────────────────────────┬────────────────────────────┘
                              │
 ┌────────────────────────────▼────────────────────────────┐
 │                4. INTERACTIVE DASHBOARD                 │
 │          Streamlit + Plotly Visualization Engine        │
 └─────────────────────────────────────────────────────────┘
 Model Formulations1. Classical Econometric: GARCH(1,1)Modeled on scaled log returns $r_t = \ln(P_t / P_{t-1})$:$$\sigma_t^2 = \omega + \alpha \cdot \epsilon_{t-1}^2 + \beta \cdot \sigma_{t-1}^2$$Where $\omega > 0$ is the baseline variance, $\alpha$ captures ARCH reaction shocks, and $\beta$ accounts for GARCH persistence.2. Deep Learning: LSTM NetworkCaptures sequential dependencies across historical rolling windows using memory gates (input, forget, and output gates) to output daily variance proxies without linear constraints.3. Attention Architecture: Temporal TransformerApplies multi-head self-attention mechanisms to weigh historical return contexts simultaneously:$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$Identifies long-range dependency horizons and sudden volatility spikes across digital asset pairs.🛡️ Dynamic Position Sizing AlgorithmRather than using fixed position sizes, the Risk Engine scales allocation inversely with predicted 1-day conditional volatility:$$\text{1-Day 95\% VaR} = Z_{0.95} \times \hat{\sigma}_{t+1} \approx 1.645 \times \hat{\sigma}_{t+1}$$$$\text{Recommended Position (\$)} = \frac{\text{Capital} \times \text{Max Portfolio Risk \%}}{\text{1-Day 95\% VaR}}$$When predicted volatility increases, maximum allowed capital exposure is automatically dialed back to safeguard trading equity. Repository StructurePlaintextcrypto-volatility-engine/
├── app.py                 # Interactive Streamlit Web Application
├── main.py                # Command-line execution pipeline
├── requirements.txt       # Project dependencies
├── README.md              # Documentation
└── src/
    ├── __init__.py
    ├── data_loader.py     # Data retrieval & log-return generation
    ├── garch_model.py     # ARCH library implementation of GARCH(1,1)
    ├── lstm_model.py      # PyTorch LSTM sequence neural network
    ├── transformer.py     # PyTorch Multi-Head Attention Transformer
    └── risk_engine.py     # Parametric VaR & Position Sizing Engine
 Quickstart Guide1. Clone the RepositoryBashgit clone [https://github.com/bamideleadedeji/crypto-volatility-engine.git]
cd crypto-volatility-engine
2. Install DependenciesSet up a virtual environment and install the required Python packages:Bashpython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
Tech Stack & DependenciesLanguage: Python 3.9+Deep Learning: PyTorchEconometrics: archData Processing: pandas, numpy, scikit-learnFinancial Data Access: yfinanceVisualization & UI: Streamlit, Plotly Author & LicenseBamidele Adedeji — Senior Financial Consultant & Applied Data Scientist
