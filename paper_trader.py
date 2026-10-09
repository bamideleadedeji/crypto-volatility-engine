import os
import ccxt
import pandas as pd
from dotenv import load_dotenv
from src.data_loader import fetch_crypto_data
from src.garch_model import fit_garch_model
from src.risk_engine import calculate_position_size

load_dotenv()

BYBIT_API_KEY = os.getenv('BYBIT_API_KEY')
BYBIT_API_SECRET = os.getenv('BYBIT_API_SECRET')

def run_forward_test():
    print('\n==================================================')
    print('  CRYPTO VOLATILITY & RISK ENGINE - FORWARD TEST  ')
    print('==================================================\n')
    
    # Initialize public exchange instance for ticker fetching
    public_exchange = ccxt.bybit({'enableRateLimit': True})
    
    symbol_yf = 'BTC-USD'
    symbol_bybit = 'BTC/USDT'
    
    print(f'[*] Fetching historical market data for {symbol_yf}...')
    df = fetch_crypto_data(symbol=symbol_yf, period='1y')
    
    print('[*] Fitting GARCH(1,1) Volatility Model...')
    _, garch_vol = fit_garch_model(df['Log_Return'])
    latest_vol = garch_vol.iloc[-1]
    
    print('[*] Calculating Value-at-Risk & Position Sizing...')
    risk_summary = calculate_position_size(account_balance=10000.0, predicted_volatility=latest_vol)
    
    print('[*] Fetching live price ticker from Bybit...')
    try:
        ticker_info = public_exchange.fetch_ticker(symbol_bybit)
        current_price = ticker_info['last']
    except Exception as e:
        print(f'[!] Bybit ticker fetch warning: {e}. Falling back to Yahoo Finance latest close.')
        current_price = df['Close'].iloc[-1]

    pos_usd_val = float(risk_summary['recommended_position_usd'].replace('$', '').replace(',', ''))
    target_btc_qty = round(pos_usd_val / current_price, 4)

    print('\n--- MODEL METRICS ---')
    print(f'Current BTC Price:        ${current_price:,.2f}')
    print(f'Predicted Volatility:     {risk_summary["predicted_daily_volatility"]}')
    print(f'1-Day 95% Parametric VaR: {risk_summary["one_day_95pct_VaR"]}')
    print(f'Target Position ($):      {risk_summary["recommended_position_usd"]}')
    print(f'Target Quantity (BTC):    {target_btc_qty} BTC')

    log_data = {
        'Timestamp': [pd.Timestamp.now()],
        'Asset': [symbol_yf],
        'Price': [current_price],
        'Daily_Vol': [risk_summary['predicted_daily_volatility']],
        'VaR_95': [risk_summary['one_day_95pct_VaR']],
        'Position_Dollars': [risk_summary['recommended_position_usd']],
        'BTC_Quantity': [target_btc_qty]
    }
    log_df = pd.DataFrame(log_data)
    log_file = 'forward_test_log.csv'

    if not os.path.exists(log_file):
        log_df.to_csv(log_file, index=False)
    else:
        log_df.to_csv(log_file, mode='a', header=False, index=False)
    print(f'\n[OK] Results logged successfully to "{log_file}"')

if __name__ == '__main__':
    run_forward_test()
