import ccxt
import pandas as pd


def get_bybit_testnet_client(api_key: str, api_secret: str):
    """Initializes a CCXT Bybit client set to Testnet mode."""
    exchange = ccxt.bybit({
        'apiKey': api_key,
        'secret': api_secret,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'spot',  # or 'linear' for USDT Futures
        }
    })
    # Enable Bybit Testnet URL endpoints
    exchange.set_sandbox_mode(True)
    return exchange


def execute_bybit_testnet_order(api_key: str, api_secret: str, symbol: str, amount_usd: float, current_price: float):
    """
    Executes a risk-sized market order on Bybit Testnet.
    Calculates exact contract quantity based on parametric VaR allocation.
    """
    try:
        client = get_bybit_testnet_client(api_key, api_secret)
        
        # Normalize symbol for CCXT (e.g., BTC-USD -> BTC/USDT)
        base_asset = symbol.split("-")[0]
        ccxt_symbol = f"{base_asset}/USDT"
        
        # Calculate quantity
        units_to_buy = amount_usd / current_price
        
        # Execute Spot Market Buy Order
        order = client.create_order(
            symbol=ccxt_symbol,
            type='market',
            side='buy',
            amount=units_to_buy
        )
        
        return {
            "status": "SUCCESS",
            "order_id": order.get("id"),
            "filled_units": units_to_buy,
            "cost_usd": amount_usd,
            "symbol": ccxt_symbol
        }
    except Exception as e:
        return {
            "status": "ERROR",
            "message": str(e)
        }
