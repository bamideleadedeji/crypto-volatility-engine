import ccxt
import pandas as pd


def get_bybit_testnet_client(api_key: str, api_secret: str):
    """Initializes a CCXT Bybit V5 REST client explicitly configured for Testnet sandbox."""
    exchange = ccxt.bybit({
        'apiKey': api_key,
        'secret': api_secret,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'spot',  # Unified Spot trading
        }
    })
    # Force sandbox endpoint URLs
    exchange.set_sandbox_mode(True)
    return exchange


def execute_bybit_testnet_order(
    api_key: str, 
    api_secret: str, 
    symbol: str, 
    amount_usd: float, 
    current_price: float,
    side: str = "buy"
) -> dict:
    """
    Executes a market order on Bybit V5 Testnet sized by the Parametric VaR Engine.
    
    Parameters:
        api_key (str): Bybit Testnet Key from Render Environment Secrets
        api_secret (str): Bybit Testnet Secret from Render Environment Secrets
        symbol (str): Asset symbol (e.g., 'BTC-USD', 'ETH-USD')
        amount_usd (float): Recommended dollar position size from VaR calculation
        current_price (float): Latest spot price
        side (str): 'buy' or 'sell'
        
    Returns:
        dict: Normalized order status and execution metadata
    """
    if not api_key or not api_secret:
        return {
            "status": "ERROR",
            "message": "Missing API key or secret in Render Environment Variables."
        }

    try:
        client = get_bybit_testnet_client(api_key, api_secret)

        # Normalize symbol for CCXT format (e.g., BTC-USD -> BTC/USDT)
        base_asset = symbol.split("-")[0] if "-" in symbol else symbol.replace("USD", "")
        ccxt_symbol = f"{base_asset}/USDT"

        # Calculate unit quantity
        units = round(amount_usd / current_price, 6)

        if units <= 0:
            return {
                "status": "ERROR",
                "message": "Calculated unit order size is too small."
            }

        # Submit Market Order to Bybit Testnet Sandbox
        order = client.create_order(
            symbol=ccxt_symbol,
            type='market',
            side=side,
            amount=units
        )

        return {
            "status": "SUCCESS",
            "order_id": order.get("id", "N/A"),
            "filled_units": units,
            "cost_usd": amount_usd,
            "symbol": ccxt_symbol,
            "raw_response": order
        }

    except ccxt.BaseError as ccxt_err:
        return {
            "status": "ERROR",
            "message": f"CCXT Bybit API Error: {str(ccxt_err)}"
        }
    except Exception as err:
        return {
            "status": "ERROR",
            "message": f"Execution Exception: {str(err)}"
        }
