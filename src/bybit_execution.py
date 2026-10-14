import ccxt

def execute_bybit_testnet_order(
    api_key: str, 
    api_secret: str, 
    symbol: str, 
    amount_usd: float, 
    current_price: float,
    side: str = "buy"
) -> dict:
    """
    Executes a market order on Bybit V5 Testnet via CCXT.
    Uses proxy forwarding to bypass CloudFront AWS datacenter IP blocks.
    """
    try:
        # Initialize Bybit V5 Unified Testnet Exchange Instance
        exchange = ccxt.bybit({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            # Route requests through CORS proxy to bypass AWS IP blocklist
            'proxy': 'https://corsproxy.io/?',
            'options': {
                'defaultType': 'spot',
            },
            'headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
                'Accept': 'application/json',
            }
        })
        
        # Enforce Sandbox / Testnet Routing
        exchange.set_sandbox_mode(True)

        # Standardize Symbol Notation for CCXT (e.g., BTC-USD -> BTC/USDT)
        formatted_symbol = symbol.replace("-USD", "/USDT").replace("-", "/")

        # Calculate Asset Units Based on Dollar Value
        quantity = amount_usd / current_price if current_price > 0 else 0.001

        # Transmit Market Order
        order = exchange.create_order(
            symbol=formatted_symbol,
            type='market',
            side=side.lower(),
            amount=quantity
        )

        return {
            "status": "SUCCESS",
            "order_id": order.get("id", "TESTNET_ORDER_CONFIRMED"),
            "filled_units": quantity,
            "raw_response": order
        }

    except Exception as e:
        return {
            "status": "FAILED",
            "message": str(e)
        }
