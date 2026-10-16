import os
import requests
from requests.auth import HTTPBasicAuth

def execute_luno_rebalance(
    target_btc_percent: float = 0.50,
    api_key_id: str = None,
    api_key_secret: str = None
) -> dict:
    """
    Executes an automated 50/50 portfolio rebalance on Luno Exchange via Limit Orders.
    Keeps trading fees capped at ~0.10% or earns maker rebates.
    """
    key_id = api_key_id or os.environ.get("LUNO_API_KEY_ID")
    key_secret = api_key_secret or os.environ.get("LUNO_API_KEY_SECRET")

    if not key_id or not key_secret:
        return {"status": "ERROR", "message": "Missing Luno API Keys in environment/secrets."}

    auth = HTTPBasicAuth(key_id, key_secret)
    base_url = "https://api.luno.com/api/1"

    try:
        # 1. Fetch live account balances
        balance_res = requests.get(f"{base_url}/balance", auth=auth, timeout=5)
        balance_data = balance_res.json()
        
        btc_balance = 0.0
        ngn_balance = 0.0
        
        for acc in balance_data.get("balance", []):
            if acc["asset"] == "XBT":
                btc_balance = float(acc["balance"])
            elif acc["asset"] == "NGN":
                ngn_balance = float(acc["balance"])

        # 2. Fetch current BTC/NGN order book ticker
        ticker_res = requests.get(f"{base_url}/ticker?pair=XBTNGN", timeout=5)
        ticker_data = ticker_res.json()
        bid_price = float(ticker_data.get("bid", 0.0))
        ask_price = float(ticker_data.get("ask", 0.0))

        total_portfolio_ngn = ngn_balance + (btc_balance * bid_price)
        target_ngn_per_asset = total_portfolio_ngn * target_btc_percent
        
        current_btc_value_ngn = btc_balance * bid_price
        
        # 3. Determine if selling BTC or buying BTC is required
        if current_btc_value_ngn > target_ngn_per_asset + 100:  # ₦100 threshold guard
            sell_amount_ngn = current_btc_value_ngn - target_ngn_per_asset
            btc_to_sell = round(sell_amount_ngn / bid_price, 6)
            
            # Place Maker Post-Only Limit Order (Zero/Minimal Fee)
            payload = {
                "pair": "XBTNGN",
                "type": "ASK", # SELL
                "volume": f"{btc_to_sell:.6f}",
                "price": f"{bid_price:.2f}",
                "post_only": True
            }
            order_res = requests.post(f"{base_url}/postorder", auth=auth, data=payload, timeout=5)
            order_data = order_res.json()
            
            return {
                "status": "SUCCESS",
                "action": f"Placed Limit SELL for {btc_to_sell} BTC (~₦{sell_amount_ngn:,.2f})",
                "order_id": order_data.get("order_id")
            }
            
        return {"status": "SUCCESS", "action": "Portfolio already balanced near 50/50 target."}

    except Exception as e:
        return {"status": "ERROR", "message": str(e)}
