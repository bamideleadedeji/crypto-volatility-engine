"""
Institutional Crypto Investment Engine (ICIE) - Luno API Connector
Enforces strict exchange authentication, decimal precision, 
balance + reserved wallet aggregation, and secure order dispatch.
"""

import logging
from decimal import Decimal
import requests

logger = logging.getLogger("LunoConnector")

class LunoConnector:
    def __init__(self, api_key_id: str, api_key_secret: str):
        self.api_key_id = api_key_id
        self.api_key_secret = api_key_secret
        self.base_url = "https://api.luno.com/api/1"

    def get_balances(self):
        """
        Fetches live wallet balances from Luno /balance endpoint.
        Returns a dictionary mapping asset symbols to total quantities (balance + reserved).
        """
        endpoint = f"{self.base_url}/balance"
        try:
            response = requests.get(endpoint, auth=(self.api_key_id, self.api_key_secret), timeout=10)
            if response.status_code != 200:
                raise Exception(f"Luno API Error {response.status_code}: {response.text}")
            
            data = response.json()
            balances = data.get('balance', [])
            
            wallet_summary = {}
            for item in balances:
                asset = item.get('asset')
                # Institutional rule: sum free balance + reserved balance
                free = Decimal(str(item.get('balance', '0')))
                reserved = Decimal(str(item.get('reserved', '0')))
                total_qty = free + reserved
                
                if asset in ['XBT', 'ETH', 'SOL', 'NGN']:
                    wallet_summary[asset] = total_qty
                    
            return wallet_summary
        except Exception as e:
            logger.error(f"Failed to fetch Luno balances: {str(e)}")
            raise

    def get_spot_prices(self):
        """
        Fetches current spot ticker prices for XBTNGN, ETHNGN, and SOLNGN.
        """
        pairs = {
            'XBT': 'XBTNGN',
            'ETH': 'ETHNGN',
            'SOL': 'SOLNGN'
        }
        
        spot_prices = {}
        try:
            for coin, pair in pairs.items():
                endpoint = f"{self.base_url}/ticker?pair={pair}"
                response = requests.get(endpoint, timeout=10)
                if response.status_code == 200:
                    ticker_data = response.json()
                    spot_prices[coin] = Decimal(str(ticker_data.get('last_trade', '0')))
                else:
                    logger.warning(f"Failed to fetch ticker for {pair}: {response.status_code}")
                    spot_prices[coin] = Decimal('0')
            return spot_prices
        except Exception as e:
            logger.error(f"Failed to fetch spot prices: {str(e)}")
            raise

    def get_portfolio_valuation(self):
        """
        Computes total portfolio capital by combining exact wallet quantities 
        with live spot prices, matching the Luno mobile app benchmark.
        """
        balances = self.get_balances()
        prices = self.get_spot_prices()
        
        asset_values = {}
        crypto_subtotal = Decimal('0')
        
        for coin in ['XBT', 'ETH', 'SOL']:
            qty = balances.get(coin, Decimal('0'))
            price = prices.get(coin, Decimal('0'))
            val = qty * price
            asset_values[coin] = val
            crypto_subtotal += val
            
        ngn_reserve = balances.get('NGN', Decimal('0'))
        total_portfolio = crypto_subtotal + ngn_reserve
        
        return {
            'total_portfolio': total_portfolio,
            'crypto_subtotal': crypto_subtotal,
            'ngn_reserve': ngn_reserve,
            'wallet_quantities': balances,
            'spot_prices': prices,
            'asset_values': asset_values
        }

    def post_order(self, pair: str, side: str, volume: Decimal, price: Decimal):
        """
        Dispatches a limit order to Luno /postorder endpoint with strict error logging.
        """
        endpoint = f"{self.base_url}/postorder"
        payload = {
            'pair': pair,
            'type': side,
            'volume': str(volume),
            'price': str(price)
        }
        try:
            response = requests.post(endpoint, data=payload, auth=(self.api_key_id, self.api_key_secret), timeout=10)
            if response.status_code != 200:
                logger.error(f"Luno Order Rejection {response.status_code}: {response.text}")
                return {"success": False, "status_code": response.status_code, "error": response.text}
            
            res_data = response.json()
            logger.info(f"Order successfully placed on {pair} ({side}): {res_data}")
            return {"success": True, "data": res_data}
        except Exception as e:
            logger.error(f"Exception during Luno order dispatch: {str(e)}")
            return {"success": False, "error": str(e)}
