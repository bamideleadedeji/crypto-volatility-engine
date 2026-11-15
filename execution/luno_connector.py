"""
Institutional Crypto Investment Engine (ICIE) - Luno API Connector
Author: Bamidele Adedeji, Principal Consultant (Bamidele & Associates)
Enforces strict exchange authentication, BID/ASK order side mapping, decimal precision,
free + reserved wallet tracking, order-book spread inspection, and secure order dispatch.
"""

import logging
from decimal import Decimal
import requests

logger = logging.getLogger("LunoConnector")


class LunoConnector:
    def __init__(self, api_key_id: str, api_key_secret: str):
        if not api_key_id or not api_key_secret:
            raise ValueError("CRITICAL: Luno API credentials must be non-empty strings.")
        self.api_key_id = api_key_id
        self.api_key_secret = api_key_secret
        self.base_url = "https://luno.com"

    def get_wallet_breakdown(self) -> dict:
        """
        Fetches live wallet balances from Luno /balance endpoint.
        Separates 'free' (available to trade) from 'reserved' (locked in open orders)
        while computing 'total' (free + reserved) for exact portfolio valuation.
        """
        endpoint = f"{self.base_url}/balance"
        try:
            response = requests.get(endpoint, auth=(self.api_key_id, self.api_key_secret), timeout=15)
            if response.status_code != 200:
                raise RuntimeError(f"Luno API Error {response.status_code}: {response.text}")

            data = response.json()
            balances = data.get('balance', [])

            breakdown = {}
            for item in balances:
                asset = item.get('asset')
                if asset in ['XBT', 'ETH', 'SOL', 'NGN']:
                    free = Decimal(str(item.get('balance', '0')))
                    reserved = Decimal(str(item.get('reserved', '0')))
                    breakdown[asset] = {
                        'free': free,
                        'reserved': reserved,
                        'total': free + reserved
                    }

            for required in ['XBT', 'ETH', 'SOL', 'NGN']:
                if required not in breakdown:
                    breakdown[required] = {'free': Decimal('0'), 'reserved': Decimal('0'), 'total': Decimal('0')}

            return breakdown
        except Exception as e:
            logger.error(f"Failed to fetch Luno wallet breakdown: {str(e)}")
            raise

    def get_balances(self) -> dict:
        """
        Returns total quantities (free + reserved) per asset for valuation compatibility.
        """
        breakdown = self.get_wallet_breakdown()
        return {asset: info['total'] for asset, info in breakdown.items()}

    def get_market_tickers(self) -> dict:
        """
        Fetches live spot last_trade, best bid, best ask, and spread percentage for XBTNGN, ETHNGN, SOLNGN.
        """
        pairs = {
            'XBT': 'XBTNGN',
            'ETH': 'ETHNGN',
            'SOL': 'SOLNGN'
        }

        tickers = {}
        try:
            for coin, pair in pairs.items():
                endpoint = f"{self.base_url}/ticker?pair={pair}"
                response = requests.get(endpoint, timeout=15)
                if response.status_code != 200:
                    raise RuntimeError(f"Ticker HTTP {response.status_code} for {pair}: {response.text}")

                ticker_data = response.json()
                last_trade = Decimal(str(ticker_data.get('last_trade', '0')))
                bid = Decimal(str(ticker_data.get('bid', '0')))
                ask = Decimal(str(ticker_data.get('ask', '0')))
                spread_pct = ((ask - bid) / ask) if ask > 0 else Decimal('0')

                tickers[coin] = {
                    'pair': pair,
                    'last_trade': last_trade,
                    'bid': bid,
                    'ask': ask,
                    'spread_pct': spread_pct
                }
            return tickers
        except Exception as e:
            logger.error(f"Failed to fetch live market tickers: {str(e)}")
            raise

    def get_spot_prices(self) -> dict:
        """
        Fetches current spot last_trade prices for XBT, ETH, and SOL.
        """
        tickers = self.get_market_tickers()
        return {coin: data['last_trade'] for coin, data in tickers.items()}

    def get_portfolio_valuation(self) -> dict:
        """
        Computes atomic total portfolio capital combining exact wallet quantities (free + reserved)
        with live spot prices and order-book bid/ask telemetry.
        """
        breakdown = self.get_wallet_breakdown()
        tickers = self.get_market_tickers()

        balances = {asset: info['total'] for asset, info in breakdown.items()}
        free_balances = {asset: info['free'] for asset, info in breakdown.items()}
        reserved_balances = {asset: info['reserved'] for asset, info in breakdown.items()}
        prices = {coin: data['last_trade'] for coin, data in tickers.items()}

        asset_values = {}
        crypto_subtotal = Decimal('0')

        for coin in ['XBT', 'ETH', 'SOL']:
            qty = balances.get(coin, Decimal('0'))
            price = prices.get(coin, Decimal('0'))
            if price <= 0:
                raise RuntimeError(f"CRITICAL: Invalid zero/negative spot price returned for {coin}.")
            val = qty * price
            asset_values[coin] = val
            crypto_subtotal += val

        ngn_total = balances.get('NGN', Decimal('0'))
        ngn_free = free_balances.get('NGN', Decimal('0'))
        total_portfolio = crypto_subtotal + ngn_total

        return {
            'total_portfolio': total_portfolio,
            'crypto_subtotal': crypto_subtotal,
            'ngn_reserve': ngn_free,
            'ngn_total': ngn_total,
            'wallet_quantities': balances,
            'free_quantities': free_balances,
            'reserved_quantities': reserved_balances,
            'spot_prices': prices,
            'market_tickers': tickers,
            'asset_values': asset_values
        }

    def get_order_status(self, order_id: str) -> dict:
        """
        Queries Luno /orders/{id} to verify exact execution fill status and cleared volumes.
        """
        endpoint = f"{self.base_url}/orders/{order_id}"
        try:
            response = requests.get(endpoint, auth=(self.api_key_id, self.api_key_secret), timeout=15)
            if response.status_code != 200:
                return {"success": False, "status_code": response.status_code, "error": response.text}
            return {"success": True, "data": response.json()}
        except Exception as e:
            logger.error(f"Failed to query order status for {order_id}: {str(e)}")
            return {"success": False, "error": str(e)}

    def post_order(self, pair: str, side: str, volume: Decimal, price: Decimal) -> dict:
        """
        Dispatches a limit order to Luno /postorder endpoint.
        Maps 'BUY' -> 'BID' and 'SELL' -> 'ASK' per Luno REST API specification.
        """
        normalized_side = side.strip().upper()
        if normalized_side in ["BUY", "BID"]:
            luno_order_type = "BID"
        elif normalized_side in ["SELL", "ASK"]:
            luno_order_type = "ASK"
        else:
            return {"success": False, "error": f"Invalid order side '{side}'. Must be BUY/BID or SELL/ASK."}

        endpoint = f"{self.base_url}/postorder"
        payload = {
            'pair': pair.strip().upper(),
            'type': luno_order_type,
            'volume': str(volume),
            'price': str(price)
        }
        try:
            response = requests.post(endpoint, data=payload, auth=(self.api_key_id, self.api_key_secret), timeout=15)
            if response.status_code != 200:
                logger.error(f"Luno Order Rejection {response.status_code} ({pair} {luno_order_type}): {response.text}")
                return {"success": False, "status_code": response.status_code, "error": response.text}

            res_data = response.json()
            if "error" in res_data:
                logger.error(f"Luno API returned logical error on {pair}: {res_data}")
                return {"success": False, "error": res_data.get("error"), "data": res_data}

            logger.info(f"Luno order accepted on {pair} ({luno_order_type}): {res_data}")
            return {"success": True, "data": res_data}
        except Exception as e:
            logger.error(f"Exception during Luno order dispatch: {str(e)}")
            return {"success": False, "error": str(e)}
