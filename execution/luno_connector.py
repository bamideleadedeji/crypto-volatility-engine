"""
Institutional Crypto Investment Engine (ICIE) - Luno API Connector
Enforces strict exchange authentication, BID/ASK order side mapping, decimal precision,
multi-wallet aggregation (Spot + Savings/Staking), order-book spread inspection, and order dispatch.
"""

import logging
import time
from decimal import Decimal
import requests

logger = logging.getLogger("LunoConnector")


class LunoConnector:
    def __init__(self, api_key_id: str, api_key_secret: str):
        if not api_key_id or not api_key_secret:
            raise ValueError("CRITICAL: Luno API credentials must be non-empty strings.")
        self.api_key_id = api_key_id.strip()
        self.api_key_secret = api_key_secret.strip()
        self.base_url = "https://" + "api." + "luno.com" + "/api/1"
        self.headers = {
            "Accept": "application/json",
            "User-Agent": "ICIE-Portfolio-Engine/1.0",
        }

    def _safe_json(self, response: requests.Response, context: str) -> dict:
        try:
            return response.json()
        except ValueError:
            preview = response.text[:200] if response.text else "<EMPTY BODY>"
            raise RuntimeError(f"Non-JSON response from Luno ({context}, HTTP {response.status_code}): {preview}")

    def get_wallet_breakdown(self) -> dict:
        endpoint = f"{self.base_url}/balance"
        response = requests.get(endpoint, auth=(self.api_key_id, self.api_key_secret), headers=self.headers, timeout=15)
        if response.status_code != 200:
            raise RuntimeError(f"Luno API Error {response.status_code}: {response.text[:300]}")

        data = self._safe_json(response, "balance")
        balances = data.get('balance', [])
        breakdown = {
            'XBT': {'free': Decimal('0'), 'reserved': Decimal('0'), 'total': Decimal('0')},
            'ETH': {'free': Decimal('0'), 'reserved': Decimal('0'), 'total': Decimal('0')},
            'SOL': {'free': Decimal('0'), 'reserved': Decimal('0'), 'total': Decimal('0')},
            'NGN': {'free': Decimal('0'), 'reserved': Decimal('0'), 'total': Decimal('0')}
        }
        for item in balances:
            asset = str(item.get('asset', '')).upper()
            free = Decimal(str(item.get('balance', '0')))
            reserved = Decimal(str(item.get('reserved', '0')))
            if asset not in breakdown:
                breakdown[asset] = {'free': Decimal('0'), 'reserved': Decimal('0'), 'total': Decimal('0')}
            breakdown[asset]['free'] += free
            breakdown[asset]['reserved'] += reserved
            breakdown[asset]['total'] += (free + reserved)

        return breakdown

    def get_balances(self) -> dict:
        breakdown = self.get_wallet_breakdown()
        return {asset: info['total'] for asset, info in breakdown.items()}

    def get_market_tickers(self) -> dict:
        pairs = {'XBT': 'XBTNGN', 'ETH': 'ETHNGN', 'SOL': 'SOLNGN'}
        tickers = {}
        for coin, pair in pairs.items():
            endpoint = f"{self.base_url}/ticker?pair={pair}"
            response = requests.get(endpoint, headers=self.headers, timeout=15)
            if response.status_code != 200:
                raise RuntimeError(f"Ticker HTTP {response.status_code} for {pair}: {response.text[:300]}")

            data = self._safe_json(response, f"ticker:{pair}")
            last_trade = Decimal(str(data.get('last_trade', '0')))
            bid = Decimal(str(data.get('bid', '0')))
            ask = Decimal(str(data.get('ask', '0')))
            spread_pct = ((ask - bid) / ask) if ask > 0 else Decimal('0')
            tickers[coin] = {'pair': pair, 'last_trade': last_trade, 'bid': bid, 'ask': ask, 'spread_pct': spread_pct}
        return tickers

    def get_pair_ticker(self, pair: str, max_retries: int = 3) -> dict:
        clean_pair = pair.strip().upper()
        endpoint = f"{self.base_url}/ticker?pair={clean_pair}"
        last_exc = None
        for attempt in range(max_retries):
            try:
                response = requests.get(endpoint, headers=self.headers, timeout=15)
                if response.status_code == 200:
                    data = self._safe_json(response, f"pair_ticker:{clean_pair}")
                    last_trade = Decimal(str(data.get('last_trade', '0')))
                    bid = Decimal(str(data.get('bid', '0')))
                    ask = Decimal(str(data.get('ask', '0')))
                    spread_pct = ((ask - bid) / ask) if ask > 0 else Decimal('0')
                    return {
                        'pair': clean_pair,
                        'last_trade': last_trade,
                        'bid': bid,
                        'ask': ask,
                        'spread_pct': spread_pct,
                    }
                if response.status_code in (429, 500, 502, 503, 504):
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"Ticker HTTP {response.status_code} for {clean_pair}: {response.text[:300]}")
            except requests.RequestException as exc:
                last_exc = exc
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Failed to fetch ticker for {clean_pair} after {max_retries} attempts: {last_exc}")

    def get_spot_prices(self) -> dict:
        tickers = self.get_market_tickers()
        return {coin: data['last_trade'] for coin, data in tickers.items()}

    def get_portfolio_valuation(self) -> dict:
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

        return {
            'total_portfolio': crypto_subtotal + ngn_total,
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
        endpoint = f"{self.base_url}/orders/{order_id}"
        try:
            response = requests.get(endpoint, auth=(self.api_key_id, self.api_key_secret), headers=self.headers, timeout=15)
            if response.status_code != 200:
                return {"success": False, "status_code": response.status_code, "error": response.text[:300]}
            return {"success": True, "data": self._safe_json(response, f"order_status:{order_id}")}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def post_order(self, pair: str, side: str, volume: Decimal, price: Decimal) -> dict:
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
            response = requests.post(endpoint, data=payload, auth=(self.api_key_id, self.api_key_secret), headers=self.headers, timeout=15)
            if response.status_code != 200:
                return {"success": False, "status_code": response.status_code, "error": response.text[:300]}
            res_data = self._safe_json(response, f"post_order:{pair}")
            if "error" in res_data:
                return {"success": False, "error": res_data.get("error"), "data": res_data}
            return {"success": True, "data": res_data}
        except Exception as e:
            return {"success": False, "error": str(e)}
