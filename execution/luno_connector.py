import requests
from requests.auth import HTTPBasicAuth
from execution.broker_interface import BrokerInterface

class LunoConnector(BrokerInterface):
    """
    Institutional Luno exchange connector implementing the BrokerInterface.
    Equipped with robust timeout management, error handling, and response validation.
    """
    
    def __init__(self, api_key: str, api_secret: str):
        self.auth = HTTPBasicAuth(api_key, api_secret)
        self.base_url = "https://api.luno.com/api/1"

    def fetch_balances(self) -> list:
        """Fetches account balances across all currency wallets with error propagation."""
        try:
            res = requests.get(f"{self.base_url}/balance", auth=self.auth, timeout=10)
            if res.status_code != 200:
                raise Exception(f"Luno Balance API Error: HTTP {res.status_code} - {res.text}")
            return res.json().get("balance", [])
        except requests.exceptions.RequestException as e:
            raise Exception(f"Luno Balance Network Connection Failed: {str(e)}")

    def fetch_ticker(self, pair: str) -> float:
        """Fetches the latest spot trade price for a given trading pair (e.g., XBTNGN)."""
        try:
            res = requests.get(f"{self.base_url}/ticker?pair={pair}", timeout=8)
            if res.status_code != 200:
                raise Exception(f"Luno Ticker API Error for {pair}: HTTP {res.status_code}")
            return float(res.json().get("last_trade", 0.0))
        except requests.exceptions.RequestException as e:
            raise Exception(f"Luno Ticker Network Connection Failed for {pair}: {str(e)}")

    def place_market_order(self, pair: str, side: str, volume: str = None, counter_volume: str = None) -> dict:
        """
        Executes a market order (BUY/SELL) on Luno with strict validation.
        Enforces base_volume (for sells) or counter_volume (for buys) matching Luno API specs.
        """
        payload = {"pair": pair, "type": side.upper()}
        if volume:
            payload["base_volume"] = str(volume)  # Fixed: Luno expects base_volume, not volume
        if counter_volume:
            payload["counter_volume"] = str(counter_volume)
            
        try:
            res = requests.post(f"{self.base_url}/marketorder", auth=self.auth, data=payload, timeout=12)
            if res.status_code == 200 and res.text.strip():
                return res.json()
            raise Exception(f"Luno Order Execution Rejected: HTTP {res.status_code} - {res.text}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Luno Order Network Connection Failed: {str(e)}")

    def check_order_status(self, order_id: str) -> str:
        """Queries the status of a specific order ID by its unique identifier."""
        try:
            res = requests.get(f"{self.base_url}/orders/{order_id}", auth=self.auth, timeout=8)
            if res.status_code == 200:
                return res.json().get("state", "UNKNOWN").upper()
            return "UNKNOWN"
        except requests.exceptions.RequestException:
            return "CONNECTION_ERROR"
