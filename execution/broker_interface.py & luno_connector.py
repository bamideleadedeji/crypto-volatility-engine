from abc import ABC, abstractmethod
import requests
from requests.auth import HTTPBasicAuth

class BrokerInterface(ABC):
    @abstractmethod
    def fetch_balances(self) -> dict:
        pass

    @abstractmethod
    def fetch_ticker(self, pair: str) -> float:
        pass

    @abstractmethod
    def place_market_order(self, pair: str, side: str, volume: str = None, counter_volume: str = None) -> dict:
        pass

    @abstractmethod
    def check_order_status(self, order_id: str) -> str:
        pass


class LunoConnector(BrokerInterface):
    """Institutional Luno exchange connector implementing the BrokerInterface."""
    
    def __init__(self, api_key: str, api_secret: str):
        self.auth = HTTPBasicAuth(api_key, api_secret)
        self.base_url = "https://api.luno.com/api/1"

    def fetch_balances(self) -> list:
        res = requests.get(f"{self.base_url}/balance", auth=self.auth, timeout=10)
        if res.status_code != 200:
            raise Exception(f"Luno Balance API Error: HTTP {res.status_code} - {res.text}")
        return res.json().get("balance", [])

    def fetch_ticker(self, pair: str) -> float:
        res = requests.get(f"{self.base_url}/ticker?pair={pair}", timeout=8)
        if res.status_code != 200:
            raise Exception(f"Failed to fetch ticker for {pair}: HTTP {res.status_code}")
        return float(res.json().get("last_trade", 0.0))

    def place_market_order(self, pair: str, side: str, volume: str = None, counter_volume: str = None) -> dict:
        payload = {"pair": pair, "type": side}
        if volume:
            payload["volume"] = volume
        if counter_volume:
            payload["counter_volume"] = counter_volume
            
        res = requests.post(f"{self.base_url}/marketorder", auth=self.auth, data=payload, timeout=10)
        if res.status_code == 200 and res.text.strip():
            return res.json()
        raise Exception(f"Luno Order Failed: HTTP {res.status_code} - {res.text}")

    def check_order_status(self, order_id: str) -> str:
        res = requests.get(f"{self.base_url}/orders/{order_id}", auth=self.auth, timeout=8)
        if res.status_code == 200:
            return res.json().get("state", "").upper()
        return "UNKNOWN"
