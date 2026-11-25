from abc import ABC, abstractmethod

class BrokerInterface(ABC):
    """
    Abstract broker interface defining standard institutional exchange operations.
    Enforces consistent method signatures across multi-exchange expansions (Phase 6/7).
    """

    @abstractmethod
    def fetch_balances(self) -> list:
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
