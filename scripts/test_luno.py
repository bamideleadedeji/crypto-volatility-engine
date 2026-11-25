import os
import sys

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from execution.luno_connector import LunoConnector

def test_luno_connection():
    print("⚡ Initializing Luno Connector Test...")
    
    # Pull credentials from environment or input directly for testing
    api_key = os.environ.get("LUNO_API_KEY_ID")
    api_secret = os.environ.get("LUNO_API_KEY_SECRET")
    
    if not api_key or not api_secret:
        print("❌ Error: LUNO_API_KEY_ID and LUNO_API_KEY_SECRET environment variables not set.")
        return

    connector = LunoConnector(api_key, api_secret)

    # 1. Test Ticker Fetching
    pairs = ["XBTNGN", "ETHNGN", "SOLNGN"]
    print("\n--- Testing Ticker Endpoints ---")
    for pair in pairs:
        try:
            price = connector.fetch_ticker(pair)
            print(f"✅ {pair} Spot Price: ₦{price:,.2f}")
        except Exception as e:
            print(f"❌ Failed to fetch {pair}: {str(e)}")

    # 2. Test Balance Fetching
    print("\n--- Testing Wallet Balance Endpoints ---")
    try:
        balances = connector.fetch_balances()
        for wallet in balances:
            asset = wallet.get("asset")
            balance = float(wallet.get("balance", 0.0))
            if balance > 0:
                print(f"💰 Wallet [{asset}]: {balance:,.4f}")
    except Exception as e:
        print(f"❌ Failed to fetch balances: {str(e)}")

if __name__ == "__main__":
    test_luno_connection()
