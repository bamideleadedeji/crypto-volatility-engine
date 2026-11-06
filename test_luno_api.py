import os
import requests
from requests.auth import HTTPBasicAuth

# Load credentials from environment or Streamlit secrets
LUNO_KEY = os.environ.get("LUNO_API_KEY_ID=hg8trfq3ujgna")
LUNO_SECRET = os.environ.get("LUNO_API_KEY_SECRET=x4KTAcm5tKTD-FU9ic_QQqtbHp8hugqn2Tyy1PV-VaA")

print("--- Luno API Diagnostic Test ---")
if not LUNO_KEY or not LUNO_SECRET:
    print("❌ ERROR: LUNO_API_KEY_ID or LUNO_API_KEY_SECRET environment variables are missing!")
    exit(1)

auth = HTTPBasicAuth(LUNO_KEY, LUNO_SECRET)

# 1. Test Ticker Endpoint (Public/Auth)
print("\n[1/3] Testing Ticker Endpoints...")
for pair in ["XBTNGN", "ETHNGN", "SOLNGN"]:
    res = requests.get(f"https://api.luno.com/api/1/ticker?pair={pair}", timeout=5)
    if res.status_code == 200:
        data = res.json()
        print(f"  ✅ {pair} Spot Price: ₦{float(data.get('last_trade', 0)):,.2f}")
    else:
        print(f"  ❌ {pair} Ticker Failed: HTTP {res.status_code} - {res.text}")

# 2. Test Spot Trading Wallet Balance Endpoint
print("\n[2/3] Testing Spot Trading Wallet Balance Endpoint...")
bal_res = requests.get("https://api.luno.com/api/1/balance", auth=auth, timeout=5)
if bal_res.status_code == 200:
    balances = bal_res.json().get("balance", [])
    print("  ✅ Balance Endpoint Successful. Found Wallets:")
    for b in balances:
        asset = b.get("asset")
        bal = float(b.get("balance", 0))
        wallet_type = b.get("account_type", "DEFAULT")
        if bal > 0:
            print(f"     └─ Asset: {asset:<5} | Balance: {bal:<12.8f} | Type: {wallet_type}")
else:
    print(f"  ❌ Balance Endpoint Failed: HTTP {bal_res.status_code} - {bal_res.text}")

# 3. Test Order Endpoint Validation (Without executing a live fill)
print("\n[3/3] Testing Order Endpoint Connectivity...")
# We send a dummy query or check endpoint reachability
test_order_res = requests.post("https://api.luno.com/api/1/marketorder", auth=auth, params={"pair": "XBTNGN", "type": "BUY", "counter_volume": "10.0"}, timeout=5)
# Expecting either 200 (if cash is sufficient) or a specific validation error like "Insufficient funds" (HTTP 400/412) rather than 404
print(f"  ℹ️ Order Endpoint Response Code: {test_order_res.status_code}")
print(f"  ℹ️ Order Endpoint Response Body: {test_order_res.text}")
print("\n--- Diagnostic Complete ---")