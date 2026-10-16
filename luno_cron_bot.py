import os
import requests
from requests.auth import HTTPBasicAuth
import pandas as pd
from datetime import datetime

# ----------------------------------------------------------------------
# 1. Credentials & Configuration
# ----------------------------------------------------------------------
LUNO_KEY = os.environ.get("LUNO_API_KEY_ID")
LUNO_SECRET = os.environ.get("LUNO_API_KEY_SECRET")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

PAIR = "XBTNGN"
SPLIT_RATIO = 0.50  # Hardcoded 50/50 Allocation Rule

# ----------------------------------------------------------------------
# 2. Telegram Alert Dispatcher
# ----------------------------------------------------------------------
def send_telegram_alert(message: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram credentials not found in environment.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        res = requests.post(url, json=payload, timeout=10)
        res.raise_for_status()
        print("✅ Telegram alert transmitted successfully.")
    except Exception as e:
        print(f"❌ Telegram transmission error: {e}")

# ----------------------------------------------------------------------
# 3. LUNO Execution Engine
# ----------------------------------------------------------------------
def execute_luno_5050_pilot():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Executing Luno 50/50 Automated Pilot...")
    
    if not LUNO_KEY or not LUNO_SECRET:
        error_msg = "❌ Luno API credentials missing. Execution aborted."
        print(error_msg)
        send_telegram_alert(error_msg)
        return

    auth = HTTPBasicAuth(LUNO_KEY, LUNO_SECRET)

    try:
        # A. Fetch NGN Balance
        bal_res = requests.get("https://api.luno.com/api/1/balance", auth=auth, timeout=10)
        bal_res.raise_for_status()
        balances = bal_res.json().get("balance", [])
        
        ngn_balance = 0.0
        for b in balances:
            if b.get("asset") == "NGN":
                ngn_balance = float(b.get("balance", 0.0))
                break

        print(f"• Current Luno NGN Balance: ₦{ngn_balance:,.2f}")

        if ngn_balance < 100.0:
            msg = f"⚠️ Luno NGN balance too low to rebalance: ₦{ngn_balance:,.2f}"
            print(msg)
            send_telegram_alert(msg)
            return

        # B. Calculate 50/50 Trade Allocation
        target_ngn_trade = round(ngn_balance * SPLIT_RATIO, 2)
        remaining_fiat_buffer = round(ngn_balance - target_ngn_trade, 2)

        # C. Post Immediate Market Buy Order to Luno
        order_payload = {
            "pair": PAIR,
            "type": "BUY",
            "counter_volume": f"{target_ngn_trade:.2f}"  # Spends exact NGN amount
        }
        
        order_res = requests.post("https://api.luno.com/api/1/postmarketorder", auth=auth, data=order_payload, timeout=10)
        order_data = order_res.json()

        if "order_id" in order_data:
            order_id = order_data["order_id"]
            
            telegram_msg = (
                f"🚨 *AUTOMATED LUNO 50/50 TRADE EXECUTED*\n\n"
                f"• *Timestamp:* `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`\n"
                f"• *Venue:* Luno REST API\n"
                f"• *Pair:* `{PAIR}`\n"
                f"• *Total Portfolio NGN:* `₦{ngn_balance:,.2f}`\n"
                f"• *Allocated Order Size (50%):* `₦{target_ngn_trade:,.2f}`\n"
                f"• *Fiat Cash Buffer:* `₦{remaining_fiat_buffer:,.2f}`\n"
                f"• *Luno Order ID:* `{order_id}`\n"
                f"• *Status:* SUCCESS ✅"
            )
            print(f"✅ Order Executed Successfully! ID: {order_id}")
            send_telegram_alert(telegram_msg)

        else:
            err_msg = order_data.get("error", str(order_data))
            fail_msg = f"❌ *Luno Order Placement Failed*\nReason: `{err_msg}`"
            print(fail_msg)
            send_telegram_alert(fail_msg)

    except Exception as ex:
        exc_msg = f"🚨 *Luno Automation Bot Exception*\nDetails: `{str(ex)}`"
        print(exc_msg)
        send_telegram_alert(exc_msg)

if __name__ == "__main__":
    execute_luno_5050_pilot()
