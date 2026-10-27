import os
import csv
import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime

# ----------------------------------------------------------------------
# 1. Credentials & Configuration
# ----------------------------------------------------------------------
LUNO_KEY = os.environ.get("LUNO_API_KEY_ID")
LUNO_SECRET = os.environ.get("LUNO_API_KEY_SECRET")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

PAIR = "XBTNGN"
TARGET_RATIO = 0.50          # 50/50 BTC/NGN target allocation
MIN_TRADE_NGN = 500.0       # Minimum NGN trade threshold (avoids invalid micro-orders)
AUDIT_LOG_FILE = "luno_live_pilot_audit.csv"  # Dedicated live audit trail file

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
# 3. Isolated Audit Logger
# ----------------------------------------------------------------------
def log_audit_entry(venue, pair, price_ngn, amount_ngn, units_btc, status, order_id="N/A"):
    file_exists = os.path.isfile(AUDIT_LOG_FILE)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    with open(AUDIT_LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Timestamp', 'Venue', 'Pair', 'Price_NGN', 'Amount_NGN', 'Units_BTC', 'Status', 'Order_ID'])
        writer.writerow([timestamp, venue, pair, f"₦{price_ngn:,.2f}", f"₦{amount_ngn:,.2f}", f"{units_btc:.6f}", status, order_id])

# ----------------------------------------------------------------------
# 4. LUNO Execution Engine
# ----------------------------------------------------------------------
def execute_luno_5050_pilot():
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{now_str}] Executing Luno 50/50 Automated Pilot Engine...")

    if not LUNO_KEY or not LUNO_SECRET:
        error_msg = "❌ Luno API credentials missing. Execution aborted."
        print(error_msg)
        send_telegram_alert(error_msg)
        return

    auth = HTTPBasicAuth(LUNO_KEY, LUNO_SECRET)

    try:
        # A. Fetch Balances (NGN & BTC)
        bal_res = requests.get("https://api.luno.com/api/1/balance", auth=auth, timeout=10)
        bal_res.raise_for_status()
        balances = bal_res.json().get("balance", [])

        ngn_balance = 0.0
        btc_balance = 0.0

        for b in balances:
            asset = b.get("asset")
            if asset == "NGN":
                ngn_balance = float(b.get("balance", 0.0))
            elif asset in ["XBT", "BTC"]:
                btc_balance = float(b.get("balance", 0.0))

        # B. Fetch Current Market Price
        ticker_res = requests.get(f"https://api.luno.com/api/1/ticker?pair={PAIR}", auth=auth, timeout=10)
        ticker_res.raise_for_status()
        btc_price = float(ticker_res.json().get("last_trade", 0.0))

        if btc_price <= 0:
            raise ValueError("Invalid BTC market price received from Luno ticker.")

        btc_value_ngn = btc_balance * btc_price
        total_portfolio_ngn = ngn_balance + btc_value_ngn
        target_btc_value = total_portfolio_ngn * TARGET_RATIO

        # Rebalance delta in NGN
        rebalance_delta_ngn = target_btc_value - btc_value_ngn

        print(f"• Luno NGN Balance: ₦{ngn_balance:,.2f}")
        print(f"• Luno BTC Balance: {btc_balance:.6f} BTC (₦{btc_value_ngn:,.2f})")
        print(f"• Total Portfolio Value: ₦{total_portfolio_ngn:,.2f}")
        print(f"• Rebalance Delta NGN: ₦{rebalance_delta_ngn:,.2f}")

        # C. Evaluate Minimum Order Threshold Guard
        if abs(rebalance_delta_ngn) < MIN_TRADE_NGN:
            msg = (
                f"ℹ️ *LUNO 50/50 PILOT STATUS: BALANCED*\n\n"
                f"• *Timestamp:* `{now_str}`\n"
                f"• *Total Portfolio Value:* `₦{total_portfolio_ngn:,.2f}`\n"
                f"• *NGN Buffer:* `₦{ngn_balance:,.2f}`\n"
                f"• *BTC Value:* `₦{btc_value_ngn:,.2f}`\n"
                f"• *Rebalance Delta:* `₦{rebalance_delta_ngn:,.2f}`\n"
                f"• *Action:* Portfolio is within target 50/50 tolerance (< ₦{MIN_TRADE_NGN:,.2f}). No trade required. ✅"
            )
            print("Portfolio in balance. Skipping trade.")
            log_audit_entry("Luno REST API", PAIR, btc_price, 0.0, 0.0, "HOLD_BALANCED")
            send_telegram_alert(msg)
            return

        # D. Execute Market Order with Query Parameter Formatting
        if rebalance_delta_ngn > 0:
            trade_amount_ngn = min(rebalance_delta_ngn, ngn_balance)
            order_payload = {
                "pair": PAIR,
                "type": "BUY",
                "counter_volume": f"{trade_amount_ngn:.2f}"
            }
            side_str = "BUY"
            expected_units = trade_amount_ngn / btc_price
        else:
            trade_units_btc = abs(rebalance_delta_ngn) / btc_price
            trade_units_btc = min(trade_units_btc, btc_balance)
            order_payload = {
                "pair": PAIR,
                "type": "SELL",
                "volume": f"{trade_units_btc:.6f}"
            }
            side_str = "SELL"
            trade_amount_ngn = trade_units_btc * btc_price
            expected_units = trade_units_btc

        # Uses params= for exact Luno query string encoding
        order_res = requests.post(
            "https://api.luno.com/api/1/postmarketorder",
            auth=auth,
            params=order_payload,
            timeout=15
        )
        
        try:
            order_data = order_res.json()
        except Exception:
            order_data = {"error": f"HTTP {order_res.status_code}: {order_res.text}"}

        if order_res.status_code == 200 and "order_id" in order_data:
            order_id = order_data["order_id"]
            
            telegram_msg = (
                f"🚨 *AUTOMATED LUNO 50/50 TRADE EXECUTED*\n\n"
                f"• *Timestamp:* `{now_str}`\n"
                f"• *Venue:* Luno REST API\n"
                f"• *Pair:* `{PAIR}` ({side_str})\n"
                f"• *Execution Price:* `₦{btc_price:,.2f}`\n"
                f"• *Trade Value:* `₦{trade_amount_ngn:,.2f}`\n"
                f"• *Total Portfolio:* `₦{total_portfolio_ngn:,.2f}`\n"
                f"• *Luno Order ID:* `{order_id}`\n"
                f"• *Status:* SUCCESS ✅"
            )
            print(f"✅ Order Executed Successfully! Order ID: {order_id}")
            log_audit_entry("Luno REST API", PAIR, btc_price, trade_amount_ngn, expected_units, "SUCCESS", order_id)
            send_telegram_alert(telegram_msg)

        else:
            err_msg = order_data.get("error", order_res.text or str(order_data))
            fail_msg = f"❌ *Luno Order Placement Failed*\nReason: `{err_msg}`"
            print(fail_msg)
            log_audit_entry("Luno REST API", PAIR, btc_price, trade_amount_ngn, expected_units, f"FAILED: {err_msg}")
            send_telegram_alert(fail_msg)

    except Exception as ex:
        exc_msg = f"🚨 *Luno Automation Bot Exception*\nDetails: `{str(ex)}`"
        print(exc_msg)
        send_telegram_alert(exc_msg)

if __name__ == "__main__":
    execute_luno_5050_pilot()
