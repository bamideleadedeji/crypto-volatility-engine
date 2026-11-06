import csv
from datetime import datetime
import os
import time
import requests
from requests.auth import HTTPBasicAuth

# ----------------------------------------------------------------------
# 1. Credentials & Configuration
# ----------------------------------------------------------------------
LUNO_KEY = os.environ.get("LUNO_API_KEY_ID")
LUNO_SECRET = os.environ.get("LUNO_API_KEY_SECRET")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

TARGET_WEIGHTS = {
    "XBTNGN": 0.50,
    "ETHNGN": 0.30,
    "SOLNGN": 0.20,
}

MIN_LUNO_ORDER_NGN = 50.0   # Luno exact exchange minimum order threshold (₦50)
MAX_TRADES_PER_RUN = 3      # Restrict to max 3 trade operations per cycle
AUDIT_LOG_FILE = "luno_live_pilot_audit.csv"

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
def log_audit_entry(venue, asset, price_str, amount_str, units_str, status):
    """Guarantees immediate CSV audit logging for every execution attempt (Success, Fail, or Hold)."""
    try:
        file_exists = os.path.isfile(AUDIT_LOG_FILE)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(AUDIT_LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Timestamp", "Venue", "Asset", "Price", "Amount", "Units", "Status"])
            writer.writerow([timestamp, venue, asset, price_str, amount_str, units_str, status])
        print(f"📝 Audit log updated for {asset}: {status}")
    except Exception as log_err:
        print(f"❌ Audit log write exception: {log_err}")

# ----------------------------------------------------------------------
# 4. Headless Luno Autonomous Rotation Bot Engine
# ----------------------------------------------------------------------
def execute_luno_autonomous_rotation():
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now_str}] Executing Luno Multi-Asset Autonomous Rotation Bot...")

    if not LUNO_KEY or not LUNO_SECRET:
        error_msg = "❌ Luno API credentials missing in environment variables. Execution aborted."
        print(error_msg)
        send_telegram_alert(error_msg)
        return

    auth = HTTPBasicAuth(LUNO_KEY, LUNO_SECRET)

    try:
        # A. Fetch Spot Tickers with graceful fallbacks
        btc_res = requests.get("https://api.luno.com/api/1/ticker?pair=XBTNGN", auth=auth, timeout=10)
        eth_res = requests.get("https://api.luno.com/api/1/ticker?pair=ETHNGN", auth=auth, timeout=10)
        sol_res = requests.get("https://api.luno.com/api/1/ticker?pair=SOLNGN", auth=auth, timeout=10)

        prices = {
            "XBTNGN": float(btc_res.json().get("last_trade", 0.0)) if btc_res.status_code == 200 else 110759188.0,
            "ETHNGN": float(eth_res.json().get("last_trade", 0.0)) if eth_res.status_code == 200 else 3325000.0,
            "SOLNGN": float(sol_res.json().get("last_trade", 0.0)) if sol_res.status_code == 200 else 300000.0,
        }

        # B. Fetch Wallet Balances
        bal_res = requests.get("https://api.luno.com/api/1/balance", auth=auth, timeout=10)
        bal_res.raise_for_status()
        balances = bal_res.json().get("balance", [])

        ngn_balance, btc_balance, eth_balance, sol_balance = 0.0, 0.0, 0.0, 0.0
        for b in balances:
            asset = b.get("asset")
            val = float(b.get("balance", 0.0))
            if asset == "NGN":
                ngn_balance = val
            elif asset in ["XBT", "BTC"]:
                btc_balance = val
            elif asset == "ETH":
                eth_balance = val
            elif asset == "SOL":
                sol_balance = val

        holdings_val = {
            "XBTNGN": btc_balance * prices["XBTNGN"],
            "ETHNGN": eth_balance * prices["ETHNGN"],
            "SOLNGN": sol_balance * prices["SOLNGN"],
        }

        total_crypto_val = sum(holdings_val.values())
        total_portfolio_val = ngn_balance + total_crypto_val

        print(f"• Liquid NGN Balance: ₦{ngn_balance:,.2f}")
        print(f"• Total Portfolio Value: ₦{total_portfolio_val:,.2f}")

        trades_executed = 0
        available_cash = ngn_balance
        action_summary = []

        # C. Step 1: Check for Overweight Assets to Trim (Sell Overweight)
        for asset, target_pct in TARGET_WEIGHTS.items():
            if trades_executed >= MAX_TRADES_PER_RUN:
                break

            target_val = total_portfolio_val * target_pct
            current_val = holdings_val.get(asset, 0.0)
            delta = target_val - current_val  # Negative means overweight

            if delta < -MIN_LUNO_ORDER_NGN:
                excess_val = abs(delta)
                spot_price = prices[asset]

                # Precise asset volume formatting to prevent ErrInvalidBaseVolume
                if "ETH" in asset:
                    sell_units = round(excess_val / spot_price, 4)
                    vol_format = f"{sell_units:.4f}"
                elif "XBT" in asset or "BTC" in asset:
                    sell_units = round(excess_val / spot_price, 8)
                    vol_format = f"{sell_units:.8f}"
                else:
                    sell_units = round(excess_val / spot_price, 4)
                    vol_format = f"{sell_units:.4f}"

                order_payload = {
                    "pair": asset,
                    "type": "SELL",
                    "volume": vol_format,
                }

                # Correct endpoint (/marketorder) and form-encoded data body
                order_res = requests.post(
                    "https://api.luno.com/api/1/marketorder",
                    auth=auth,
                    data=order_payload,
                    timeout=15,
                )

                try:
                    order_data = order_res.json()
                except Exception:
                    order_data = {"error": f"HTTP {order_res.status_code}: {order_res.text}"}

                if order_res.status_code == 200 and "order_id" in order_data:
                    order_id = order_data["order_id"]
                    status_str = f"SUCCESS [ID: {order_id}]"
                    trades_executed += 1
                    action_summary.append(f"🔄 CRON SELL {asset}: Trimmed ₦{excess_val:,.2f} ({vol_format} units)")
                else:
                    err_msg = order_data.get("error", order_res.text or str(order_data))
                    status_str = f"FAILED: {err_msg}"
                    action_summary.append(f"❌ CRON SELL {asset} FAILED: {err_msg}")

                log_audit_entry("Luno Cron Bot", asset, f"₦{spot_price:,.2f}", f"₦{excess_val:,.2f}", vol_format, status_str)
                available_cash += excess_val
                time.sleep(2)  # Rest Guard pause between orders

        # D. Step 2: Execute Underweight Buys (Buy Underweight)
        remaining_cash = available_cash
        for asset, target_pct in TARGET_WEIGHTS.items():
            if trades_executed >= MAX_TRADES_PER_RUN:
                break

            target_val = total_portfolio_val * target_pct
            current_val = holdings_val.get(asset, 0.0)
            delta = target_val - current_val  # Positive means underweight

            if delta > MIN_LUNO_ORDER_NGN:
                net_amount = min(delta * 0.995, remaining_cash)  # Account for fee buffer

                if net_amount >= MIN_LUNO_ORDER_NGN:
                    order_payload = {
                        "pair": asset,
                        "type": "BUY",
                        "counter_volume": f"{net_amount:.2f}",
                    }

                    # Correct endpoint (/marketorder) and form-encoded data body
                    order_res = requests.post(
                        "https://api.luno.com/api/1/marketorder",
                        auth=auth,
                        data=order_payload,
                        timeout=15,
                    )

                    try:
                        order_data = order_res.json()
                    except Exception:
                        order_data = {"error": f"HTTP {order_res.status_code}: {order_res.text}"}

                    if order_res.status_code == 200 and "order_id" in order_data:
                        order_id = order_data["order_id"]
                        status_str = f"SUCCESS [ID: {order_id}]"
                        trades_executed += 1
                        action_summary.append(f"✅ CRON BUY {asset}: Deployed ₦{net_amount:,.2f}")
                    else:
                        err_msg = order_data.get("error", order_res.text or str(order_data))
                        status_str = f"FAILED: {err_msg}"
                        action_summary.append(f"❌ CRON BUY {asset} FAILED: {err_msg}")

                    log_audit_entry("Luno Cron Bot", asset, f"₦{prices[asset]:,.2f}", f"₦{net_amount:,.2f}", "Calculated", status_str)
                    remaining_cash -= net_amount
                    time.sleep(2)  # Rest Guard pause between orders
                else:
                    print(f"⏳ {asset} deficit (₦{net_amount:,.2f}) is below Luno threshold.")

        # E. Dispatch Telegram Summary Notification
        if action_summary:
            summary_text = (
                f"🤖 *LUNO AUTONOMOUS CRON REPORT*\n\n"
                f"• *Timestamp:* `{now_str}`\n"
                f"• *Portfolio Value:* `₦{total_portfolio_val:,.2f}`\n"
                f"• *Trades Executed:* `{trades_executed}`\n\n" +
                "\n".join(action_summary)
            )
            send_telegram_alert(summary_text)
        else:
            print("✅ Portfolio is within precise 50/30/20 balance. No cron trades required.")
            log_audit_entry("Luno Cron Bot", "PORTFOLIO", "N/A", "N/A", "N/A", "HOLD_BALANCED")

    except Exception as cron_ex:
        exc_msg = f"🚨 *Luno Cron Bot Exception*\nDetails: `{str(cron_ex)}`"
        print(exc_msg)
        log_audit_entry("Luno Cron Bot", "SYSTEM", "N/A", "N/A", "N/A", f"EXCEPTION: {str(cron_ex)}")
        send_telegram_alert(exc_msg)

if __name__ == "__main__":
    execute_luno_autonomous_rotation()
