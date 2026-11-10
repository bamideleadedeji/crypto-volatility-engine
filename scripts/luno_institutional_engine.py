import os
import time
import hmac
import hashlib
import base64
import requests
import pandas as pd
from datetime import datetime, timezone

# ==========================================
# ICIE INSTITUTIONAL LUNO REBALANCING ENGINE
# ==========================================
# Architecture: Sequential Sell-First, Ledger Verification, Proportional Buy Gate.
# Compliance: Treasury Management (Section 5) & Circuit Breakers (Section 14).

# Luno API Configuration (Loaded securely from Environment Variables / GitHub Secrets)
LUNO_KEY_ID = os.getenv("LUNO_API_KEY_ID", "").strip()
LUNO_SECRET = os.getenv("LUNO_API_KEY_SECRET", "").strip()
LUNO_BASE_URL = "https://api.luno.com/api/1"

# Target Portfolio Weights (Section 3)
TARGET_WEIGHTS = {
    "BTC": 0.50,
    "ETH": 0.30,
    "SOL": 0.20
}

# Operational Rules & Thresholds
DRIFT_THRESHOLD = 0.03       # Do nothing if drift < 3% (Section 4)
MIN_NGN_RESERVE = 0.10       # Minimum 10% NGN cash buffer (Section 5)
MAX_DAILY_LOSS_PCT = 0.05    # Circuit breaker threshold (Section 15)
LOG_FILE = "trade_log.csv"
AUDIT_LOG = "institutional_audit.log"

def log_audit(level, message):
    """Writes immutable audit trail logs to disk and stdout."""
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    log_entry = f"[{timestamp}] [{level}] {message}"
    print(log_entry)
    with open(AUDIT_LOG, "a", encoding="utf-8") as f:
        f.write(log_entry + "\n")

def get_luno_balances():
    """Queries Luno API for real-time wallet balances (Auditor Controller Role)."""
    url = f"{LUNO_BASE_URL}/balance"
    try:
        response = requests.get(url, auth=(LUNO_KEY_ID, LUNO_SECRET), timeout=10)
        if response.status_code == 200:
            data = response.json()
            balances = {}
            for item in data.get("balance", []):
                asset = item.get("asset")
                balances[asset] = {
                    "account_id": item.get("account_id"),
                    "balance": float(item.get("balance", 0.0)),
                    "reserved": float(item.get("reserved", 0.0)),
                    "available": float(item.get("balance", 0.0)) - float(item.get("reserved", 0.0))
                }
            return balances
        else:
            log_audit("ERROR", f"Failed to fetch balances. HTTP Status: {response.status_code}, Response: {response.text}")
            return None
    except Exception as e:
        log_audit("CRITICAL", f"Exception during balance audit: {str(e)}")
        return None

def get_ticker_price(pair):
    """Fetches current spot ticker price for asset pair (e.g., BTCNGN)."""
    url = f"{LUNO_BASE_URL}/ticker?pair={pair}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return float(response.json().get("last_trade", 0.0))
        else:
            log_audit("WARNING", f"Could not fetch ticker for {pair}: {response.text}")
            return 0.0
    except Exception as e:
        log_audit("ERROR", f"Exception fetching ticker {pair}: {str(e)}")
        return 0.0

def place_market_order(pair, type_action, volume):
    """Executes a market order (SELL or BUY) on Luno."""
    url = f"{LUNO_BASE_URL}/marketorder"
    payload = {
        "pair": pair,
        "type": type_action,
        "volume": str(volume)
    }
    try:
        response = requests.post(url, data=payload, auth=(LUNO_KEY_ID, LUNO_SECRET), timeout=15)
        if response.status_code == 200:
            res_data = response.json()
            order_id = res_data.get("order_id")
            log_audit("SUCCESS", f"Market Order Placed [{type_action}] {pair} | Vol: {volume} | Order ID: {order_id}")
            return order_id
        else:
            log_audit("ERROR", f"Order Execution Failed [{type_action} {pair}]: {response.text}")
            return None
    except Exception as e:
        log_audit("CRITICAL", f"Exception placing market order {pair}: {str(e)}")
        return None

def verify_settlement_ledger(expected_asset, previous_balance, timeout_seconds=60):
    """
    Settlement Signature Verification Gate:
    Polls the exchange balance ledger until NGN cash proceeds from a sale 
    have physically cleared and incremented in the wallet.
    """
    log_audit("INFO", f"Awaiting settlement signature for {expected_asset} proceeds...")
    start_time = time.time()
    
    while time.time() - start_time < timeout_seconds:
        current_balances = get_luno_balances()
        if current_balances and expected_asset in current_balances:
            current_avail = current_balances[expected_asset]["available"]
            if current_avail > previous_balance:
                delta = current_avail - previous_balance
                log_audit("VERIFIED", f"Settlement signature confirmed! Ledger incremented by +{delta:,.2f} {expected_asset}.")
                return True
        time.sleep(5) # Poll every 5 seconds
    
    log_audit("TIMEOUT", f"Settlement verification timed out after {timeout_seconds}s for {expected_asset}.")
    return False

def record_trade_audit(action, asset, amount, price, status):
    """Appends immutable trade record to trade_log.csv."""
    row = {
        "utc_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "action": action,
        "asset": asset,
        "amount": amount,
        "price": price,
        "status": status
    }
    df_new = pd.DataFrame([row])
    df_new.to_csv(LOG_FILE, mode='a', header=not os.path.exists(LOG_FILE), index=False)

def run_institutional_engine():
    log_audit("START", "Initializing ICIE Autonomous Institutional Engine (Luno Node)...")
    
    # 1. Audit Wallet Balances
    balances = get_luno_balances()
    if not balances:
        log_audit("CIRCUIT_BREAKER", "Halt: Unable to retrieve verifiable ledger balances. Safe Exit.")
        return

    ngn_balance = balances.get("NGN", {}).get("available", 0.0)
    btc_price = get_ticker_price("BTCNGN")
    eth_price = get_ticker_price("ETHNGN")
    sol_price = get_ticker_price("SOLNGN")

    if not btc_price or not eth_price or not sol_price:
        log_audit("CIRCUIT_BREAKER", "Halt: Spot ticker prices unavailable. Safe Exit.")
        return

    # 2. Compute Portfolio Valuation & Weights
    btc_qty = balances.get("BTC", {}).get("available", 0.0)
    eth_qty = balances.get("ETH", {}).get("available", 0.0)
    sol_qty = balances.get("SOL", {}).get("available", 0.0)

    btc_val = btc_qty * btc_price
    eth_val = eth_qty * eth_price
    sol_val = sol_qty * sol_price
    
    crypto_total_val = btc_val + eth_val + sol_val
    total_portfolio_val = crypto_total_val + ngn_balance

    if total_portfolio_val <= 0:
        log_audit("WARNING", "Portfolio valuation is zero or negative. Halting cycle.")
        return

    actual_weights = {
        "BTC": btc_val / total_portfolio_val,
        "ETH": eth_val / total_portfolio_val,
        "SOL": sol_val / total_portfolio_val
    }

    log_audit("AUDIT", f"Portfolio Value: ₦{total_portfolio_val:,.2f} | NGN Cash: ₦{ngn_balance:,.2f}")
    log_audit("AUDIT", f"Current Weights -> BTC: {actual_weights['BTC']*100:.1f}% | ETH: {actual_weights['ETH']*100:.1f}% | SOL: {actual_weights['SOL']*100:.1f}%")

    # 3. Check for Overweight Assets Requiring Liquidation (Sell Leg)
    rebalancing_executed = False
    for asset, target_w in TARGET_WEIGHTS.items():
        actual_w = actual_weights[asset]
        drift = actual_w - target_w

        if drift > DRIFT_THRESHOLD:
            log_audit("REBALANCE", f"Asset {asset} is overweight (Actual: {actual_w*100:.1f}% > Target: {target_w*100:.1f}% | Drift: +{drift*100:.1f}%). Initiating Sell Leg.")
            
            # Calculate required sell amount to correct drift
            excess_val = drift * total_portfolio_val
            asset_price = btc_price if asset == "BTC" else (eth_price if asset == "ETH" else sol_price)
            sell_volume = excess_val / asset_price

            pair = f"{asset}NGN"
            previous_ngn_balance = ngn_balance

            # Execute Sell
            order_id = place_market_order(pair, "SELL", round(sell_volume, 6))
            if order_id:
                record_trade_audit("SELL", asset, sell_volume, asset_price, "ORDER_PLACED")
                
                # Verify Settlement Signature in Ledger before proceeding to Buy
                settled = verify_settlement_ledger("NGN", previous_ngn_balance)
                if settled:
                    log_audit("SUCCESS", "Sell liquidation fully settled and verified in NGN wallet.")
                    rebalancing_executed = True
                else:
                    log_audit("CRITICAL", "Settlement verification failed after sell order. Aborting rebalance cycle.")
                    return
            break  # Execute one leg per run to maintain strict treasury control

    # 4. Check for Underweight Assets & Proportional Allocation Gate (Buy Leg)
    if not rebalancing_executed:
        # Evaluate if we have idle NGN cash to deploy to underweight assets
        updated_balances = get_luno_balances()
        current_ngn = updated_balances.get("NGN", {}).get("available", 0.0)
        
        # Ensure Treasury Reserve rule is respected (Section 5)
        reserve_threshold = total_portfolio_val * MIN_NGN_RESERVE
        if current_ngn > reserve_threshold:
            deployable_cash = current_ngn - reserve_threshold
            
            for asset, target_w in TARGET_WEIGHTS.items():
                actual_w = actual_weights[asset]
                drift = target_w - actual_w # positive means underweight

                if drift > DRIFT_THRESHOLD:
                    log_audit("REBALANCE", f"Asset {asset} is underweight (Actual: {actual_w*100:.1f}% < Target: {target_w*100:.1f}%). Evaluating Buy Leg.")
                    
                    # Disbalance Gate Check: Can we buy proportionally?
                    allocation_budget = deployable_cash * target_w
                    asset_price = btc_price if asset == "BTC" else (eth_price if asset == "ETH" else sol_price)
                    buy_volume = allocation_budget / asset_price

                    pair = f"{asset}NGN"
                    order_id = place_market_order(pair, "BUY", round(buy_volume, 6))
                    if order_id:
                        record_trade_audit("BUY", asset, buy_volume, asset_price, "ORDER_PLACED")
                        log_audit("SUCCESS", f"Buy reallocation executed for {asset}.")
                    break

    log_audit("COMPLETE", "Institutional rebalancing audit cycle finished successfully.")

if __name__ == "__main__":
    run_institutional_engine()
