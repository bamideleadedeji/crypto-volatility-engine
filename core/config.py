import json
import os

CONFIG_FILE = "config.json"

DEFAULT_CONFIG = {
    "target_weights": {"XBTNGN": 0.50, "ETHNGN": 0.30, "SOLNGN": 0.20},
    "drift_tolerance": 0.03,
    "max_failures": 5,
    "cooldown_minutes": 15,
    "min_order_ngn": 50.0,
    "max_trades_per_run": 4,
    "paper_mode": True,
    "taker_fee_pct": 0.0075,
    "max_single_asset_weight": 0.60,
    "max_daily_loss_pct": 0.05
}

def load_config() -> dict:
    if not os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "w") as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)
    try:
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return DEFAULT_CONFIG
