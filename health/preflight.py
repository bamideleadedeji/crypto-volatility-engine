"""
Institutional Crypto Investment Engine (ICIE) - Startup Pre-Flight Health & Schema Validator
Performs 7-point operational readiness checks and defensive portfolio schema validation.
"""

import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from decimal import Decimal
import requests


def run_preflight_checks(
    config: dict | None,
    luno_key: str,
    luno_secret: str,
    data_dir: str,
    trade_log_path: str,
) -> list[dict]:
    """Executes 7-point operational health diagnostics before rendering or trading."""
    checks = []

    # 1. Configuration loaded
    cfg_ok = isinstance(config, dict) and len(config) > 0
    checks.append({"Check": "1. Configuration Loaded", "Healthy": cfg_ok, "Detail": "Active" if cfg_ok else "Fallback defaults"})

    # 2. API credentials present
    creds_ok = bool(luno_key and luno_secret)
    checks.append({"Check": "2. Luno API Credentials", "Healthy": creds_ok, "Detail": "Loaded from Secrets/Env" if creds_ok else "Missing Key/Secret"})

    # 3. Luno public/exchange API reachable
    try:
        luno_host = ".".join(["api", "luno", "com"])
        r = requests.get(f"https://{luno_host}/api/1/ticker", params={"pair": "XBTNGN"}, timeout=8)
        api_ok = r.status_code == 200
        api_msg = f"HTTP {r.status_code} (Reachable)"
    except Exception as exc:
        api_ok = False
        api_msg = f"Unreachable ({exc})"
    checks.append({"Check": "3. Luno Exchange Reachable", "Healthy": api_ok, "Detail": api_msg})

    # 4. Data directory writable
    try:
        os.makedirs(data_dir, exist_ok=True)
        fd, probe = tempfile.mkstemp(prefix=".probe_", dir=data_dir)
        os.close(fd)
        os.remove(probe)
        dir_ok, dir_msg = True, f"Writable ({data_dir})"
    except Exception as exc:
        dir_ok, dir_msg = False, f"Write error ({exc})"
    checks.append({"Check": "4. Data Directory Writable", "Healthy": dir_ok, "Detail": dir_msg})

    # 5. SQLite writable
    try:
        db_probe = os.path.join(data_dir, "luno_institutional_engine.db")
        with sqlite3.connect(db_probe, timeout=10) as conn:
            conn.execute("SELECT 1")
        db_ok, db_msg = True, "SQLite Connected"
    except Exception as exc:
        db_ok, db_msg = False, f"SQLite error ({exc})"
    checks.append({"Check": "5. SQLite Persistence", "Healthy": db_ok, "Detail": db_msg})

    # 6. Trade log accessible
    log_parent = os.path.dirname(os.path.abspath(trade_log_path))
    log_ok = os.path.exists(trade_log_path) or os.access(log_parent, os.W_OK)
    checks.append({"Check": "6. Audit Trade Log", "Healthy": log_ok, "Detail": os.path.basename(trade_log_path) if log_ok else "Inaccessible"})

    # 7. System UTC clock sane (>= 2026)
    now_utc = datetime.now(timezone.utc)
    clock_ok = now_utc.year >= 2026
    checks.append({"Check": "7. UTC System Clock", "Healthy": clock_ok, "Detail": now_utc.strftime("%Y-%m-%d %H:%M UTC")})

    return checks


def validate_portfolio_state(state: dict | None, fallback: dict) -> tuple[dict, bool, str]:
    """Validates required keys and numeric integrity of connector portfolio state."""
    if not isinstance(state, dict):
        return fallback, False, "Portfolio payload is not a valid dictionary."
    required_keys = {"total_portfolio", "ngn_reserve", "wallet_quantities", "spot_prices", "asset_values"}
    missing = required_keys - set(state.keys())
    if missing:
        return fallback, False, f"Portfolio payload missing keys: {sorted(missing)}"
    try:
        tpv = Decimal(str(state.get("total_portfolio", "0")))
        if tpv < 0:
            return fallback, False, "Negative total_portfolio reported by connector."
    except Exception as exc:
        return fallback, False, f"Malformed total_portfolio numeric value ({exc})."
    return state, True, "Portfolio schema verified."
