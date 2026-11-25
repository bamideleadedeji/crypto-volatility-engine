#!/usr/bin/env python3
"""
ICIE Weekly Sunday Excel Audit Reporter (Gmail Delivery)
Builds a 4-sheet institutional Luno portfolio, GARCH(1,1)/VaR regime, High-Water Mark,
and trade execution Excel workbook (.xlsx) and emails it via Gmail SMTP.
"""

import io
import os
import sys
import json
import smtplib
from datetime import datetime, timezone
from decimal import Decimal
from email.message import EmailMessage
import pandas as pd

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from core.config import load_config
from core.logging import setup_logger
from execution.luno_connector import LunoConnector
from risk.volatility_gate import VolatilityRegimeGate
from strategy.rebalancer import TARGET_WEIGHTS

logger = setup_logger("ICIEWeeklyAudit")


def generate_weekly_excel_bytes(connector: LunoConnector, gate: VolatilityRegimeGate, config: dict) -> tuple[bytes, dict]:
    """Queries live Luno balances, HWM state, GARCH/VaR regimes, and trade_log.csv into a 4-sheet .xlsx workbook."""
    state = connector.get_portfolio_valuation()
    tpv = float(state.get("total_portfolio", Decimal("0")))
    ngn_free = float(state.get("ngn_reserve", Decimal("0")))
    balances = state.get("wallet_quantities", {})
    prices = state.get("spot_prices", {})
    asset_vals = state.get("asset_values", {})

    data_dir = os.environ.get("ICIE_DATA_DIR") or config.get("data_dir") or os.path.join(repo_root, "data")
    hwm_path = os.path.join(data_dir, "hwm_state.json")
    peak_hwm = max(tpv, float(os.environ.get("ICIE_SEED_PEAK", config.get("seed_peak", 59848.50))))
    if os.path.exists(hwm_path):
        try:
            with open(hwm_path, "r", encoding="utf-8") as f:
                peak_hwm = max(peak_hwm, float(json.load(f).get("peak_value", peak_hwm)))
        except Exception:
            pass
    dd_pct = ((peak_hwm - tpv) / peak_hwm * 100.0) if (peak_hwm > 0 and tpv > 0) else 0.0

    regimes = {coin: gate.evaluate_asset_regime(coin) for coin in ["XBT", "ETH", "SOL"]}
    holdings_rows = []
    for sym in ["XBT", "ETH", "SOL"]:
        qty = float(balances.get(sym, Decimal("0")))
        px = float(prices.get(sym, Decimal("0")))
        val = float(asset_vals.get(sym, Decimal("0")))
        wt = (val / tpv * 100.0) if tpv > 0 else 0.0
        tgt = float(TARGET_WEIGHTS.get(sym, 0.0)) * 100.0
        reg = regimes.get(sym, {})
        holdings_rows.append({
            "Asset": sym,
            "Quantity": qty,
            "Spot_Price_NGN": round(px, 2),
            "Value_NGN": round(val, 2),
            "Actual_Weight_Pct": round(wt, 2),
            "Target_Weight_Pct": round(tgt, 2),
            "Drift_Pct": round(wt - tgt, 2),
            "GARCH_Ann_Vol_Pct": round(float(reg.get("ann_vol", 0.0)) * 100.0, 2),
            "CF_VaR_24h_Pct": round(float(reg.get("cf_var_24h", 0.0)) * 100.0, 2),
            "Vol_Scalar": reg.get("vol_scalar", 1.0),
            "Regime_Status": reg.get("regime", "UNKNOWN"),
        })

    holdings_df = pd.DataFrame(holdings_rows)
    regime_df = pd.DataFrame(list(regimes.values()))
    log_path = os.path.join(repo_root, "trade_log.csv")
    log_df = pd.read_csv(log_path) if os.path.exists(log_path) else pd.DataFrame()

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    summary_df = pd.DataFrame([
        {"Metric": "Weekly Audit Timestamp (UTC)", "Value": now_utc},
        {"Metric": "Total Portfolio Capital (NGN)", "Value": f"₦{tpv:,.2f}"},
        {"Metric": "Cleared Free NGN Reserve", "Value": f"₦{ngn_free:,.2f}"},
        {"Metric": "Peak High-Water Mark (NGN)", "Value": f"₦{peak_hwm:,.2f}"},
        {"Metric": "Active Drawdown (%)", "Value": f"{dd_pct:.2f}% (10.00% Max Ceiling)"},
        {"Metric": "Total Logged Executions / Holds", "Value": len(log_df)},
    ])

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="01_Executive_Summary", index=False)
        holdings_df.to_excel(writer, sheet_name="02_Live_Luno_Holdings", index=False)
        regime_df.to_excel(writer, sheet_name="03_GARCH_VaR_Regimes", index=False)
        if not log_df.empty:
            log_df.tail(150).to_excel(writer, sheet_name="04_Trade_Audit_Log", index=False)

    kpis = {"tpv": tpv, "ngn_free": ngn_free, "peak_hwm": peak_hwm, "dd_pct": dd_pct, "now_utc": now_utc}
    return buf.getvalue(), kpis
def send_weekly_audit_email():
    luno_key = os.environ.get("LUNO_API_KEY_ID", "").strip()
    luno_secret = os.environ.get("LUNO_API_KEY_SECRET", "").strip()
    gmail_sender = os.environ.get("GMAIL_SENDER", "").strip()
    gmail_pass = os.environ.get("GMAIL_APP_PASSWORD", "").strip()
    gmail_recipient = os.environ.get("GMAIL_RECIPIENT", gmail_sender).strip()

    if not luno_key or not luno_secret:
        logger.error("Missing LUNO_API_KEY_ID or LUNO_API_KEY_SECRET. Aborting weekly audit.")
        sys.exit(1)
    if not gmail_sender or not gmail_pass:
        logger.error("Missing GMAIL_SENDER or GMAIL_APP_PASSWORD in GitHub Secrets. Aborting weekly audit.")
        sys.exit(1)

    config = load_config()
    connector = LunoConnector(luno_key, luno_secret)
    gate = VolatilityRegimeGate(api_key_id=luno_key, api_key_secret=luno_secret)
    excel_bytes, kpis = generate_weekly_excel_bytes(connector, gate, config)

    date_tag = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    filename = f"Luno_ICIE_Weekly_Audit_{date_tag}.xlsx"

    msg = EmailMessage()
    msg["Subject"] = f"📊 Weekly Luno ICIE Risk & Portfolio Audit ({date_tag}) — ₦{kpis['tpv']:,.2f}"
    msg["From"] = gmail_sender
    msg["To"] = gmail_recipient
    msg.set_content(
        f"Good evening Bamidele,\n\n"
        f"Attached is your automated weekly Institutional Crypto Investment Engine (ICIE v2.4) Excel Audit Workbook.\n\n"
        f"--- WEEKLY EXECUTIVE SNAPSHOT ({kpis['now_utc']}) ---\n"
        f"• Total Portfolio Capital : ₦{kpis['tpv']:,.2f}\n"
        f"• Cleared Free NGN Reserve: ₦{kpis['ngn_free']:,.2f}\n"
        f"• Peak High-Water Mark    : ₦{kpis['peak_hwm']:,.2f}\n"
        f"• Active Drawdown         : {kpis['dd_pct']:.2f}% (10.00% Max Ceiling)\n\n"
        f"Tip: Click 'Save to Drive' directly on the attached .xlsx file in Gmail to archive it in your Google Drive.\n"
    )
    msg.add_attachment(
        excel_bytes,
        maintype="application",
        subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=filename,
    )

    smtp_host = ".".join(["smtp", "gmail", "com"])
    with smtplib.SMTP_SSL(smtp_host, 465, timeout=30) as smtp:
        smtp.login(gmail_sender, gmail_pass)
        smtp.send_message(msg)

    logger.info(f"[SUCCESS] Weekly ICIE Excel audit workbook ({filename}) emailed cleanly to {gmail_recipient}.")


if __name__ == "__main__":
    send_weekly_audit_email()
