import os
import requests
from typing import Dict, List, Optional

def check_risk_alerts(
    predicted_vol: float,
    portfolio_var_usd: float,
    vol_threshold_pct: float = 5.0,
    var_threshold_usd: float = 500.0,
    symbol: str = "BTC-USD"
) -> Dict:
    """
    Evaluates current predicted volatility and portfolio VaR against user-defined risk thresholds.
    """
    vol_pct = predicted_vol * 100.0 if predicted_vol < 1.0 else predicted_vol
    
    vol_exceeded = vol_pct >= vol_threshold_pct
    var_exceeded = portfolio_var_usd >= var_threshold_usd
    
    alerts_triggered = []
    if vol_exceeded:
        alerts_triggered.append(f"⚠️ **Volatility Risk Limit Exceeded:** Predicted daily volatility `{vol_pct:.2f}%` >= limit `{vol_threshold_pct:.2f}%`")
    if var_exceeded:
        alerts_triggered.append(f"🚨 **1-Day VaR Exposure Exceeded:** Predicted 1-Day VaR `${portfolio_var_usd:,.2f}` >= limit `${var_threshold_usd:,.2f}`")
        
    status = "ALERT" if (vol_exceeded or var_exceeded) else "OK"
    
    summary_msg = (
        f"📊 *[Crypto Volatility Engine Risk Alert]*\n"
        f"• *Asset:* `{symbol}`\n"
        f"• *Predicted Daily Volatility:* `{vol_pct:.2f}%` (Limit: `{vol_threshold_pct:.2f}%`)\n"
        f"• *1-Day VaR Exposure:* `${portfolio_var_usd:,.2f}` (Limit: `${var_threshold_usd:,.2f}`)\n"
        f"• *System Status:* `{'TRIGGERED / RISK BREACH' if status == 'ALERT' else 'NORMAL / WITHIN BUDGET'}`\n"
    )
    if alerts_triggered:
        summary_msg += "\n*Triggers:*\n" + "\n".join(alerts_triggered)
        
    return {
        "status": status,
        "symbol": symbol,
        "vol_pct": vol_pct,
        "vol_threshold_pct": vol_threshold_pct,
        "portfolio_var_usd": portfolio_var_usd,
        "var_threshold_usd": var_threshold_usd,
        "vol_exceeded": vol_exceeded,
        "var_exceeded": var_exceeded,
        "alerts_triggered": alerts_triggered,
        "summary_msg": summary_msg
    }


def send_telegram_alert(
    bot_token: Optional[str],
    chat_id: Optional[str],
    message: str
) -> Dict[str, str]:
    """
    Transmits an alert message to a specified Telegram Chat via Telegram Bot API.
    """
    if not bot_token or not chat_id:
        return {
            "status": "FAILED",
            "message": "Telegram Bot Token or Chat ID not configured in Environment/Secrets."
        }
        
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }
    
    try:
        response = requests.post(url, json=payload, timeout=5)
        if response.status_code == 200:
            return {"status": "SUCCESS", "message": "Telegram alert dispatched successfully."}
        else:
            return {"status": "FAILED", "message": f"Telegram API HTTP {response.status_code}: {response.text}"}
    except Exception as e:
        return {"status": "FAILED", "message": f"Network transmission error: {str(e)}"}
