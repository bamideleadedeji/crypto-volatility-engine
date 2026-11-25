@echo off
cd /d C:\Users\user\crypto-volatility-engine

echo ==================================================
echo  RUNNING DAILY CRYPTO RISK ENGINE AND AUTO-PUSH
echo ==================================================

python paper_trader.py

if errorlevel 1 (
    echo [!] paper_trader.py encountered an error.
) else (
    echo.
    echo [*] Appending logs and pushing to GitHub...
    git add forward_test_log.csv paper_trader.py
    git commit -m "docs(auto): append daily forward test log entry [automated run]"
    git push origin main
    echo [OK] Execution and repository sync complete.
)
