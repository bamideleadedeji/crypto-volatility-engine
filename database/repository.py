import sqlite3
from datetime import datetime
import pandas as pd
from core.logging import setup_logger

logger = setup_logger("Repository")
DB_NAME = "luno_institutional_engine.db"

def init_database():
    """Initializes SQLite with WAL mode and institutional tables."""
    with sqlite3.connect(DB_NAME, timeout=30) as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL;")
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                venue TEXT,
                asset TEXT,
                price REAL,
                amount REAL,
                units TEXT,
                status TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS circuit_breaker_state (
                id INTEGER PRIMARY KEY,
                failure_count INTEGER,
                is_tripped BOOLEAN,
                last_trip_time TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                portfolio_value REAL
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS paper_portfolio (
                asset TEXT PRIMARY KEY,
                units REAL,
                cash REAL,
                timestamp TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS orders (
                order_hash TEXT PRIMARY KEY,
                order_id TEXT,
                asset TEXT,
                side TEXT,
                amount REAL,
                status TEXT,
                timestamp TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_attribution (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                gross_return REAL,
                net_return REAL,
                fees_paid REAL,
                primary_driver TEXT,
                narrative TEXT
            )
        ''')
        
        cursor.execute("SELECT COUNT(*) FROM circuit_breaker_state")
        if cursor.fetchone()[0] == 0:
            cursor.execute("INSERT INTO circuit_breaker_state (id, failure_count, is_tripped, last_trip_time) VALUES (1, 0, 0, '')")
        conn.commit()
    logger.info("Database repository initialized successfully.")

def log_audit(venue: str, asset: str, price: float, amount: float, units: str, status: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(DB_NAME, timeout=30) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO audit_logs (timestamp, venue, asset, price, amount, units, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp, venue, asset, price, amount, units, status))
        conn.commit()
    logger.info(f"Audit Log Recorded -> {asset} [{status}] Amount: ₦{amount:,.2f}")

def get_paper_portfolio(initial_cash: float, initial_units: dict) -> tuple:
    with sqlite3.connect(DB_NAME, timeout=30) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT asset, units, cash FROM paper_portfolio")
        rows = cursor.fetchall()
        
    if not rows:
        cash = initial_cash
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(DB_NAME, timeout=30) as conn:
            cursor = conn.cursor()
            for asset, qty in initial_units.items():
                cursor.execute("INSERT OR REPLACE INTO paper_portfolio (asset, units, cash, timestamp) VALUES (?, ?, ?, ?)", (asset, qty, cash, timestamp))
            conn.commit()
        return cash, initial_units
    else:
        cash = rows[0][2]
        units = {row[0]: row[1] for row in rows}
        return cash, units

def update_paper_portfolio(cash: float, units: dict):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(DB_NAME, timeout=30) as conn:
        cursor = conn.cursor()
        for asset, qty in units.items():
            cursor.execute("INSERT OR REPLACE INTO paper_portfolio (asset, units, cash, timestamp) VALUES (?, ?, ?, ?)", (asset, qty, cash, timestamp))
        conn.commit()

def check_idempotency(order_hash: str) -> bool:
    """Checks if an order hash already exists in the orders table to prevent duplicate executions."""
    with sqlite3.connect(DB_NAME, timeout=30) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT status FROM orders WHERE order_hash = ?", (order_hash,))
        row = cursor.fetchone()
    return row is not None

def record_order(order_hash: str, order_id: str, asset: str, side: str, amount: float, status: str):
    """Records or updates an order hash in the SQLite WAL database."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(DB_NAME, timeout=30) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO orders (order_hash, order_id, asset, side, amount, status, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (order_hash, order_id, asset, side, amount, status, timestamp))
        conn.commit()

def log_performance_attribution(gross_return: float, net_return: float, fees_paid: float, primary_driver: str, narrative: str):
    """Records monthly performance attribution reports to SQLite WAL."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(DB_NAME, timeout=30) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO performance_attribution (timestamp, gross_return, net_return, fees_paid, primary_driver, narrative)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (timestamp, gross_return, net_return, fees_paid, primary_driver, narrative))
        conn.commit()
    logger.info(f"Performance Attribution Logged -> Net Return: {net_return:.2f}% | Driver: {primary_driver}")

def fetch_audit_ledger(limit: int = 50) -> pd.DataFrame:
    """Fetches recent audit log entries as a Pandas DataFrame for dashboard inspection."""
    with sqlite3.connect(DB_NAME, timeout=30) as conn:
        return pd.read_sql_query(f"SELECT * FROM audit_logs ORDER BY id DESC LIMIT {limit}", conn)

def fetch_performance_attribution_ledger(limit: int = 20) -> pd.DataFrame:
    """Fetches stored monthly performance attribution reports."""
    with sqlite3.connect(DB_NAME, timeout=30) as conn:
        return pd.read_sql_query(f"SELECT * FROM performance_attribution ORDER BY id DESC LIMIT {limit}", conn)
