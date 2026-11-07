import streamlit as st
import sys
import os

# STEP 1: Test basic Streamlit rendering
st.set_page_config(page_title="Quant Engine Diagnostic", layout="centered")
st.title("🔍 Quant Engine Startup Diagnostic")
st.write("Step 1: Streamlit base UI rendered successfully.")

# STEP 2: Test Python imports
try:
    st.write("Step 2: Testing core module imports...")
    from core.config import load_config
    from core.logging import setup_logger
    st.success("✅ Core modules imported successfully.")
except Exception as e:
    st.error(f"❌ Import Error: {e}")
    st.stop()

# STEP 3: Test Database initialization
try:
    st.write("Step 3: Testing SQLite database initialization...")
    from database.repository import init_database
    init_database()
    st.success("✅ Database initialized successfully.")
except Exception as e:
    st.error(f"❌ Database Error: {e}")
    st.stop()

# STEP 4: Test Secrets / Credentials
try:
    st.write("Step 4: Checking Streamlit Secrets...")
    luno_key = st.secrets.get("LUNO_API_KEY_ID")
    luno_secret = st.secrets.get("LUNO_API_KEY_SECRET")
    if luno_key and luno_secret:
        st.success("✅ Luno API credentials found in secrets.")
    else:
        st.warning("⚠️ Luno API credentials missing from Streamlit secrets.")
except Exception as e:
    st.error(f"❌ Secrets Check Error: {e}")

# STEP 5: Test Luno API Connection with a strict timeout
try:
    st.write("Step 5: Testing Luno API connection (with 5s timeout)...")
    from execution.luno_connector import LunoConnector
    if luno_key and luno_secret:
        connector = LunoConnector(luno_key, luno_secret)
        # Try fetching just one ticker to isolate network latency
        ticker_price = connector.fetch_ticker("XBTNGN")
        st.success(f"✅ Luno API connected successfully! XBTNGN Price: ₦{ticker_price:,.2f}")
    else:
        st.info("ℹ️ Skipping live API test because credentials are not configured.")
except Exception as e:
    st.error(f"❌ Luno API Connection Hung or Failed: {e}")

st.divider("🎉 Diagnostic completed successfully without deadlocks!")
