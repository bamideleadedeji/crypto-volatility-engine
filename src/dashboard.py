import os 
import pandas as pd 
import plotly.express as px 
import plotly.graph_objects as go 
import streamlit as st 
 
st.set_page_config(page_title="Crypto Volatility & Risk Dashboard", page_icon="??", layout="wide") 
 
@st.cache_data(ttl=60) 
def load_data(file_path): 
    if not os.path.exists(file_path): 
        return None 
    df = pd.read_csv(file_path) 
    df["Timestamp"] = pd.to_datetime(df["Timestamp"]) 
    if df["Daily_Vol"].dtype == object: 
        df["Daily_Vol"] = df["Daily_Vol"].astype(str).str.rstrip("%%").astype(float) 
    if df["VaR_95"].dtype == object: 
        df["VaR_95"] = df["VaR_95"].astype(str).str.rstrip("%%").astype(float) 
    if df["Position_Dollars"].dtype == object: 
        df["Position_Dollars"] = df["Position_Dollars"].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False).astype(float) 
    return df.sort_values("Timestamp") 
 
def main(): 
    st.title("?? Crypto Volatility & Risk Engine") 
    df = load_data("forward_test_log.csv") 
    if df is None or df.empty: 
        st.warning("No log data found in forward_test_log.csv.") 
        return 
    latest = df.iloc[-1] 
    c1, c2, c3, c4 = st.columns(4) 
    c1.metric("BTC Price", f"${latest['Price']:,.2f}") 
    c2.metric("Daily Volatility", f"{latest['Daily_Vol']:.2f}%%") 
    c3.metric("1-Day 95%% VaR", f"{latest['VaR_95']:.2f}%%") 
    c4.metric("Target Position", f"${latest['Position_Dollars']:,.2f}") 
    st.divider() 
    st.subheader("GARCH(1,1) Volatility Trend") 
    fig_vol = px.line(df, x="Timestamp", y="Daily_Vol", markers=True) 
    st.plotly_chart(fig_vol, use_container_width=True) 
    st.subheader("Audit Log") 
    st.dataframe(df.sort_values("Timestamp", ascending=False), use_container_width=True) 
 
if __name__ == "__main__": 
    main() 
