import pathlib 
import pandas as pd 
import plotly.express as px 
import streamlit as st 
 
st.set_page_config(page_title='Crypto Volatility Engine', page_icon='??', layout='wide') 
 
@st.cache_data(ttl=60) 
def load_data(file_path): 
    if not os.path.exists(file_path): 
        return None 
    df = pd.read_csv(file_path) 
    df['Timestamp'] = pd.to_datetime(df['Timestamp']) 
    for col in ['Price', 'Daily_Vol', 'VaR_95', 'Position_Dollars', 'BTC_Quantity']: 
        if col in df.columns: 
            df[col] = df[col].astype(str).str.replace('$', '', regex=False).str.replace('%%', '', regex=False).str.replace(',', '', regex=False) 
            df[col] = pd.to_numeric(df[col], errors='coerce') 
    return df.sort_values('Timestamp') 
 
def main(): 
    st.title('?? Crypto Volatility Risk Engine') 
    df = load_data('forward_test_log.csv') 
    if df is None or df.empty: 
        st.warning('No log data found in forward_test_log.csv. Please run paper_trader.py first.') 
        return 
    latest = df.iloc[-1] 
    c1, c2, c3, c4 = st.columns(4) 
    c1.metric('BTC Price', f'${float(latest["Price"]):,.2f}') 
    c2.metric('Daily Volatility', f'{float(latest["Daily_Vol"]):.2f}%%') 
    c3.metric('1-Day 95%% VaR', f'{float(latest["VaR_95"]):.2f}%%') 
    c4.metric('Target Position', f'${float(latest["Position_Dollars"]):,.2f}') 
    st.divider() 
    st.subheader('GARCH(1,1) Volatility Trend') 
    fig_vol = px.line(df, x='Timestamp', y='Daily_Vol', markers=True) 
    st.plotly_chart(fig_vol, use_container_width=True) 
    st.subheader('Audit Log') 
    st.dataframe(df.sort_values('Timestamp', ascending=False), use_container_width=True) 
 
if __name__ == '__main__': 
    main() 
code = code.replace('%%%%', '%%') 
pathlib.Path('app.py').write_text(code, encoding='utf-8') 
print('[OK] app.py generated cleanly!') 
