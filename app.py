import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from datetime import datetime
import pytz

# --- 1. SETUP ---
st.set_page_config(
    page_title="GEX Flow", 
    layout="wide"
)

st.title("📊 SPY Structural Flow Dashboard")
st.caption("Tracking Options Market Constraints.")

# --- 2. SIDEBAR ---
with st.sidebar:
    st.header("🎛️ Controls")
    tk_in = st.text_input("Symbol", "SPY")
    ticker_input = tk_in.upper()
    z_val = st.slider("Zoom Window (±%)", 3, 15, 8, 1)
    zoom_pct = z_val / 100.0
    r_rate = st.number_input("Rate (r)", value=0.05)
    min_dte = st.number_input("Min DTE", value=0)
    max_dte = st.number_input("Max DTE", value=90)

# --- 3. DATA FETCH ---
with st.spinner("Fetching Data..."):
    try:
        stock = yf.Ticker(ticker_input)
        price = stock.fast_info.get('last_price')
        if price is None or np.isnan(price):
            price = stock.info.get('regularMarketPrice')
        if price is None or np.isnan(price):
            hist = stock.history(period="1d")
            if not hist.empty:
                price = float(hist['Close'].iloc[-1])
    except Exception:
        price = None

if not price or np.isnan(price):
    st.sidebar.error("❌ Failed to pull stock price.")
    st.stop()

price_key = float(round(price, 2))

# --- 4. DATA PROCESSING ---
@st.cache_data(ttl=300)
def load_gex(ticker, r, tgt_p, min_d, max_d):
    try:
        stock_obj = yf.Ticker(ticker)
        expirations = stock_obj.options
        if not expirations:
            return None, 0.20, None
            
        near_chain = stock_obj.option_chain(expirations[0])
        try:
            diff = (near_chain.calls['strike'] - tgt_p).abs()
            atm_iv = float(near_chain.calls.loc[diff.idxmin(), 'impliedVolatility'])
        except Exception:
            atm_iv = 0.20
            
        compiled = []
        today = datetime.now().date()
        
        for exp in expirations:
            try:
                exp_dt = datetime.strptime(exp, "%Y-%m-%d").date()
                dte = (exp_dt - today).days
                if dte < min_d or dte > max_d:
                    continue
                    
                T = max(dte, 1) / 365.0
                dte_w = max(0.1, (max_d + 1 - dte) / max(max_d, 1))
                
                chain = stock_obj.option_chain(exp)
                
                for opt_type, df in [('call', chain.calls), ('put', chain.puts)]:
                    c = df[['strike', 'openInterest', 'impliedVolatility']].dropna()
                    c = c[(c['openInterest'] > 0) & (c['impliedVolatility'] > 0)].copy()
                    if c.empty:
                        continue
                        
                    K = c['strike'].values
                    iv = c['impliedVolatility'].values
                    oi = c['openInterest'].values
                    
                    d1 = (np.log(tgt_p / K) + (r + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
                    gamma = norm.pdf(d1) / (tgt_p * iv * np.sqrt(T))
                    gamma = np.nan_to_num(gamma, nan=0.0)
                    
                    sign = 1.0 if opt_type == 'call' else -1.0
                    c['GEX'] = oi * gamma * 100 * (tgt_p**2) * dte_w * sign
                    c['Option_Type'] = opt_type
                    compiled.append(c[['strike', 'GEX', 'Option_Type']])
            except Exception:
                continue
                
        if not compiled:
            return None, atm_iv, None
            
        master = pd.concat(compiled, ignore_index=True)
        agg = master.groupby('strike')['GEX'].sum().reset_index()
        return agg, atm_iv, tgt_p
    except Exception:
        return None, 0.20, None

data_matrix, atm_iv, max_pain = load_gex(
    ticker_input, r_rate, price_key, min_dte, max_dte
)

if data_matrix is None:
    st.sidebar.error("⚠️ No data found. Widen DTE filters.")
    st.stop()

# --- 5. REGIME METRICS ---
total_gex = float(data_matrix['GEX'].sum())
zero_gamma = float(price_key)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(f"Spot {ticker_input}", f"${price_key:.2f}")
with col2:
    st.metric("Net GEX ($)", f"${total_gex / 1e6:.2f}M")
with col3:
    st.metric("ATM IV", f"{atm_iv * 100:.1f}%")
with col4:
    if total_gex > 0:
        st.success("🟢 CALM REGIME")
    else:
        st.error("🔴 ACCELERATION")

# --- 6. CHART ---
st.subheader("📊 Volatility Exposure Profile")

low_b = price_key * (1.0 - zoom_pct)
high_b = price_key * (1.0 + zoom_pct)
f_df = data_matrix[(data_matrix['strike'] >= low_b) & (data_matrix['strike'] <= high_b)]

fig = go.Figure()
fig.add_trace(go.Bar(
    x=f_df['strike'], 
    y=f_df['GEX'],
    marker_color=np.where(f_df['GEX'] >= 0, '#2ecc71', '#e74c3c'),
    showlegend=False
))

fig.add_vline(x=price_key, line_dash="dash", line_color="#3498db")
fig.update_layout(template="plotly_dark", xaxis_title="Strike", yaxis_title="GEX ($)")
st.plotly_chart(fig, use_container_width=True)
