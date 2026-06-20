import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from datetime import datetime
import pytz

# --- 1. PAGE SETUP & THEME HANDLING ---
st.set_page_config(page_title="GEX Flow Dashboard", layout="wide", initial_sidebar_state="expanded")

st.title("📊 Structural Flow & GEX Dashboard")
st.caption("Institutional-grade mapping of options market maker hedging constraints.")

st.warning("⚠️ **Disclaimer:** This dashboard is for educational and research purposes only. Options trading involves significant risk. This is not financial, legal, or tax advice.")

# --- 2. HIGH-SPEED VECTORIZED QUANT ENGINE ---
def process_chain_vectorized(df, option_type, S, T, r_rate, dte_weight):
    df = df[['strike', 'openInterest', 'impliedVolatility']].copy()
    df = df[(df['openInterest'] > 0) & (df['impliedVolatility'] > 0)].copy()
    if df.empty:
        return pd.DataFrame()
        
    K = df['strike'].values
    iv = df['impliedVolatility'].values
    oi = df['openInterest'].values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log(S / K) + (r_rate + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
        d2 = d1 - iv * np.sqrt(T)
        
        pdf_d1 = norm.pdf(d1)
        
        gamma = np.nan_to_num(pdf_d1 / (S * iv * np.sqrt(T)), nan=0.0, posinf=0.0)
        vanna = np.nan_to_num(-pdf_d1 * (d2 / iv), nan=0.0, posinf=0.0)
        charm = np.nan_to_num(pdf_d1 * (r_rate / (iv * np.sqrt(T)) - (d1 * d2) / (2 * T)) * (-1.0 / 365.0), nan=0.0, posinf=0.0)
        
        sign = 1.0 if option_type == 'call' else -1.0
        
    df['GEX'] = oi * gamma * 100 * (S**2) * dte_weight * sign
    df['Vanna'] = oi * vanna * 100 * dte_weight   
    df['Charm'] = oi * charm * 100 * dte_weight   
    df['IV_Raw'] = iv
    df['Option_Type'] = option_type
    
    return df[['strike', 'GEX', 'Vanna', 'Charm', 'IV_Raw', 'Option_Type']]

# --- 3. VECTORIZED MAX PAIN ENGINE (O(n) BROADCASTED) ---
def calculate_max_pain_vectorized(opt_chain):
    try:
        calls = opt_chain.calls[['strike', 'openInterest']].dropna()
        puts = opt_chain.puts[['strike', 'openInterest']].dropna()
        
        all_strikes = sorted(set(calls['strike']) | set(puts['strike']))
        strikes = np.array(all_strikes)
        if len(strikes) == 0:
            return None
            
        c_strikes = calls['strike'].values
        c_oi = calls['openInterest'].values
        p_strikes = puts['strike'].values
        p_oi = puts['openInterest'].values
        
        strikes_col = strikes[:, np.newaxis]
        
        call_loss = np.maximum(c_strikes - strikes_col, 0) * c_oi * 100
        put_loss = np.maximum(strikes_col - p_strikes, 0) * p_oi * 100
        
        total_loss = call_loss.sum(axis=1) + put_loss.sum(axis=1)
        return float(strikes[np.argmin(total_loss)])
    except Exception:
        return None

# --- SIDEBAR CONTROLS & PERSISTENT ADHD REFERENCE ---
with st.sidebar:
    st.header("🎛️ Parameters")
    ticker_input = st.text_input(label="Target Ticker Symbol", value="BB").upper()
    
    st.markdown("---")
    st.subheader("⚙️ Model Settings")
    zoom_pct = st.slider("Chart Zoom Window (±%)", min_value=3, max_value=15, value=8, step=1) / 100.0
    risk_free_rate = st.number_input("Risk-Free Rate (r)", value=0.05, step=0.01)

    st.markdown("---")
    st.subheader("🧠 Playbook Reference Manual")
    with st.container(border=True):
        st.markdown("""
        **🟢 Above 0 (Positive Gamma Zone):**
        The market shows signs of stabilization. Dealers balance their positions by trading against the current direction, historically resulting in compressed volatility.
        
        **🔴 Below 0 (Negative Gamma Zone):**
        Directional momentum properties often accelerate. Market-maker hedging flows align with the prevailing trend, increasing the risk of expanded volatility.
        
        **⚡ The Transition Boundary:**
        Proximity to the estimated Zero-Gamma node correlates with elevated baseline variance and unpinned liquidity characteristics.
        """)

# --- 4. DATA ENGINE ---
with st.spinner("Executing Vectorized Volatility Quant Matrices..."):
    try:
        stock = yf.Ticker(ticker_input)
        current_price = stock.fast_info.get('last_price') or stock.info.get('regularMarketPrice')
    except Exception:
        current_price = None

if current_price is None:
    st.error(f"❌ Failed to extract market price updates for {ticker_input}.")
    st.stop()

@st.cache_data(ttl=300)
def load_and_compute_gex_engine(ticker, r_rate, current_price):
    try:
        stock = yf.Ticker(ticker)
        expirations = stock.options
        if not expirations:
            return None, None, None, None, None
            
        try:
            near_exp = expirations[0]
            near_chain = stock.option_chain(near_exp)
            calls = near_chain.calls
            atm_idx = (calls['strike'] - current_price).abs().idxmin()
            atm_iv_now = float(calls.loc[atm_idx, 'impliedVolatility'])
            max_pain_val = calculate_max_pain_vectorized(near_chain)
        except Exception:
            atm_iv_now = 0.20
            max_pain_val = None
            
        try:
            vix = yf.Ticker("^VIX")
            hist = vix.history(period="5d")
            vix_delta_val = float(hist['Close'].iloc[-1] - hist['Close'].iloc[0]) if len(hist) >= 2 else 0.0
        except Exception:
            vix_delta_val = 0.0
            
        compiled_dfs = []
        today = datetime.now().date()
        
        for exp_str in expirations:
            try:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                dte = (exp_date - today).days
                
                if dte <= 0 or dte > 45:
                    continue
                dte_weight = max(0.1, (46 - dte) / 45.0)
                T = dte / 365.0
                
                opt_chain = stock.option_chain(exp_str)
                call_res = process_chain_vectorized(opt_chain.calls, 'call', current_price
