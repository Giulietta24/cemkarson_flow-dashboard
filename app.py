import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from datetime import datetime
import pytz

# --- 1. PAGE SETUP & CONFIGURATION ---
st.set_page_config(
    page_title="GEX Flow Dashboard", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Accessibility-guarded CSS customization
st.markdown("""
<style>
    @media (prefers-reduced-motion: no-preference) {
        .stAlert { animation: fadeIn 0.5s ease-in-out; }
        @keyframes fadeIn { 0% { opacity: 0; } 100% { opacity: 1; } }
    }
</style>
""", unsafe_allow_html=True)

st.warning("⚠️ For educational and research purposes only. Not financial advice.")
st.title("📊 SPY Structural Flow Dashboard")
st.caption("Tracking Options Market Maker Hedging Constraints.")

# --- 2. PERSISTENT LEFT SIDEBAR PANEL ---
with st.sidebar:
    st.header("🎛️ Controls & Parameters")
    ticker_input = st.text_input(label="Target Ticker Symbol", value="SPY").upper()
    
    st.markdown("---")
    st.subheader("⚙️ Model Settings")
    z_val = st.slider("Chart Zoom Window (±%)", 3, 15, 6, 1)
    zoom_pct = z_val / 100.0
    risk_free_rate = st.number_input("Risk-Free Rate (r)", value=0.05, step=0.01)
    
    st.markdown("### 🗓️ Expiration Tuning")
    min_dte = st.number_input("Minimum DTE Filter", min_value=0, max_value=10, value=0)
    max_dte = st.number_input("Maximum DTE Filter", min_value=11, max_value=90, value=45)

    st.markdown("---")
    st.subheader("🧠 2-Second Cheat Sheet Playbook")
    with st.container(border=True):
        st.markdown("""
        **🟢 Above Purple Line (Calm Zone):**
        * Bias favors premium sellers and steady bounces.
        
        **🔴 Below Purple Line (Danger Zone):**
        * Expect wider swings and fast directional extension.
        """)

# --- 3. HIGH-SPEED VECTORIZED QUANT ENGINE ---
def process_chain_vectorized(df, option_type, S, T, r_rate, dte_weight):
    df = df[['strike', 'openInterest', 'impliedVolatility']].copy()
    df = df[(df['openInterest'] > 0) & (df['impliedVolatility'] > 0)].copy()
    if df.empty:
        return pd.DataFrame()
        
    K = df['strike'].values
    iv = df['impliedVolatility'].values
    oi = df['openInterest'].values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        term1 = np.log(S / K)
        term2 = (r_rate + 0.5 * iv**2) * T
        denom = iv * np.sqrt(T)
        
        d1 = (term1 + term2) / denom
        d2 = d1 - denom
        pdf_d1 = norm.pdf(d1)
        
        g_val = pdf_d1 / (S * iv * np.sqrt(T))
        v_val = -pdf_d1 * (d2 / iv)
        
        c_term = r_rate / (iv * np.sqrt(T)) - (d1 * d2) / (2 * T)
        c_val = pdf_d1 * c_term * (-1.0 / 365.0)
        
        gamma = np.nan_to_num(g_val, nan=0.0, posinf=0.0)
        vanna = np.nan_to_num(v_val, nan=0.0, posinf=0.0)
        charm = np.nan_to_num(c_val, nan=0.0, posinf=0.0)
        
        sign = 1.0 if option_type == 'call' else -1.0
        
    df['GEX'] = oi * gamma * 100 * (S**2) * dte_weight * sign
    df['Vanna'] = oi * vanna * 100 * dte_weight   
    df['Charm'] = oi * charm * 100 * dte_weight   
    df['IV_Raw'] = iv
    df['Option_Type'] = option_type
    
    return df[['strike', 'GEX', 'Vanna', 'Charm', 'IV_Raw', 'Option_Type']]

# --- 4. VECTORIZED MAX PAIN ENGINE ---
def calculate_max_pain_vectorized(opt_chain):
    try:
        calls = opt_chain.calls[['strike', 'openInterest']].dropna()
        puts = opt_chain.puts[['strike', 'openInterest']].dropna()
        
        all_strikes = sorted(set(calls['strike']) | set(puts['strike']))
        strikes = np.array(all_strikes)
        if len(strikes) == 0:
            return None
            
        c_strikes, c_oi = calls['strike'].values, calls['openInterest'].values
        p_strikes, p_oi = puts['strike'].values, puts['openInterest'].values
        
        strikes_col = strikes[:, np.newaxis]
        call_loss = np.maximum(c_strikes - strikes_col, 0) * c_oi * 100
        put_loss = np.maximum(strikes_col - p_strikes, 0) * p_oi * 100
        
        total_loss = call_loss.sum(axis=1) + put_loss.sum(axis=1)
        return float(strikes[np.argmin(total_loss)])
    except Exception:
        return None

# --- 5. METRIC SCALING COMPONENT ---
def format_scaled_exposure(val):
    abs_val = abs(val)
    sign = "-" if val < 0 else ""
    if abs_val >= 1e9:
        return f"{sign}${abs_val / 1e9:.2f}B"
    elif abs_val >= 1e6:
        return f"{sign}${abs_val / 1e6:.2f}M"
    else:
        return f"{sign}${abs_val:,.0f}"

def format_scaled_shares(val):
    abs_val = abs(val)
    sign = "-" if val < 0 else ""
    if abs_val >= 1e6:
        return f
