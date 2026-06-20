import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from datetime import datetime
import pytz

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="GEX Flow", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

st.warning("⚠️ Research only. Not financial advice.")
st.title("📊 SPY Structural Flow Dashboard")
st.caption("Tracking Options Market Maker Constraints.")

# --- 2. SIDEBAR PANEL ---
with st.sidebar:
    st.header("🎛️ Controls")
    tk_in = st.text_input(label="Symbol", value="SPY")
    ticker_input = tk_in.upper()
    
    st.markdown("---")
    st.subheader("⚙️ Settings")
    z_val = st.slider("Zoom Window (±%)", 3, 15, 8, 1)
    zoom_pct = z_val / 100.0
    risk_free_rate = st.number_input(
        "Risk-Free Rate (r)", 
        value=0.05, 
        step=0.01
    )
    
    st.markdown("### 🗓️ Expirations")
    min_dte = st.number_input(
        "Min DTE", 
        min_value=0, 
        max_value=10, 
        value=0
    )
    max_dte = st.number_input(
        "Max DTE", 
        min_value=11, 
        max_value=365, 
        value=90
    )

# --- 3. QUANT ENGINE ---
def process_chain_vectorized(df, opt_type, S, T, r, d_wt):
    req_cols = ['strike', 'openInterest', 'impliedVolatility']
    df = df[req_cols].copy()
    
    valid_mask = (df['openInterest'] > 0) & (
        df['impliedVolatility'] > 0
    )
    df = df[valid_mask].copy()
    if df.empty:
        return pd.DataFrame()
        
    K = df['strike'].values
    iv = df['impliedVolatility'].values
    oi = df['openInterest'].values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        t1 = np.log(S / K)
        t2 = (r + 0.5 * iv**2) * T
        denom = iv * np.sqrt(T)
        
        d1 = (t1 + t2) / denom
        d2 = d1 - denom
        pdf_d1 = norm.pdf(d1)
        
        g_val = pdf_d1 / (S * iv * np.sqrt(T))
        v_val = -pdf_d1 * (d2 / iv)
        
        c_t = r / (iv * np.sqrt(T)) - (d1 * d2) / (2 * T)
        c_val = pdf_d1 * c_t * (-1.0 / 365.0)
        
        gamma = np.nan_to_num(g_val, 0.0, 0.0)
        vanna = np.nan_to_num(v_val, 0.0, 0.0)
        charm = np.nan_to_num(c_val, 0.0, 0.0)
        
        sign = 1.0 if opt_type == 'call' else -1.0
        
    df['GEX'] = oi * gamma * 100 * (S**2) * d_wt * sign
    df['Vanna'] = oi * vanna * 100 * d_wt   
    df['Charm'] = oi * charm * 100 * d_wt   
    df['IV_Raw'] = iv
    df['Option_Type'] = opt_type
    
    out_cols = [
        'strike', 'GEX', 'Vanna', 
        'Charm', 'IV_Raw', 'Option_Type'
    ]
    return df[out_cols]

# --- 4. MAX PAIN ---
def calculate_max_pain_vectorized(opt_chain):
    try:
        c_df = opt_chain.calls
        p_df = opt_chain.puts
        calls = c_df[['strike', 'openInterest']].dropna()
        puts = p_df[['strike', 'openInterest']].dropna()
        
        stk_set = set(calls['strike']) | set(puts['strike'])
        all_strikes = sorted(stk_set)
        strikes = np.array(all_strikes)
        if len(strikes) == 0:
            return None
            
        c_stk, c_oi = calls['strike'].values, calls['openInterest'].values
        p_stk, p_oi = puts['strike'].values, puts['openInterest'].values
        
        stk_col = strikes[:, np.newaxis]
        c_loss = np.maximum(c_stk - stk_col, 0) * c_oi * 100
        p_loss = np.maximum(stk_col - p_stk, 0) * p_oi * 100
        
        total_loss = c_loss.sum(axis=1) + p_loss.sum(axis=1)
        return float(strikes[np.argmin(total_loss)])
    except Exception:
        return None

# --- 5. FORMATTING ---
def format_scaled_exposure(val):
    abs_val = abs(val)
    sign = "-" if val < 0 else ""
    if abs_val >= 1e9:
        return f
