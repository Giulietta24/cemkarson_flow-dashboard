import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from datetime import datetime
import pytz

# --- 1. PAGE SETUP & CONFIGURATION ---
st.set_page_config(page_title="GEX Flow Dashboard", layout="wide", initial_sidebar_state="expanded")

st.title("📊 SPY Structural Flow Dashboard")
st.caption("Tracking Options Market Maker Hedging Constraints.")

# --- 2. THE PERSISTENT LEFT SIDEBAR PANEL ---
with st.sidebar:
    st.header("🎛️ Controls & Parameters")
    ticker_input = st.text_input(label="Target Ticker Symbol", value="SPY").upper()
    
    st.markdown("---")
    st.subheader("⚙️ Model Settings")
    zoom_pct = st.slider("Chart Zoom Window (±%)", min_value=3, max_value=15, value=6, step=1) / 100.0
    risk_free_rate = st.number_input("Risk-Free Rate (r)", value=0.05, step=0.01)

    # --- QUICK-LOOK CHEAT SHEET ---
    st.markdown("---")
    st.subheader("🧠 2-Second Cheat Sheet Playbook")
    with st.container(border=True):
        st.markdown("""
        **🟢 Above Purple Line (Calm Zone):**
        * Market is stable.
        * Market maker programs buy dips and sell rallies.
        * Bias favors premium sellers and steady bounces.
        
        **🔴 Below Purple Line (Danger Zone):**
        * Slippery slope.
        * Market maker programs sell drops and follow momentum.
        * Expect wider swings and fast directional extension.
        
        **⚡ Stacking Directly On the Line:**
        * High variance zone.
        * Fast algorithmic cross-currents.
        * Hands off the keyboard until a breakout side is chosen!
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

# --- 5. HUMAN-READABLE METRIC SCALING COMPONENT ---
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
        return f"{sign}{abs_val / 1e6:.2f}M"
    else:
        return f"{sign}{abs_val:,.0f}"

# --- 6. DATA INGESTION ENGINE ---
with st.spinner("Executing Volatility Quant Matrices..."):
    try:
        stock = yf.Ticker(ticker_input)
        # Resilient price fallback cascade
        current_price = stock.fast_info.get('last_price')
        if current_price is None or np.isnan(current_price):
            current_price = stock.info.get('regularMarketPrice')
        if current_price is None or np.isnan(current_price):
            hist = stock.history(period="1d")
            if not hist.empty:
                current_price = float(hist['Close'].iloc[-1])
    except Exception:
        current_price = None

if current_price is None or np.isnan(current_price):
    st.error(f"❌ Failed to extract market price updates for {ticker_input}. Yahoo Finance API might be throttled or encountering downtime.")
    st.stop()

@st.cache_data(ttl=300)
def load_and_compute_gex_engine(ticker, r_rate, current_price):
    try:
        stock_obj = yf.Ticker(ticker)
        expirations = stock_obj.options
        if not expirations:
            return None, None, None, None
            
        try:
            near_chain = stock_obj.option_chain(expirations[0])
            atm_idx = (near_chain.calls['strike'] - current_price).abs().idxmin()
            atm_iv_now = float(near_chain.calls.loc[atm_idx, 'impliedVolatility'])
            max_pain_val = calculate_max_pain_vectorized(near_chain)
        except Exception:
            atm_iv_now = 0.20
            max_pain_val = None
            
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
                
                opt_chain = stock_obj.option_chain(exp_str)
                call_res = process_chain_vectorized(opt_chain.calls, 'call', current_price, T, r_rate, dte_weight)
                put_res = process_chain_vectorized(opt_chain.puts, 'put', current_price, T, r_rate, dte_weight)
                
                if not call_res.empty: compiled_dfs.append(call_res)
                if not put_res.empty: compiled_dfs.append(put_res)
            except Exception:
                continue
            
        if not compiled_dfs:
            return None, atm_iv_now, max_pain_val, "No Compiled Data"
            
        master_df = pd.concat(compiled_dfs, ignore_index=True)
        master_df = master_df[master_df.groupby('strike')['GEX'].transform('sum').abs() > 10000]
        
        agg_df = master_df.groupby('strike').agg({
            'GEX': 'sum', 'Vanna': 'sum', 'Charm': 'sum', 'IV_Raw': 'mean'
        }).reset_index()
        
        raw_call_split = master_df[master_df['Option_Type'] == 'call'].groupby('strike')
