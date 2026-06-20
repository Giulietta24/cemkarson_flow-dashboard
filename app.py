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
        .stAlert {
            animation: fadeIn 0.5s ease-in-out;
        }
        @keyframes fadeIn {
            0% { opacity: 0; }
            100% { opacity: 1; }
        }
    }
</style>
""", unsafe_allow_html=True)

# Required educational risk disclaimer banner
st.warning("⚠️ For educational and research purposes only. Not financial advice.")

st.title("📊 SPY Structural Flow Dashboard")
st.caption("Tracking Options Market Maker Hedging Constraints.")

# --- 2. PERSISTENT LEFT SIDEBAR PANEL ---
with st.sidebar:
    st.header("🎛️ Controls & Parameters")
    ticker_input = st.text_input(label="Target Ticker Symbol", value="SPY").upper()
    
    st.markdown("---")
    st.subheader("⚙️ Model Settings")
    zoom_pct = st.slider("Chart Zoom Window (±%)", min_value=3, max_value=15, value=6, step=1) / 100.0
    risk_free_rate = st.number_input("Risk-Free Rate (r)", value=0.05, step=0.01)
    
    st.markdown("### 🗓️ Expiration Tuning")
    min_dte = st.number_input("Minimum DTE Filter", min_value=0, max_value=10, value=0)
    max_dte = st.number_input("Maximum DTE Filter", min_value=11, max_value=90, value=45)

    # ADHD COGNITIVE OVERLAY / CHEAT SHEET PLAYBOOK
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
        vanna = np.nan_to_num(-pdf_
