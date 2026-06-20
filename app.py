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

# BUG FIX 5 — Accessibility guard for prefers-reduced-motion
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

# BUG FIX 1 — Restore mandatory risk warning / educational disclaimer
st.warning("⚠️ For educational and research purposes only. Not financial advice.")

st.title("📊 SPY Structural Flow Dashboard")
st.caption("Tracking Options Market Maker Hedging Constraints.")

# --- 2. PERSISTENT LEFT SIDEBAR PANEL ---
with st.sidebar:
    st.header("🎛️ Controls & Parameters")
    ticker_input = st.text_input(label="Target Ticker Symbol", value="SPY
