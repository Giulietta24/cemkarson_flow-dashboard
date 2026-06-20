import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from datetime import datetime

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="GEX Flow", layout="wide")
st.title("📊 SPY Structural Flow Dashboard")
st.caption("Tracking Options Market Maker Constraints.")

# --- 2. SIDEBAR PANEL ---
with st.sidebar:
    st.header("🎛️ Controls")
    tk_raw = st.text_input("Symbol", "SPY")
    ticker_input = tk_raw.upper()
    z_val = st.slider("Zoom Window (±%)", 3, 15, 8)
    zoom_pct = z_val / 100.0
    r_rate = st.number_input("Rate (r)", value=0.05)
    min_dte = st.number_input("Min DTE", value=0)
    max_dte = st.number_input("Max DTE", value=90)

# --- 3. DATA FETCHING ---
with st.spinner("Fetching Market Spot..."):
    try:
        stock = yf.Ticker(ticker_input)
        price = stock.fast_info.get("last_price")
        if price is None or np.isnan(price):
            price = stock.info.get("regularMarketPrice")
        if price is None or np.isnan(price):
            hist = stock.history(period="1d")
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])
    except Exception:
        price = None

if not price or np.isnan(price):
    st.sidebar.error("❌ Stock price fetch failed.")
    st.stop()

price_key = float(round(price, 2))

# --- 4. QUANT DATA MATRIX ENGINE ---
@st.cache_data(ttl=300)
def compute_gex_profile(ticker, r, tgt_p, min_d, max_d):
    try:
        stk_obj = yf.Ticker(ticker)
        expirations = stk_obj.options
        if not expirations:
            return None, 0.20
            
        near_chain = stk_obj.option_chain(expirations[0])
        try:
            diff = (near_chain.calls["strike"] - tgt_p).abs()
            idx = diff.idxmin()
            atm_iv = float(near_chain.calls.loc[idx, "impliedVolatility"])
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
                
                chain = stk_obj.option_chain(exp)
                
                for opt_type, df in [("call", chain.calls), ("put", chain.puts)]:
                    c = df[["strike", "openInterest", "impliedVolatility"]].dropna()
                    mask = (c["openInterest"] > 0) & (c["impliedVolatility"] > 0)
                    c = c[mask].copy()
                    if c.empty:
                        continue
                        
                    K = c["strike"].values
                    iv = c["impliedVolatility"].values
                    oi = c["openInterest"].values
                    
                    d1 = (np.log(tgt_p / K) + (r + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
                    gamma = norm.pdf(d1) / (tgt_p * iv * np.sqrt(T))
                    gamma = np.nan_to_num(gamma, nan=0.0)
                    
                    sign = 1.0 if opt_type == "call" else -1.0
                    c["GEX"] = oi * gamma * 100 * (tgt_p**2) * dte_w * sign
                    compiled.append(c[["strike", "GEX"]])
            except Exception:
                continue
                
        if not compiled:
            return None, atm_iv
            
        master = pd.concat(compiled, ignore_index=True)
        agg = master.groupby("strike")["GEX"].sum().reset_index()
        return agg, atm_iv
    except Exception:
        return None, 0.20

data_matrix, atm_iv = compute_gex_profile(
    ticker_input, r_rate, price_key, min_dte, max_dte
)

if data_matrix is None:
    st.sidebar.error("⚠️ No active rows found. Expand Max DTE.")
    st.stop()

# --- 5. DASHBOARD SUMMARY SCOREBOARD ---
total_gex = float(data_matrix["GEX"].sum())

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(f"Spot {ticker_input}", f"${price_key:.2f}")
with col2:
    st.metric("Net GEX ($)", f"${total_gex / 1e6:.2f}M")
with col3:
    st.metric("ATM Implied Vol", f"{atm_iv * 100:.1f}%")

if total_gex > 0:
    st.success("🟢 CALM REGIME: Volatility Suppressed")
else:
    st.error("🔴 ACCELERATION REGIME: High Variance Tail Risk")

# --- 6. CHARTS ---
st.subheader("📊 Volatility Exposure Profile")

low_b = price_key * (1.0 - zoom_pct)
high_b = price_key * (1.0 + zoom_pct)
mask = (data_matrix["strike"] >= low_b) & (data_matrix["strike"] <= high_b)
f_df = data_matrix[mask].copy()

fig = go.Figure()
colors = np.where(f_df["GEX"] >= 0, "#2ecc71", "#e74c3c")

fig.add_trace(go.Bar(
    x=f_df["strike"], 
    y=f_df["GEX"], 
    marker_color=colors, 
    showlegend=False
))

fig.add_vline(x=price_key, line_dash="dash", line_color="#3498db")
fig.update_layout(template="plotly_dark", xaxis_title="Strike Price ($)", yaxis_title="Net Gamma Exposure ($)")
st.plotly_chart(fig, use_container_width=True)
