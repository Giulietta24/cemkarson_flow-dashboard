import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# --- Page Config ---
st.set_page_config(page_title="Structural Flow Dashboard", layout="wide")
st.title("📊 Karsan Flow & Gamma Exposure Dashboard")
st.markdown("*Dealers' mechanical hedging flows wagging the equity dog.*")

# --- Inputs ---
with st.sidebar:
    st.header("Dashboard Settings")
    ticker_input = st.text_input("Ticker Symbol (ETF preferred)", value="SPY")
    st.info("Note: Free yfinance data approximates dealer positioning assuming retail buys calls/sells puts.")

@st.cache_data(ttl=3600)
def load_options_data(ticker):
    stock = yf.Ticker(ticker)
    current_price = stock.fast_info['last_price']
    
    # Get closest monthly expiration date
    expirations = stock.options
    if not expirations:
        return None, None
    
    # Let's pull the first 3 expirations to aggregate short-term dealer gamma
    total_gamma_df = pd.DataFrame()
    
    for exp in expirations[:3]:
        opt_chain = stock.option_chain(exp)
        calls = opt_chain.calls
        puts = opt_chain.puts
        
        # Keep relevant columns
        calls = calls[['strike', 'openInterest', 'impliedVolatility']].copy()
        puts = puts[['strike', 'openInterest', 'impliedVolatility']].copy()
        
        # Core Karsan Assumption: Retail buys Calls (Dealers Short Gamma) 
        # and Retail buys Puts (Dealers Short Gamma down there too, or vice versa).
        # Standard Wall Street heuristic for Net GEX approximation:
        calls['Net_Gamma'] = calls['openInterest'] * 0.1  # Simplified Gamma proxy
        puts['Net_Gamma'] = puts['openInterest'] * -0.1   # Simplified Gamma proxy
        
        calls['Type'] = 'Call'
        puts['Type'] = 'Put'
        
        total_gamma_df = pd.concat([total_gamma_df, calls, puts], ignore_index=True)
        
    # Aggregate by Strike Price
    agg_df = total_gamma_df.groupby('strike').sum(numeric_only=True).reset_index()
    return current_price, agg_df

# --- Fetch Data ---
with st.spinner("Fetching Options Chains and calculating dealer Greeks..."):
    current_price, gamma_data = load_options_data(ticker_input)

if gamma_data is not None:
    # Filter data around current spot price for cleaner visualization (+/- 8%)
    lower_bound = current_price * 0.92
    upper_bound = current_price * 1.08
    filtered_df = gamma_data[(gamma_data['strike'] >= lower_bound) & (gamma_data['strike'] <= upper_bound)]

    # Calculate Total Market Maker Gamma Profile
    total_gex = filtered_df['Net_Gamma'].sum()
    
    # Determine Market State
    market_state = "🚀 VOLATILITY TRAP (Long Gamma/Supportive)" if total_gex > 0 else "⚠️ UNPINNED MARKET (Short Gamma/Fragile)"
    
    # --- UI Metrics Layout ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label=f"Current {ticker_input} Price", value=f"${current_price:.2f}")
    with col2:
        st.metric(label="Net Market Maker Gamma Profile", value=f"{total_gex:,.0f} units")
    with col3:
        st.metric(label="Structural Environment", value="Long Gamma" if total_gex > 0 else "Short Gamma")

    st.subheader(f"Current Environment Status: {market_state}")
    
    if total_gex > 0:
        st.success("**How to Trade This:** Mean-reversion rules. Volatility is pinned. Sell out-of-the-money options (theta burn), buy dips near support, do not chase massive breakouts because dealer flows will pull the price back down.")
    else:
        st.warning("**How to Trade This:** Momentum and tail-risk rules. Dealers are short gamma. If the market drops, they must sell into it, creating cascading waterfalls. Buy outright puts/calls, play explosive breakout momentum, avoid selling premium.")

    ---
    # --- Plotting the Gamma Profile Curve ---
    st.subheader("Dealer Net Gamma Concentration by Strike")
    st.markdown("The largest peaks act as 'magnets' or pins. Crossing below the zero line signals a shift into extreme volatility.")
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=filtered_df['strike'],
        y=filtered_df['Net_Gamma'],
        marker_color=np.where(filtered_df['Net_Gamma'] >= 0, '#2ecc71', '#e74c3c'),
        name='Net Gamma Exposure'
    ))
    
    # Add vertical line for spot price
    fig.add_vline(x=current_price, line_dash="dash", line_color="#3498db", annotation_text="Current Price")
    
    fig.update_layout(
        xaxis_title="Strike Price ($)",
        yaxis_title="Estimated Dealer Gamma Exposure",
        template="plotly_dark",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    st.error("Could not fetch options data for this ticker. Please ensure it has a liquid options chain.")
