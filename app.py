import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import pytz

# --- 1. PAGE SETUP & VISUAL ANCHORS ---
st.set_page_config(page_title="Flow Dashboard", layout="wide", initial_sidebar_state="expanded")

st.title("📊 Karsan Flow & Gamma Dashboard")
st.caption("⚡ Quick Summary: This tool tracks options market maker hedges to predict if the stock market will drift upward smoothly or drop violently.")

# --- 2. SIDEBAR WITH PERMANENT MEMORY ANCHOR ---
with st.sidebar:
    st.header("🎛️ Dashboard Controls")
    
    ticker_input = st.text_input(
        label="Target Ticker Symbol", 
        value="SPY",
        help="Type any liquid ticker (like SPY, QQQ, IWM, AAPL). Highly traded index ETFs give the most accurate flow readings."
    ).upper()
    
    st.markdown("---")
    
    st.subheader("🧠 Buy/Sell Memory Anchor")
    with st.container(border=True):
        st.markdown("""
        ### 🟢 The "Buy Asset" Setup
        * **Signal:** Total GEX is positive (Green), spot is just below a massive green Gamma Wall.
        * **Action:** **BUY** asset or **SELL** OTM puts.
        * **Why:** Charm (time decay) and Vanna (dropping vol) force dealers to mechanically buy over the coming days.
        
        ### 🔴 The "Buy Vol" Setup
        * **Signal:** Price breaks below green clusters into red bars (Negative Gamma).
        * **Action:** **SELL** longs / **BUY** Puts or VIX calls.
        * **Why:** Market is 'unpinned'. Dealers must panic-dump shares to hedge drops, accelerating crashes.
        """)

st.divider()

# --- 3. OPTIMIZED DATA PIPELINE WITH TIMESTAMPS ---
@st.cache_data(ttl=300)
def load_options_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        current_price = stock.fast_info['last_price']
        expirations = stock.options
        
        if not expirations:
            return None, None, None
            
        total_gamma_df = pd.DataFrame()
        
        # Aggregate front 3 expirations where hedging pressure is most intense
        for exp in expirations[:3]:
            opt_chain = stock.option_chain(exp)
            calls = opt_chain.calls[['strike', 'openInterest']].copy()
            puts = opt_chain.puts[['strike', 'openInterest']].copy()
            
            calls['Net_Gamma'] = calls['openInterest'] * 0.1
            puts['Net_Gamma'] = puts['openInterest'] * -0.1
            
            total_gamma_df = pd.concat([total_gamma_df, calls, puts], ignore_index=True)
            
        agg_df = total_gamma_df.groupby('strike').sum(numeric_only=True).reset_index()
        
        est_tz = pytz.timezone('US/Eastern')
        fetch_timestamp = datetime.now(est_tz).strftime("%Y-%m-%d %I:%M:%S %p EST")
        
        return current_price, agg_df, fetch_timestamp
    except Exception as e:
        return None, None, None

# --- 4. EXECUTION & VISUAL RESULTS ---
with st.spinner("Fetching market flows... Hang tight!"):
    current_price, gamma_data, data_time = load_options_data(ticker_input)

if gamma_data is not None:
    st.info(f"📅 **Data Freshness Timestamp:** {data_time} (Note: Public options chain data is subject to a 15-20 min exchange delay).")

    # Zoom window around current spot price (+/- 8%)
    lower_bound = current_price * 0.92
    upper_bound = current_price * 1.08
    filtered_df = gamma_data[(gamma_data['strike'] >= lower_bound) & (gamma_data['strike'] <= upper_bound)].copy()

    total_gex = filtered_df['Net_Gamma'].sum()
    
    # --- FLIP ZONE DETECTION MATH ---
    # Find closest strikes immediately above and below the current spot price
    strikes_below = filtered_df[filtered_df['strike'] <= current_price]
    strikes_above = filtered_df[filtered_df['strike'] > current_price]
    
    has_red_below = (strikes_below['Net_Gamma'] < 0).any() if not strikes_below.empty else False
    has_green_above = (strikes_above['Net_Gamma'] > 0).any() if not strikes_above.empty else False
    
    # Find exact nearby key trigger levels for the user interface
    nearest_upper_strike = strikes_above.iloc[0]['strike'] if not strikes_above.empty else current_price + 1
    nearest_lower_strike = strikes_below.iloc[-1]['strike'] if not strikes_below.empty else current_price - 1
    
    # Define if we are standing directly on the border line (Flip Zone)
    is_flip_zone = has_red_below and has_green_above and (abs(current_price - nearest_lower_strike) / current_price < 0.015)

    # --- Big Bold Status Metrics ---
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label=f"Current {ticker_input} Price", value=f"${current_price:.2f}")
    with col2:
        if is_flip_zone:
            state_label = "⚡ RAZOR'S EDGE (Gamma Flip Zone)"
        else:
            state_label = "🟢 LONG GAMMA (Safe Drift)" if total_gex > 0 else "🔴 SHORT GAMMA (High Risk)"
        st.metric(label="Current Structural Mode", value=state_label)

    st.divider()

    # --- 5. ACTIONABLE PLAYBOOK (ADHD DYNAMIC FOCUS BLOCKS) ---
    st.subheader("🎯 Active Execution Playbook")
    
    if is_flip_zone:
        st.warning(f"""
        ### **⚡️ CURRENT SETUP: The Razor's Edge (Gamma Flip Zone)**
        
        * **Dashboard Signal:** You are standing right on the boundary between a green zone above and a red zone below. Do not look at total market statistics blindly here!
        * **The Action (How to Play It):** Do not guess or front-run the direction. Let the market cross a confirmation line first:
          * 🚀 **The BUY Trigger:** If the price clears **${nearest_upper_strike:.2f}** and holds, **BUY the asset**. The green tractor beam is taking over, and dealers will mechanically push you up toward higher walls.
          * 📉 **The SHORT Trigger:** If the price slips below **${nearest_lower_strike:.2f}**, **SELL your long positions / BUY Puts**. You have fallen off the cliff into the unpinned red zone, and dealers will panic-sell underlying stock to hedge, accelerating a rapid drop.
        """)
    elif total_gex > 0:
        st.success("""
        ### **🟢 CURRENT SETUP: Buy Asset / Sell Premium**
        
        * **Dashboard Signal:** The total GEX metric is highly positive (**Green**), and the current stock price is sitting comfortably inside a cluster of green "Gamma Wall" peaks.
        * **The Action (How to Play It):** **BUY** the underlying asset, buy the dips, or **SELL** out-of-the-money puts to harvest theta burn. Avoid chasing vertical intraday breakouts.
        * **Why:** Time decay (**Charm**) and dropping volatility (**Vanna**) will force dealers to continuously buy the underlying shares mechanically over the coming days, guaranteeing an active "structural floor" beneath your trade.
        """)
    else:
        st.error("""
        ### **🔴 CURRENT SETUP: Buy Volatility / Avoid Dips**
        
        * **Dashboard Signal:** The price is submerged deep inside a zone dominated entirely by red bars (**Negative Gamma**).
        * **The Action (How to Play It):** **SELL** long portfolios, buy outright momentum options, or **BUY VIX calls**. Do not try to catch the falling knife.
        * **Why:** The market is structurally **"unpinned."** Any selling pressure forces dealers to mechanically dump shares to re-hedge, amplifying downward cascades. 
        """)

    st.divider()

    # --- 6. INTERACTIVE VISUALIZATION ---
    st.subheader("📊 The Gamma Wall Map")
    st.caption("Identify your current position relative to the nearest walls. Green walls block drops; Red walls accelerate them.")
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=filtered_df['strike'],
        y=filtered_df['Net_Gamma'],
        marker_color=np.where(filtered_df['Net_Gamma'] >= 0, '#2ecc71', '#e74c3c'),
        name='Net Gamma Exposure',
        hovertemplate="Strike: %{x}<br>Net Flow: %{y:,.0f}<extra></extra>"
    ))
    
    fig.add_vline(
        x=current_price, 
        line_dash="dash", 
        line_color="#3498db", 
        line_width=3,
        annotation_text=" 🎯 YOU ARE HERE", 
        annotation_position="top right"
    )
    
    fig.update_layout(
        xaxis_title="Strike Price ($)",
        yaxis_title="Dealer Hedging Flow Power",
        template="plotly_dark",
        height=450,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    st.error(f"❌ Couldn't load options for '{ticker_input}'. Double-check the ticker symbol, or try a highly liquid one like SPY or QQQ.")
