import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import pytz  # To keep time zones explicitly accurate

# --- 1. PAGE SETUP & VISUAL ANCHORS ---
st.set_page_config(page_title="Flow Dashboard", layout="wide", initial_sidebar_state="expanded")

st.title("📊 Karsan Flow & Gamma Dashboard")
st.caption("⚡ Quick Summary: This tool tracks options market maker hedges to predict if the stock market will drift upward smoothly or drop violently.")

# --- 2. SIDEBAR WITH FLOATING LABELS & PERMANENT MEMORY ANCHOR ---
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
@st.cache_data(ttl=300)  # Reduced to 5 minutes so users get fresher data on refresh
def load_options_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        current_price = stock.fast_info['last_price']
        expirations = stock.options
        
        if not expirations:
            return None, None, None
            
        total_gamma_df = pd.DataFrame()
        
        for exp in expirations[:3]:
            opt_chain = stock.option_chain(exp)
            calls = opt_chain.calls[['strike', 'openInterest']].copy()
            puts = opt_chain.puts[['strike', 'openInterest']].copy()
            
            calls['Net_Gamma'] = calls['openInterest'] * 0.1
            puts['Net_Gamma'] = puts['openInterest'] * -0.1
            
            total_gamma_df = pd.concat([total_gamma_df, calls, puts], ignore_index=True)
            
        agg_df = total_gamma_df.groupby('strike').sum(numeric_only=True).reset_index()
        
        # Capture the precise system execution timestamp (US Eastern Time for markets)
        est_tz = pytz.timezone('US/Eastern')
        fetch_timestamp = datetime.now(est_tz).strftime("%Y-%m-%d %I:%M:%S %p EST")
        
        return current_price, agg_df, fetch_timestamp
    except Exception as e:
        return None, None, None

# --- 4. EXECUTION & VISUAL RESULTS ---
with st.spinner("Fetching market flows... Hang tight!"):
    current_price, gamma_data, data_time = load_options_data(ticker_input)

if gamma_data is not None:
    # ADHD Visual Cue: Date/Time Stamp Container clearly shown up top
    st.info(f"📅 **Data Freshness Timestamp:** {data_time} (Note: Public options chain data is subject to a 15-20 min exchange delay).")

    lower_bound = current_price * 0.92
    upper_bound = current_price * 1.08
    filtered_df = gamma_data[(gamma_data['strike'] >= lower_bound) & (gamma_data['strike'] <= upper_bound)]

    total_gex = filtered_df['Net_Gamma'].sum()
    
    # --- Big Bold Metrics ---
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label=f"Current {ticker_input} Price", value=f"${current_price:.2f}")
    with col2:
        state_label = "🟢 LONG GAMMA ENVIRONMENT" if total_gex > 0 else "🔴 SHORT GAMMA ENVIRONMENT"
        st.metric(label="Current Structural Mode", value=state_label)

    st.divider()

    # --- ACTIONABLE PLAYBOOK ---
    st.subheader("🎯 Active Execution Playbook")
    
    if total_gex > 0:
        st.success("""
        ### **🟢 CURRENT SETUP: Buy Asset / Sell Premium**
        
        * **Dashboard Signal:** The total GEX metric is highly positive (**Green**), and the current stock price is sitting right below a massive green "Gamma Wall" peak.
        * **The Action:** **BUY** the underlying asset or **SELL** out-of-the-money puts.
        * **Why (The Flow Mechanics):** Time decay (**Charm**) and dropping volatility (**Vanna**) will force dealers to continuously buy the underlying shares mechanically over the coming days, guaranteeing a "structural floor" beneath your trade.
        """)
    else:
        st.error("""
        ### **🔴 CURRENT SETUP: Buy Volatility / Avoid Dips**
        
        * **Dashboard Signal:** The current price breaks **below** the cluster of positive green bars and enters a zone dominated by red bars (**Negative Gamma**).
        * **The Action:** **SELL** your long positions / **BUY** Put Options / **BUY** VIX calls.
        * **Why (The Flow Mechanics):** The market is now **"unpinned."** Any further drop forces dealers to dump shares to re-hedge, amplifying the downward spiral. Dips will not be bought by market makers; they will be aggressively shorted by them.
        """)

    st.divider()

    # --- 5. INTERACTIVE VISUALIZATION ---
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
