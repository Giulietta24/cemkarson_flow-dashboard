import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- 1. PAGE SETUP & VISUAL ANCHORS ---
st.set_page_config(page_title="Flow Dashboard", layout="wide", initial_sidebar_state="expanded")

st.title("📊 Karsan Flow & Gamma Dashboard")
st.caption("⚡ Quick Summary: This tool tracks options market maker hedges to predict if the stock market will drift upward smoothly or drop violently.")

# --- 2. ADHD-FRIENDLY CONCEPT EXPLAINER (COLLAPSED BY DEFAULT) ---
with st.expander("💡 New to this? Click here for a 30-second cheat sheet"):
    st.markdown("""
    * **The Core Idea:** Big institutions trade millions in options. Market makers (dealers) take the other side. To stay safe, dealers must mechanically buy or sell the actual stock.
    * **Volatility Trap (🟢 Long Gamma):** Dealers are forced to *buy dips* and *sell rallies*. The market gets pinned in a tight, safe range. Great time to sell options premium or buy quiet stocks.
    * **Unpinned Market (🔴 Short Gamma):** Dealers are forced to *sell when the market drops* and *buy when it rips*. This acts like rocket fuel for volatility. Great time to buy outright puts/calls or trade aggressive momentum.
    """)

st.divider()

# --- 3. SIDEBAR WITH FLOATING/HELP TEXT ---
with st.sidebar:
    st.header("🎛️ Dashboard Controls")
    
    # Streamlit uses 'help' tooltips to act as non-obtrusive floating labels/explanations
    ticker_input = st.text_input(
        label="Target Ticker Symbol", 
        value="SPY",
        help="Type any liquid ticker (like SPY, QQQ, IWM, AAPL). Highly traded index ETFs give the most accurate flow readings."
    ).upper()
    
    st.markdown("---")
    st.info("💡 **Quick Tip:** Check this dashboard during OpEx week (the 3rd week of every month) when these dealer flows are at their absolute strongest.")

# --- 4. OPTIMIZED DATA PIPELINE ---
@st.cache_data(ttl=3600)
def load_options_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        # Using faster metadata fetch
        current_price = stock.fast_info['last_price']
        expirations = stock.options
        
        if not expirations:
            return None, None
            
        total_gamma_df = pd.DataFrame()
        
        # Aggregate the front 3 near-term expirations where flow pressure is highest
        for exp in expirations[:3]:
            opt_chain = stock.option_chain(exp)
            calls = opt_chain.calls[['strike', 'openInterest']].copy()
            puts = opt_chain.puts[['strike', 'openInterest']].copy()
            
            # Simple, effective Net Gamma heuristic
            calls['Net_Gamma'] = calls['openInterest'] * 0.1
            puts['Net_Gamma'] = puts['openInterest'] * -0.1
            
            total_gamma_df = pd.concat([total_gamma_df, calls, puts], ignore_index=True)
            
        agg_df = total_gamma_df.groupby('strike').sum(numeric_only=True).reset_index()
        return current_price, agg_df
    except Exception as e:
        return None, None

# --- 5. EXECUTION & VISUAL RESULTS ---
with st.spinner("Fetching market flows... Hang tight!"):
    current_price, gamma_data = load_options_data(ticker_input)

if gamma_data is not None:
    # Zoom into the relevant price action zone (+/- 8% around current price)
    lower_bound = current_price * 0.92
    upper_bound = current_price * 1.08
    filtered_df = gamma_data[(gamma_data['strike'] >= lower_bound) & (gamma_data['strike'] <= upper_bound)]

    total_gex = filtered_df['Net_Gamma'].sum()
    
    # --- Big Bold Metrics ---
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label=f"Current {ticker_input} Price", value=f"${current_price:.2f}")
    with col2:
        # Highlighting the absolute state of the system clearly
        state_label = "🟢 VOLATILITY TRAP" if total_gex > 0 else "🔴 UNPINNED RISK ZONE"
        st.metric(label="Current Structural Mode", value=state_label)

    st.divider()

    # --- ACTIONABLE PLAYBOOK (Color-coded to trigger instant recognition) ---
    st.subheader("🎯 Today's Trading Playbook")
    
    if total_gex > 0:
        st.success("""
        **MARKET ENVIRONMENT: SAFE / MEAN-REVERTING**
        * **The Mechanics:** Market makers are insulating the market. Big crashes are structurally highly unlikely right now because dealers buy every dip.
        * **What to do:** Look to buy asset dips near support levels. Avoid chasing massive intraday breakouts (they will likely fizzle out). Great environment for collecting option premium (Theta burn).
        """)
    else:
        st.error("""
        **MARKET ENVIRONMENT: FRAGILE / EXPLOSIVE**
        * **The Mechanics:** Market makers are short gamma. If a selloff starts, dealers will panic-sell underlying stock to hedge, creating a cascade. 
        * **What to do:** Protect long portfolios. Buy outright puts or volatility instruments (VIX). If the market breaks a technical support line, expect an aggressive momentum flush downwards—do not try to catch the falling knife.
        """)

    st.divider()

    # --- 6. INTERACTIVE VISUALIZATION ---
    st.subheader("📊 The Gamma Wall Map")
    st.caption("Look for the tallest bars. Large green bars act as structural price magnets and floors; red bars act as acceleration points.")
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=filtered_df['strike'],
        y=filtered_df['Net_Gamma'],
        marker_color=np.where(filtered_df['Net_Gamma'] >= 0, '#2ecc71', '#e74c3c'),
        name='Net Gamma Exposure',
        hovertemplate="Strike: %{x}<br>Net Flow: %{y:,.0f}<extra></extra>"
    ))
    
    # Clear blue vertical anchor line for current underlying price
    fig.add_vline(
        x=current_price, 
        line_dash="dash", 
        line_color="#3498db", 
        line_width=3,
        annotation_text=" YOU ARE HERE", 
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
