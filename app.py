import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from datetime import datetime
import pytz

# --- 1. PAGE SETUP & THEME HANDLING ---
st.set_page_config(page_title="GEX Flow Dashboard", layout="wide", initial_sidebar_state="expanded")

st.title("📊 Structural Flow & GEX Dashboard")
st.caption("Quantitative mapping of options market maker hedging constraints (Gamma, Vanna, Charm).")

st.warning("⚠️ **Disclaimer:** This dashboard is for educational and research purposes only. Options trading involves significant risk. This is not financial, legal, or tax advice.")

# --- 2. BLACK-SCHOLES QUANT ENGINE ---
def calculate_bs_greeks(S, K, T, r, sigma):
    """
    Computes precise Black-Scholes Greeks per share.
    Returns: (gamma, vanna, charm)
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0, 0.0, 0.0
        
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vanna = -norm.pdf(d1) * (d2 / sigma)
    charm = (norm.pdf(d1) * (((r / (sigma * np.sqrt(T))) - ((d1 * d2) / (2 * T))))) / 365.0
    
    return gamma, vanna, charm

# --- 3. SIDEBAR CONTROLS & ADHD MEMORY ANCHOR ---
with st.sidebar:
    st.header("🎛️ Parameters")
    
    ticker_input = st.text_input(
        label="Target Ticker Symbol", 
        value="SPY",
        help="Type highly liquid tickers (e.g., SPY, QQQ, AAPL, NVDA) for optimal options volume profiling."
    ).upper()
    
    st.markdown("---")
    st.subheader("⚙️ Model Settings")
    
    zoom_pct = st.slider("Chart Zoom Window (±%)", min_value=3, max_value=15, value=8, step=1) / 100.0
    risk_free_rate = st.number_input("Risk-Free Rate (r)", value=0.05, step=0.01)
    
    st.info("🤖 **Smart Proximity Enabled:** Flip thresholds are automatically calibrated based on the ticker's live options implied volatility.")
    
    st.markdown("---")
    st.subheader("🧠 Playbook Quick Anchor")
    with st.container(border=True):
        st.markdown("""
        **🟢 Positive GEX Setup:**
        * Dealers net **long** gamma. They buy dips and sell rallies, stabilizing price action. Favor mean-reversion, premium collection, or measured long calls.
        
        **🔴 Negative GEX Setup:**
        * Dealers net **short** gamma. They must sell drops and buy rips, accelerating volatility. Hedging flows create explosive momentum down or up. Avoid selling blind premium.
        """)

# --- 4. VOLUME-WEIGHTED DATA ENGINE ---
@st.cache_data(ttl=60)
def load_and_compute_gex(ticker, r_rate):
    try:
        stock = yf.Ticker(ticker)
        current_price = stock.fast_info.get('last_price') or stock.info.get('regularMarketPrice')
        if not current_price:
            raise ValueError(f"Unable to retrieve valid market price for ticker: {ticker}")
            
        expirations = stock.options
        if not expirations:
            return None, None, None
            
        compiled_data = []
        today = datetime.now().date()
        
        for exp_str in expirations[:4]:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            dte = (exp_date - today).days
            if dte <= 0:
                dte = 0.5
                
            T = dte / 365.0
            dte_weight = 1.0 / np.sqrt(dte) if dte > 0 else 2.0
            
            opt_chain = stock.option_chain(exp_str)
            
            # Process Calls
            for _, row in opt_chain.calls.iterrows():
                strike = row['strike']
                oi = row['openInterest']
                iv = row['impliedVolatility']
                volume = row['volume'] if 'volume' in row and not np.isnan(row['volume']) else 0
                
                if oi > 0 and iv > 0:
                    gamma, vanna, charm = calculate_bs_greeks(current_price, strike, T, r_rate, iv)
                    
                    # Volume Weighting Scalar: Prioritizes active contracts over stale open interest
                    vol_scalar = 1.0 + (volume / (oi * 0.1 + 1.0))
                    weighted_oi = oi * vol_scalar
                    
                    compiled_data.append({
                        'strike': strike, 'Type': 'Call', 
                        'GEX': weighted_oi * gamma * 100 * (current_price ** 2) * dte_weight, 
                        'Vanna': weighted_oi * vanna * 100 * dte_weight, 
                        'Charm': weighted_oi * charm * 100 * dte_weight,
                        'IV_Raw': iv
                    })
                    
            # Process Puts
            for _, row in opt_chain.puts.iterrows():
                strike = row['strike']
                oi = row['openInterest']
                iv = row['impliedVolatility']
                volume = row['volume'] if 'volume' in row and not np.isnan(row['volume']) else 0
                
                if oi > 0 and iv > 0:
                    gamma, vanna, charm = calculate_bs_greeks(current_price, strike, T, r_rate, iv)
                    
                    # Volume Weighting Scalar
                    vol_scalar = 1.0 + (volume / (oi * 0.1 + 1.0))
                    weighted_oi = oi * vol_scalar
                    
                    compiled_data.append({
                        'strike': strike, 'Type': 'Put', 
                        'GEX': weighted_oi * gamma * 100 * (current_price ** 2) * -1.0 * dte_weight, 
                        'Vanna': weighted_oi * vanna * 100 * -1.0 * dte_weight, 
                        'Charm': weighted_oi * charm * 100 * -1.0 * dte_weight,
                        'IV_Raw': iv
                    })
                    
        df = pd.DataFrame(compiled_data)
        if df.empty:
            return current_price, None, None
            
        agg_df = df.groupby('strike').agg({
            'GEX': 'sum',
            'Vanna': 'sum',
            'Charm': 'sum',
            'IV_Raw': 'mean'
        }).reset_index()
        
        est_tz = pytz.timezone('US/Eastern')
        fetch_timestamp = datetime.now(est_tz).strftime("%Y-%m-%d %I:%M:%S %p EST")
        
        return current_price, agg_df, fetch_timestamp
        
    except Exception as e:
        st.session_state['last_error'] = str(e)
        return None, None, None

# --- 5. PROCESSING & EXECUTION ENGINE ---
with st.spinner("Executing Volume-Weighted Black-Scholes integrations..."):
    current_price, data_matrix, data_time = load_and_compute_gex(ticker_input, risk_free_rate)

if data_matrix is not None:
    st.info(f"📅 **Data Freshness Timestamp:** {data_time} | Incorporating Volume-Weighted Live Options Activity.")

    lower_bound = current_price * (1.0 - zoom_pct)
    upper_bound = current_price * (1.0 + zoom_pct)
    filtered_df = data_matrix[(data_matrix['strike'] >= lower_bound) & (data_matrix['strike'] <= upper_bound)].copy()

    total_gex = filtered_df['GEX'].sum()
    total_vanna = filtered_df['Vanna'].sum()
    total_charm = filtered_df['Charm'].sum()
    
    # --- 🤖 AUTO-PILOT FLIP THRESHOLD CALCULATION ---
    avg_implied_vol = filtered_df['IV_Raw'].mean() if not filtered_df.empty else 0.20
    auto_threshold_pct = max(0.005, min(0.035, avg_implied_vol * 0.05))
    
    strikes_below = filtered_df[filtered_df['strike'] <= current_price]
    strikes_above = filtered_df[filtered_df['strike'] > current_price]
    
    nearest_lower_strike = strikes_below.iloc[-1]['strike'] if not strikes_below.empty else current_price * 0.99
    nearest_upper_strike = strikes_above.iloc[0]['strike'] if not strikes_above.empty else current_price * 1.01
    
    has_red_below = (strikes_below['GEX'] < 0).any() if not strikes_below.empty else False
    has_green_above = (strikes_above['GEX'] > 0).any() if not strikes_above.empty else False
    
    is_flip_zone = has_red_below and has_green_above and (abs(current_price - nearest_lower_strike) / current_price < auto_threshold_pct)

    # --- 🧠 SHORTHAND SHIFT FOR ADHD SCANNABILITY ---
    # Turns raw millions into clean $X.XXM or thousands into $X.XXK
    abs_gex = abs(total_gex)
    sign = "-" if total_gex < 0 else ""
    
    if abs_gex >= 1_000_000_000:
        gex_shorthand = f"{sign}${abs_gex / 1_000_000_000:.2f}B"
    elif abs_gex >= 1_000_000:
        gex_shorthand = f"{sign}${abs_gex / 1_000_000:.2f}M"
    elif abs_gex >= 1_000:
        gex_shorthand = f"{sign}${abs_gex / 1_000:.2f}K"
    else:
        gex_shorthand = f"{sign}${abs_gex:.2f}"

    gex_action_direction = "BUY shares to stabilize drops" if total_gex >= 0 else "DUMP shares, accelerating drops"
    gex_formatted_raw = f"${total_gex:,.0f}"
    
    gex_tooltip_text = (
        f"💡 ADHD Cheat Sheet:\n\n"
        f"For every 1% that {ticker_input} moves up or down, institutional market maker software "
        f"is mechanically forced to automatically {gex_action_direction} by an estimated volume-adjusted value of "
        f"{gex_shorthand} ({gex_formatted_raw} raw).\n\n"
        f"• GREEN (+): Active price safety buffer.\n"
        f"• RED (-): High-velocity momentum fuel."
    )

    # --- Metrics Layout ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label=f"Current {ticker_input} Price", value=f"${current_price:.2f}")
    with col2:
        st.metric(
            label="Total Gamma Exposure ($ GEX)", 
            value=gex_shorthand, # Clean short format displayed on-screen!
            help=gex_tooltip_text
        )
    with col3:
        if is_flip_zone:
            state_label = "⚡ FLIP ZONE"
        else:
            state_label = "🟢 POSITIVE GEX" if total_gex > 0 else "🔴 NEGATIVE GEX"
        st.metric(label="System Status", value=state_label)

    st.caption(f"🤖 **Auto-Pilot Profile:** Avg Implied Volatility: {avg_implied_vol*100:.1f}% | Dynamic Flip Warning Proximity: {auto_threshold_pct*100:.2f}%")
    st.divider()

    # --- 6. OPERATIONALIZED PLAYBOOK CONTROLS ---
    st.subheader("🎯 Active Execution Playbook")
    
    if is_flip_zone:
        st.warning(f
