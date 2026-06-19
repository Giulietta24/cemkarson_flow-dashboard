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

# Legal Guardrail Requirement
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
    
    # 1. True Gamma (Sensitivity of Delta to Spot Price)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    # 2. True Vanna (Sensitivity of Delta to Implied Volatility)
    vanna = -norm.pdf(d1) * (d2 / sigma)
    
    # 3. True Charm (Delta Decay over Time)
    charm = (norm.pdf(d1) * (((r / (sigma * np.sqrt(T))) - ((d1 * d2) / (2 * T))))) / 365.0 # Daily decay proxy
    
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
    
    # Exposing parameters to fix hardcoded threshold vulnerabilities
    zoom_pct = st.slider("Chart Zoom Window (±%)", min_value=3, max_value=15, value=8, step=1) / 100.0
    flip_threshold_pct = st.slider("Flip Proximity Threshold (%)", min_value=0.5, max_value=3.0, value=1.5, step=0.1) / 100.0
    risk_free_rate = st.number_input("Risk-Free Rate (r)", value=0.05, step=0.01, help="Yield on short-term US Treasury bills.")
    
    st.markdown("---")
    st.subheader("🧠 Playbook Quick Anchor")
    with st.container(border=True):
        st.markdown("""
        **🟢 Positive GEX Setup:**
        * Dealers net **long** gamma. They buy dips and sell rallies, stabilizing price action. Favor mean-reversion, premium collection, or measured long calls.
        
        **🔴 Negative GEX Setup:**
        * Dealers net **short** gamma. They must sell drops and buy rips, accelerating volatility. Hedging flows create explosive momentum down or up. Avoid selling blind premium.
        """)

# --- 4. DATA ENGINE WITH ERROR LOGGING & PROPER SCALE ---
@st.cache_data(ttl=300)
def load_and_compute_gex(ticker, r_rate):
    try:
        stock = yf.Ticker(ticker)
        
        # Defensive price pulling
        current_price = stock.fast_info.get('last_price') or stock.info.get('regularMarketPrice')
        if not current_price:
            raise ValueError(f"Unable to retrieve valid market price for ticker: {ticker}")
            
        expirations = stock.options
        if not expirations:
            return None, None, None
            
        compiled_data = []
        today = datetime.now().date()
        
        # Pull up to 4 expirations to map intermediate structural flows
        for exp_str in expirations[:4]:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            dte = (exp_date - today).days
            if dte <= 0:
                dte = 0.5 # Intraday terminal weighting
                
            T = dte / 365.0 # Annualized time
            
            # Incorporate DTE weight to prioritize structural urgency
            dte_weight = 1.0 / np.sqrt(dte) if dte > 0 else 2.0
            
            opt_chain = stock.option_chain(exp_str)
            calls = opt_chain.calls
            puts = opt_chain.puts
            
            # Process Calls
            for _, row in calls.iterrows():
                strike = row['strike']
                oi = row['openInterest']
                iv = row['impliedVolatility']
                
                if oi > 0 and iv > 0:
                    gamma, vanna, charm = calculate_bs_greeks(current_price, strike, T, r_rate, iv)
                    
                    # Scaling: OI * Gamma * Contract Multiplier (100) * Spot^2 (to express in true dollar terms)
                    true_gex = oi * gamma * 100 * (current_price ** 2) * dte_weight
                    true_vanna = oi * vanna * 100 * dte_weight
                    true_charm = oi * charm * 100 * dte_weight
                    
                    compiled_data.append({
                        'strike': strike, 'Type': 'Call', 'GEX': true_gex, 
                        'Vanna': true_vanna, 'Charm': true_charm, 'Volume': row.get('volume', 0)
                    })
                    
            # Process Puts
            for _, row in puts.iterrows():
                strike = row['strike']
                oi = row['openInterest']
                iv = row['impliedVolatility']
                
                if oi > 0 and iv > 0:
                    gamma, vanna, charm = calculate_bs_greeks(current_price, strike, T, r_rate, iv)
                    
                    # Standard baseline heuristic: Dealers are net short downside puts
                    true_gex = oi * gamma * 100 * (current_price ** 2) * -1.0 * dte_weight
                    true_vanna = oi * vanna * 100 * -1.0 * dte_weight
                    true_charm = oi * charm * 100 * -1.0 * dte_weight
                    
                    compiled_data.append({
                        'strike': strike, 'Type': 'Put', 'GEX': true_gex, 
                        'Vanna': true_vanna, 'Charm': true_charm, 'Volume': row.get('volume', 0)
                    })
                    
        df = pd.DataFrame(compiled_data)
        if df.empty:
            return current_price, None, None
            
        agg_df = df.groupby('strike').sum(numeric_only=True).reset_index()
        
        est_tz = pytz.timezone('US/Eastern')
        fetch_timestamp = datetime.now(est_tz).strftime("%Y-%m-%d %I:%M:%S %p EST")
        
        return current_price, agg_df, fetch_timestamp
        
    except Exception as e:
        # Defensive programming: Do not swallow error blindly
        st.session_state['last_error'] = str(e)
        return None, None, None

# --- 5. PROCESSING & EXECUTION ENGINE ---
with st.spinner("Executing Black-Scholes matrix integrations..."):
    current_price, data_matrix, data_time = load_and_compute_gex(ticker_input, risk_free_rate)

if data_matrix is not None:
    st.info(f"📅 **Data Freshness Timestamp:** {data_time} | Calculated using explicit annualized Black-Scholes Greeks.")

    # Application of user-configured zoom
    lower_bound = current_price * (1.0 - zoom_pct)
    upper_bound = current_price * (1.0 + zoom_pct)
    filtered_df = data_matrix[(data_matrix['strike'] >= lower_bound) & (data_matrix['strike'] <= upper_bound)].copy()

    total_gex = filtered_df['GEX'].sum()
    total_vanna = filtered_df['Vanna'].sum()
    total_charm = filtered_df['Charm'].sum()
    
    # Defensive strike slicing to handle index boundaries safely
    strikes_below = filtered_df[filtered_df['strike'] <= current_price]
    strikes_above = filtered_df[filtered_df['strike'] > current_price]
    
    # Catching edge case empty tables cleanly
    nearest_lower_strike = strikes_below.iloc[-1]['strike'] if not strikes_below.empty else current_price * 0.99
    nearest_upper_strike = strikes_above.iloc[0]['strike'] if not strikes_above.empty else current_price * 1.01
    
    has_red_below = (strikes_below['GEX'] < 0).any() if not strikes_below.empty else False
    has_green_above = (strikes_above['GEX'] > 0).any() if not strikes_above.empty else False
    
    # Configurable Flip-zone computation
    is_flip_zone = has_red_below and has_green_above and (abs(current_price - nearest_lower_strike) / current_price < flip_threshold_pct)

    # --- Metrics Layout (Tracking True Dollar Denominated GEX Magnitude) ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label=f"Current {ticker_input} Price", value=f"${current_price:.2f}")
    with col2:
        st.metric(label="Total Gamma Exposure ($ GEX)", value=f"${total_gex:,.2f}")
    with col3:
        if is_flip_zone:
            state_label = "⚡ FLIP ZONE"
        else:
            state_label = "🟢 POSITIVE GEX" if total_gex > 0 else "🔴 NEGATIVE GEX"
        st.metric(label="System Architecture Status", value=state_label)

    st.divider()

    # --- 6. OPERATIONALIZED PLAYBOOK CONTROLS ---
    st.subheader("🎯 Active Execution Playbook")
    
    if is_flip_zone:
        st.warning(f"""
        ### **⚡️ SYSTEM STATUS: The Razor's Edge (Gamma Flip Zone)**
        * **Structural Setup:** You are inside the transition threshold. Total dollar exposure is balancing on the zero boundary.
        * **The Action Mechanics:** Do not open positioning inside this buffer zone. Monitor structural breakout levels:
          * 🚀 **Breakout Trigger:** Price cross and hold above **${nearest_upper_strike:.2f}**. Long Vanna/Charm forces kick back in.
          * 📉 **Breakdown Trigger:** Price drop below **${nearest_lower_strike:.2f}**. This triggers cascading short dealer hedging loops.
        """)
    elif total_gex > 0:
        st.success(f"""
        ### **🟢 SYSTEM STATUS: Positive GEX (Insulated Market Environment)**
        * **Structural Setup:** Dealers hold net positive aggregate positions. The structural system acts as a volatility stabilizer.
        * **Operationalized Dynamics:** **Vanna** (${total_vanna:,.0f}) and **Charm** (${total_charm:,.0f}) decay loops are actively functioning as supportive mechanics. As options move closer to expiry or implied volatility drops, dealers are systematically forced to buy the index to maintain delta-neutral positions.
        """)
        
        # Options specific layout tab matrix
        tab1, tab2 = st.tabs(["🔥 Long Call Option Play", "🛡️ Short Put / CSP Play"])
        with tab1:
            st.markdown(f"""
            * **Target Selection:** Open At-the-Money (ATM) or marginally In-the-Money (ITM) options targeting **30 to 45 Days to Expiration (DTE)**. 
            * **Execution Guardrail:** This limits your vulnerability to immediate decay while positioning you to capture systematic structural bid pressure.
            """)
        with tab2:
            st.markdown(f"""
            * **Target Selection:** Sell Out-of-the-Money Puts or execute Cash-Secured Puts (CSPs) at strikes **at or immediately below** major positive gamma walls.
            * **Execution Guardrail:** Dealer buy-walls provide a statistical barrier, lowering the risk of assignment.
            """)
    else:
        st.error(f"""
        ### **🔴 SYSTEM STATUS: Negative GEX (Unpinned Volatility State)**
        * **Structural Setup:** Market makers are caught in net short options inventory risk. Structural buffers have completely collapsed.
        * **Operationalized Dynamics:** Dealer hedging is reflexive and pro-cyclical. Drops force heavy institutional automated liquidations; rallies force immediate squeeze buying. Protect your underlying equities and look toward pure structural momentum or outright tail protection.
        """)

    st.divider()

    # --- 7. PLOTLY CHART COMPONENT (THEME AWARE) ---
    st.subheader("📊 Scaled Gamma Profile Architecture")
    st.caption("Bar heights express true dollar exposure capacity ($ GEX per strike). Higher green peaks serve as structural price walls.")
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=filtered_df['strike'],
        y=filtered_df['GEX'],
        marker_color=np.where(filtered_df['GEX'] >= 0, '#2ecc71', '#e74c3c'),
        name='Dollar GEX Exposure',
        hovertemplate="Strike: %{x}<br>True GEX: $% {y:,.2f}<extra></extra>"
    ))
    
    fig.add_vline(
        x=current_price, 
        line_dash="dash", 
        line_color="#3498db", 
        line_width=3,
        annotation_text=" SPOT PRICE ", 
        annotation_position="top right"
    )
    
    fig.update_layout(
        xaxis_title="Strike Price ($)",
        yaxis_title="True Dollar Gamma Exposure ($)",
        margin=dict(l=20, r=20, t=20, b=20),
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    st.error("❌ Data retrieval exception triggered.")
    if 'last_error' in st.session_state:
        st.code(f"System Error Trace: {st.session_state['last_error']}", language="text")
