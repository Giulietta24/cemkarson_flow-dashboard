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
st.caption("Quantitative mapping of options market maker hedging constraints (Gamma, Vanna, Charm) aligned with Karsan Volatility Thesis.")

st.warning("⚠️ **Disclaimer:** This dashboard is for educational and research purposes only. Options trading involves significant risk. This is not financial, legal, or tax advice.")

# --- 2. BLACK-SCHOLES QUANT ENGINE ---
def calculate_bs_greeks(S, K, T, r, sigma):
    """
    Computes precise Black-Scholes Greeks per share.
    Returns: (gamma, vanna, charm)
    FIXED: Resolved Charm double-division bug. Charm is native per-day when T is in years.
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0, 0.0, 0.0
        
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vanna = -norm.pdf(d1) * (d2 / sigma)
    
    # Karsan Fidelity Fix: Standardized daily charm derivative (negative for long options)
    charm = norm.pdf(d1) * (r / (sigma * np.sqrt(T)) - (d1 * d2) / (2 * T)) * (-1.0 / 365.0)
    
    return gamma, vanna, charm

# --- 3. IV TREND ENGINE (VANNA TRIGGER MECHANISM) ---
@st.cache_data(ttl=300)
def get_vix_iv_trend():
    """
    Fetches the 5-day delta of the CBOE Volatility Index (^VIX) to evaluate
    if macro implied volatility compression is triggering reflexively active Vanna flows.
    """
    try:
        vix = yf.Ticker("^VIX")
        hist = vix.history(period="5d")
        if len(hist) >= 2:
            return float(hist['Close'].iloc[-1] - hist['Close'].iloc[0])
        return 0.0
    except Exception:
        return 0.0

# --- SIDEBAR CONTROLS & ADHD MEMORY ANCHOR ---
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
    
    st.markdown("---")
    st.subheader("🧠 Playbook Quick Anchor")
    with st.container(border=True):
        st.markdown("""
        **🟢 Positive GEX Setup:**
        * Dealers net **long** gamma. They buy dips and sell rallies. Volatility dampening environment.
        
        **🔴 Negative GEX Setup:**
        * Dealers net **short** gamma. They sell drops and buy rips. Volatility acceleration environment.
        """)

# --- 4. HIGH-FIDELITY DATA ENGINE ---
@st.cache_data(ttl=300)
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
        
        # Pulling near-term monthly exposures where retail/institutional volume anchors the regime
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
                
                if oi > 0 and iv > 0:
                    gamma, vanna, charm = calculate_bs_greeks(current_price, strike, T, r_rate, iv)
                    
                    # Karsan Fidelity Fix: Reverted back to pristine raw Open Interest to avoid artificial skew
                    compiled_data.append({
                        'strike': strike, 'Type': 'Call', 
                        'GEX': oi * gamma * 100 * (current_price ** 2) * dte_weight, 
                        'Vanna': oi * vanna * 100 * dte_weight, 
                        'Charm': oi * charm * 100 * dte_weight,
                        'IV_Raw': iv
                    })
                    
            # Process Puts
            for _, row in opt_chain.puts.iterrows():
                strike = row['strike']
                oi = row['openInterest']
                iv = row['impliedVolatility']
                
                if oi > 0 and iv > 0:
                    gamma, vanna, charm = calculate_bs_greeks(current_price, strike, T, r_rate, iv)
                    
                    compiled_data.append({
                        'strike': strike, 'Type': 'Put', 
                        'GEX': oi * gamma * 100 * (current_price ** 2) * -1.0 * dte_weight, 
                        'Vanna': oi * vanna * 100 * -1.0 * dte_weight, 
                        'Charm': oi * charm * 100 * -1.0 * dte_weight,
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
with st.spinner("Analyzing options market structural architecture..."):
    current_price, data_matrix, data_time = load_and_compute_gex(ticker_input, risk_free_rate)
    vix_delta = get_vix_iv_trend()

if data_matrix is not None:
    st.info(f"📅 **Data Freshness Timestamp:** {data_time} | Macro VIX 5-Day Trend: {vix_delta:+.2f} points.")

    lower_bound = current_price * (1.0 - zoom_pct)
    upper_bound = current_price * (1.0 + zoom_pct)
    filtered_df = data_matrix[(data_matrix['strike'] >= lower_bound) & (data_matrix['strike'] <= upper_bound)].copy()

    total_gex = filtered_df['GEX'].sum()
    total_vanna = filtered_df['Vanna'].sum()
    total_charm = filtered_df['Charm'].sum()
    
    # --- 🤖 KARSAN FIDELITY FIX: TRUE ZERO-GAMMA CUMULATIVE FLIP POINT ---
    agg_df_sorted = data_matrix.sort_values('strike').copy()
    agg_df_sorted['cumulative_GEX'] = agg_df_sorted['GEX'].cumsum()
    
    # Identify mathematically exact sign cross
    sign_changes = agg_df_sorted[(agg_df_sorted['cumulative_GEX'] * agg_df_sorted['cumulative_GEX'].shift(1) < 0)]
    zero_gamma_strike = sign_changes['strike'].iloc[0] if not sign_changes.empty else current_price
    
    # Operationalize system states based on structural boundaries
    avg_implied_vol = filtered_df['IV_Raw'].mean() if not filtered_df.empty else 0.20
    auto_threshold_value = current_price * (max(0.005, min(0.035, avg_implied_vol * 0.05)))
    
    is_flip_zone = abs(current_price - zero_gamma_strike) <= auto_threshold_value
    vanna_active = vix_delta < -0.50  # Volatility compression decay active

    # --- SHORTHAND SHIFT FOR ADHD SCANNABILITY ---
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
    
    gex_tooltip_text = (
        f"💡 ADHD Cheat Sheet:\n\n"
        f"For every 1% that {ticker_input} moves up or down, institutional market maker software "
        f"is mechanically forced to automatically {gex_action_direction} by an estimated value of "
        f"{gex_shorthand}.\n\n"
        f"• GREEN (+): Active price safety buffer.\n"
        f"• RED (-): High-velocity momentum fuel."
    )

    # --- Metrics Layout ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label=f"Current {ticker_input} Price", value=f"${current_price:.2f}")
    with col2:
        st.metric(label="Total Gamma Exposure ($ GEX)", value=gex_shorthand, help=gex_tooltip_text)
    with col3:
        if is_flip_zone:
            state_label = "⚡ FLIP ZONE"
        else:
            state_label = "🟢 POSITIVE GEX" if total_gex > 0 else "🔴 NEGATIVE GEX"
        st.metric(label="System Status", value=state_label)

    st.caption(f"🎯 **True Cumulative Zero-Gamma Strike:** ${zero_gamma_strike:.2f} | Dynamic Proximity Range: ±${auto_threshold_value:.2f}")
    st.divider()

    # --- 6. OPERATIONALIZED VANNA & CHARM PLAYBOOK CONTROLS ---
    st.subheader("🎯 Active Execution Playbook")
    
    # 1. Check for Active Vanna Tailwind Injection
    if vanna_active and total_vanna > 0:
        st.success(f"🚀 **VANNA TAILWIND ACTIVE:** Implied Volatility is contracting (VIX down {abs(vix_delta):.2f}pts). Market maker software is mechanically forced to buy underlying stock delta to support the market. Tailwinds favor long exposure!")
    elif vix_delta > 0.50 and total_vanna > 0:
        st.error(f"⚠️ **VANNA HEADWIND ACTIVE:** Implied Volatility is spiking. Market maker delta unwinding is forcing structural automated liquidations. Stand aside or hedge.")

    # 2. Main Structural Regimes
    if is_flip_zone:
        st.warning(f"""
        ### **⚡️ SYSTEM STATUS: The Gamma Flip Node (${zero_gamma_strike:.2f})**
        * **The Market Reality:** Spot price is hovering directly on the true zero-gamma intersection. Structural gravity is completely offline; market makers are shifting inventory rapidly.
        
        #### 🛑 **ADHD Guardrail: STAND ASIDE**
        Do not entry trades inside this high-friction boundary. Wait for clear velocity confirmation:
        * 🚀 **UPWARD BREAKOUT:** If spot breaks above **${zero_gamma_strike + auto_threshold_value:.2f}**, long-gamma stabilizers activate. **BUY Calls**.
        * 📉 **DOWNWARD BREAKDOWN:** If spot slips below **${zero_gamma_strike - auto_threshold_value:.2f}**, the momentum acceleration floor tears. **BUY Puts**.
        """)
        
    elif total_gex > 0:
        st.success("""
        ### **🟢 SYSTEM STATUS: Positive GEX (Insulated Mean-Reversion Regimes)**
        * **The Market Reality:** Market makers are net long gamma. Mechanical flows blunt intraday trends. **Charm decay** is actively pulling options delta toward zero daily.
        """)
        
        tab1, tab2 = st.tabs(["💰 Play 1: Income Generation via OTM Puts", "🛡️ Play 2: Capital Preservation (CSPs)"])
        with tab1:
            st.markdown(f"""
            #### 🎯 Execution Checklist: Structural Premium Capture
            * **DTE (Time Frame):** Select **30 to 45 Days out**. This captures the absolute steepest acceleration bend of the newly updated **Charm decay curve**.
            * **Strike Selection:** Sell premium **safely below** the absolute zero-gamma flip boundary of **${zero_gamma_strike:.2f}**. 
            * **💡 Why it works:** You are trading with institutional wind at your back. Long dealer gamma pins price, while Charm decay melts the option value directly into your pocket.
            """)
        with tab2:
            st.markdown(f"""
            #### 🎯 Execution Checklist: Cash-Secured Puts (CSPs)
            * **The Goal:** Automate your "buy the dip" process to avoid psychological hesitation.
            * **The Blueprint:** Sell an out-of-the-money Put contract at a level you'd love to own shares long-term. If the stock drifts down, long dealer flows offer a soft landing near structural walls. If it stays flat, you retain the upfront cash.
            """)
            
    else:
        st.error(f"""
        ### **🔴 SYSTEM STATUS: Negative GEX (Unpinned Volatility Cascades)**
        * **The Market Reality:** Spot is trapping deeply within the short-gamma void. Price action is pro-cyclical: cascading drops force computers to amplify shorting activity. 
        
        #### 🛑 **ADHD Guardrail: PROTECT AND COLLECT**
        * Absolute execution ban on selling naked or blind options premium here. The volatility expander will blow past standard standard deviation marks.
        * **Approved Alpha Play:** Purchase **30 DTE Long Puts** or **Long VIX Call Options** to harvest premium expansion from reflexive institutional algorithmic selling.
        """)

    st.divider()

    # --- 7. PLOTLY CHART COMPONENT ---
    st.subheader("📊 Cumulative Volatility Profile Architecture")
    st.caption("Bar heights express raw dollar exposure capacity per strike. The line charts cumulative structural flows.")
    
    fig = go.Figure()
    
    # GEX Bar Trace
    fig.add_trace(go.Bar(
        x=filtered_df['strike'],
        y=filtered_df['GEX'],
        marker_color=np.where(filtered_df['GEX'] >= 0, '#2ecc71', '#e74c3c'),
        name='Strike Dollar GEX',
        hovertemplate="Strike: %{x}<br>GEX: $ %{y:,.0f}<extra></extra>"
    ))
    
    # Cumulative GEX Intersection Line
    df_sorted_filtered = filtered_df.sort_values('strike')
    df_sorted_filtered['cum_GEX_filtered'] = df_sorted_filtered['GEX'].cumsum()
    
    fig.add_trace(go.Scatter(
        x=df_sorted_filtered['strike'],
        y=df_sorted_filtered['cum_GEX_filtered'],
        line=dict(color='#f1c40f', width=3),
        name='Cumulative GEX Flow'
    ))
    
    # Spot Marker
    fig.add_vline(
        x=current_price, line_dash="dash", line_color="#3498db", line_width=2.5,
        annotation_text=" SPOT PRICE ", annotation_position="top right"
    )
    
    # Exact Zero-Gamma Intersection Marker
    fig.add_vline(
        x=zero_gamma_strike, line_dash="dot", line_color="#9b59b6", line_width=2.5,
        annotation_text=" TRUE FLIP NODE ", annotation_position="bottom left"
    )
    
    fig.update_layout(
        xaxis_title="Strike Price ($)",
        yaxis_title="Dealers Structural Dollar Exposure capacity ($)",
        margin=dict(l=20, r=20, t=20, b=20),
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    st.error("❌ Data retrieval exception triggered.")
    if 'last_error' in st.session_state:
        st.code(f"System Error Trace: {st.session_state['last_error']}", language="text")
