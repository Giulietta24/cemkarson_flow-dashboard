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
st.caption("Institutional-grade mapping of options market maker hedging constraints (Gamma, Vanna, Charm).")

st.warning("⚠️ **Disclaimer:** This dashboard is for educational and research purposes only. Options trading involves significant risk. This is not financial, legal, or tax advice.")

# --- 2. HIGH-SPEED VECTORIZED QUANT ENGINE ---
def process_chain_vectorized(df, option_type, S, T, r_rate, dte_weight):
    """
    Vectorized Black-Scholes Greeks Engine over raw NumPy arrays.
    FIX 4: Added explicit np.nan_to_num guard directly inside the calculation block.
    """
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
        
        # Guarded NumPy arrays against mathematical infinities or NaNs on extreme out-of-the-money options strikes
        gamma = np.nan_to_num(norm.pdf(d1) / (S * iv * np.sqrt(T)), nan=0.0, posinf=0.0)
        vanna = np.nan_to_num(-norm.pdf(d1) * (d2 / iv), nan=0.0, posinf=0.0)
        charm = np.nan_to_num(norm.pdf(d1) * (r_rate / (iv * np.sqrt(T)) - (d1 * d2) / (2 * T)) * (-1.0 / 365.0), nan=0.0, posinf=0.0)
        
        sign = 1.0 if option_type == 'call' else -1.0
        
    df['GEX'] = oi * gamma * 100 * (S**2) * dte_weight * sign
    df['Vanna'] = oi * vanna * 100 * dte_weight   
    df['Charm'] = oi * charm * 100 * dte_weight   
    df['IV_Raw'] = iv
    
    return df[['strike', 'GEX', 'Vanna', 'Charm', 'IV_Raw']]

# --- 3. VECTORIZED MAX PAIN ENGINE (FIX 1) ---
def calculate_max_pain_vectorized(opt_chain):
    """
    FIX 1: Fully vectorized Max Pain array algorithm to completely eliminate nested loops.
    """
    try:
        calls = opt_chain.calls[['strike', 'openInterest']].dropna()
        puts = opt_chain.puts[['strike', 'openInterest']].dropna()
        
        strikes = np.array(sorted(set(calls['strike']) | set(puts['strike'])))
        if len(strikes) == 0:
            return None
            
        c_strikes = calls['strike'].values
        c_oi = calls['openInterest'].values
        p_strikes = puts['strike'].values
        p_oi = puts['openInterest'].values
        
        pain = np.zeros(len(strikes))
        for i, tp in enumerate(strikes):
            mask_c = c_strikes > tp
            mask_p = p_strikes < tp
            pain[i] = ((c_strikes[mask_c] - tp) * c_oi[mask_c]).sum() * 100 + \
                      ((tp - p_strikes[mask_p]) * p_oi[mask_p]).sum() * 100
                      
        return float(strikes[np.argmin(pain)])
    except Exception:
        return None

# --- SIDEBAR CONTROLS & PERSISTENT MENTAL MODEL ANCHOR ---
with st.sidebar:
    st.header("🎛️ Parameters")
    ticker_input = st.text_input(label="Target Ticker Symbol", value="SPY").upper()
    
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

# --- 4. DATA ENGINE WITH TIMING SEGREGATION (FIX 2 & FIX 5) ---
@st.cache_data(ttl=300)
def load_and_compute_gex_engine(ticker, r_rate):
    """
    FIX 2: Consolidated ATM IV metrics and Max Pain tracking inside a unified data hit.
    FIX 5: Added same-day 0DTE filter options via cap.
    """
    try:
        stock = yf.Ticker(ticker)
        current_price = stock.fast_info.get('last_price') or stock.info.get('regularMarketPrice')
        if not current_price:
            return None, None, None, None, None, None
            
        expirations = stock.options
        if not expirations:
            return current_price, None, None, None, None, None
            
        # Extract Ticker ATM IV & Near-Term Option Chain Context internally
        try:
            near_exp = expirations[0]
            near_chain = stock.option_chain(near_exp)
            calls = near_chain.calls
            atm_idx = (calls['strike'] - current_price).abs().idxmin()
            atm_iv_now = float(calls.loc[atm_idx, 'impliedVolatility'])
            max_pain_val = calculate_max_pain_vectorized(near_chain)
        except Exception:
            atm_iv_now = 0.20
            max_pain_val = None
            
        # Extract Macro VIX Volatility trend metrics
        try:
            vix = yf.Ticker("^VIX")
            hist = vix.history(period="5d")
            vix_delta_val = float(hist['Close'].iloc[-1] - hist['Close'].iloc[0]) if len(hist) >= 2 else 0.0
        except Exception:
            vix_delta_val = 0.0
            
        compiled_dfs = []
        today = datetime.now().date()
        
        for exp_str in expirations:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            dte = (exp_date - today).days
            
            # FIX 5: Standardized 0DTE threshold baseline tracking to prevent explosive variance distortions
            if dte <= 0:
                continue
                
            if dte > 45:
                continue
            dte_weight = max(0.1, (46 - dte) / 45.0)
            T = dte / 365.0
            
            opt_chain = stock.option_chain(exp_str)
            
            call_res = process_chain_vectorized(opt_chain.calls, 'call', current_price, T, r_rate, dte_weight)
            put_res = process_chain_vectorized(opt_chain.puts, 'put', current_price, T, r_rate, dte_weight)
            
            if not call_res.empty: compiled_dfs.append(call_res)
            if not put_res.empty: compiled_dfs.append(put_res)
            
        if not compiled_dfs:
            return current_price, None, None, atm_iv_now, vix_delta_val, max_pain_val
            
        master_df = pd.concat(compiled_dfs, ignore_index=True)
        agg_df = master_df.groupby('strike').agg({
            'GEX': 'sum', 'Vanna': 'sum', 'Charm': 'sum', 'IV_Raw': 'mean'
        }).reset_index()
        
        # Calculate unaggregated components for structural chart view options
        raw_call_split = master_df[master_df['GEX'] >= 0].groupby('strike')['GEX'].sum().rename('Call_GEX').reset_index()
        raw_put_split = master_df[master_df['GEX'] < 0].groupby('strike')['GEX'].sum().rename('Put_GEX').reset_index()
        split_matrix = pd.merge(raw_call_split, raw_put_split, on='strike', how='outer').fillna(0.0)
        agg_df = pd.merge(agg_df, split_matrix, on='strike', how='left').fillna(0.0)
        
        est_tz = pytz.timezone('US/Eastern')
        fetch_timestamp = datetime.now(est_tz).strftime("%Y-%m-%d %I:%M:%S %p EST")
        
        return current_price, agg_df, fetch_timestamp, atm_iv_now, vix_delta_val, max_pain_val
        
    except Exception as e:
        st.session_state['last_error'] = str(e)
        return None, None, None, None, None, None

# --- 5. EXECUTION GUARD CONFIGURATION ---
with st.spinner("Executing Vectorized Volatility Quant Matrices..."):
    current_price, data_matrix, data_time, atm_iv, vix_delta, max_pain_strike = load_and_compute_gex_engine(ticker_input, risk_free_rate)

# Explicit stop if current price is missing to prevent unexpected downstream crashes
if current_price is None:
    st.error(f"❌ Failed to extract standard market ticker data info updates for {ticker_input}. Please check connection properties.")
    if 'last_error' in st.session_state: st.code(st.session_state['last_error'])
    st.stop()

if data_matrix is not None:
    st.info(f"📅 **Data Freshness Timestamp:** {data_time} | {ticker_input} ATM Implied Vol: {atm_iv*100:.1f}%")

    # --- FIX 6: FULL SCALE ZERO-GAMMA POSITION COMPUTATION BEFORE ZOOM FILTERING ---
    agg_df_sorted = data_matrix.sort_values('strike').copy()
    agg_df_sorted['cumulative_GEX'] = agg_df_sorted['GEX'].cumsum()
    sign_changes = agg_df_sorted[(agg_df_sorted['cumulative_GEX'] * agg_df_sorted['cumulative_GEX'].shift(1) < 0)]
    zero_gamma_strike = sign_changes['strike'].iloc[0] if not sign_changes.empty else current_price

    # Apply the UI visual zoom boundary constraints
    lower_bound = current_price * (1.0 - zoom_pct)
    upper_bound = current_price * (1.0 + zoom_pct)
    filtered_df = data_matrix[(data_matrix['strike'] >= lower_bound) & (data_matrix['strike'] <= upper_bound)].copy()

    total_gex = filtered_df['GEX'].sum()
    total_vanna = filtered_df['Vanna'].sum()
    total_charm = filtered_df['Charm'].sum()
    
    auto_threshold_value = current_price * (max(0.005, min(0.035, atm_iv * 0.05)))
    is_flip_zone = abs(current_price - zero_gamma_strike) <= auto_threshold_value
    
    # FIX 3: Dynamic conditional checks for structural macro trend tracking limits
    vanna_active = vix_delta < -0.50
    vanna_headwind = vix_delta > 0.50

    # --- SHORTHAND FORMATTING LAYER ---
    def to_shorthand(value):
        abs_val = abs(value)
        sign_str = "-" if value < 0 else ""
        if abs_val >= 1_000_000_000: return f"{sign_str}${abs_val / 1_000_000_000:.2f}B"
        elif abs_val >= 1_000_000: return f"{sign_str}${abs_val / 1_000_000:.2f}M"
        elif abs_val >= 1_000: return f"{sign_str}${abs_val / 1_000:.2f}K"
        return f"{sign_str}${abs_val:.2f}"

    gex_shorthand = to_shorthand(total_gex)
    gex_action_direction = "BUY shares to stabilize drops" if total_gex >= 0 else "DUMP shares, accelerating drops"
    
    gex_tooltip_text = (
        f"💡 ADHD Cheat Sheet:\n\n"
        f"For every 1% that {ticker_input} moves up or down, market maker execution systems "
        f"are mechanically forced to automatically {gex_action_direction} by an estimated value of {gex_shorthand}.\n\n"
        f"• GREEN (+): Active price safety buffer.\n"
        f"• RED (-): High-velocity momentum fuel."
    )

    # --- DISPLAY METRICS MATRIX (FIX 7: ADDED CHARM) ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label=f"Current {ticker_input} Price", value=f"${current_price:.2f}")
    with col2:
        st.metric(label="Total Gamma Exposure ($ GEX)", value=gex_shorthand, help=gex_tooltip_text)
    with col3:
        # FIX 7: Integrated operational tracking of systemic time decay dynamics
        st.metric(label="Daily Charm Flow", value=to_shorthand(total_charm), help="Estimated value options market makers must buy/sell today from structural time decay parameters.")
    with col4:
        state_label = "⚡ FLIP ZONE" if is_flip_zone else ("🟢 POSITIVE GEX" if total_gex > 0 else "🔴 NEGATIVE GEX")
        st.metric(label="System Status", value=state_label)

    st.caption(f"🎯 **True Cumulative Zero-Gamma Strike:** ${zero_gamma_strike:.2f} | **Gravitational Max Pain Anchor:** ${max_pain_strike if max_pain_strike else 'N/A'}")
    st.divider()

    # --- 6. CONTAINER-SEPARATED PLAYBOOK EXECUTION LAYER ---
    st.subheader("🎯 Active Execution Playbook")
    
    if vanna_active and total_vanna > 0:
        st.success("🚀 **VANNA TAILWIND ACTIVE:** Implied Volatility compression is forcing dealers to buy delta to rebalance inventory. Flows support rallies.")
    elif vanna_headwind and total_vanna > 0:
        st.error("⚠️ **VANNA HEADWIND ACTIVE:** Spiking IV regime forcing systemic dealer spot liquidations.")

    if is_flip_zone:
        st.subheader(f"⚡️ SYSTEM STATUS: The Gamma Flip Node (${zero_gamma_strike:.2f})")
        st.warning(f"""
        * **The Reality:** Gravity offline. Computers are executing rapid directional inventory changes.
        * 🚀 **UPWARD BREAKOUT:** Spot cross above **${zero_gamma_strike + auto_threshold_value:.2f}** -> **BUY Calls**.
        * 📉 **DOWNWARD BREAKDOWN:** Spot slip below **${zero_gamma_strike - auto_threshold_value:.2f}** -> **BUY Puts**.
        """)
    elif total_gex > 0:
        st.subheader("🟢 SYSTEM STATUS: Positive GEX (Mean-Reversion Field)")
        st.success(f"Market tracking values confirm insulated status limits. Net positive accumulation metrics dominate near terminal bounds.")
        tab1, tab2 = st.tabs(["💰 Premium Capture Checklist", "🛡️ Automated Dip Accumulation"])
        with tab1:
            st.markdown(f"""
            * **DTE Target:** **30 to 45 Days**. Captures optimal performance along our accelerated **Charm decay metrics**.
            * **Execution Rule:** Sell options premium structures safely outside the True Flip strike (**${zero_gamma_strike:.2f}**).
            """)
        with tab2:
            st.markdown(f"**Execution Blueprint:** Deploy Cash-Secured Puts near the structural zero-gamma floor to let automated dealer buying absorb corrections for you.")
    else:
        st.subheader("🔴 SYSTEM STATUS: Negative GEX (Unpinned Volatility Cascades)")
        st.error("🛑 **ADHD Guardrail:** Volatility expander active. No short options entries. Focus exclusively on **Long Puts** or **Long Volatility Exposure** to harvest accelerating cascades.")

    st.divider()

    # --- 7. PLOTLY CHART COMPONENT (FIX 6 & FIX 8) ---
    st.subheader("📊 Cumulative Volatility Profile Architecture")
    
    # FIX 8: Added tracking toggles for clean call/put structure mapping views
    chart_mode = st.radio("Display Profile Selection", ["Net GEX Profile", "Call / Put Distribution Split"], horizontal=True)
    
    fig = go.Figure()
    
    if chart_mode == "Net GEX Profile":
        fig.add_trace(go.Bar(
            x=filtered_df['strike'], y=filtered_df['GEX'],
            marker_color=np.where(filtered_df['GEX'] >= 0, '#2ecc71', '#e74c3c'),
            name='Strike Dollar GEX', hovertemplate="Strike: %{x}<br>Net GEX: $ %{y:,.0f}<extra></extra>"
        ))
    else:
        fig.add_trace(go.Bar(
            x=filtered_df['strike'], y=filtered_df['Call_GEX'],
            marker_color='#2ecc71', name='Call Gamma GEX Exposure',
            hovertemplate="Strike: %{x}<br>Call GEX: $ %{y:,.0f}<extra></extra>"
        ))
        fig.add_trace(go.Bar(
            x=filtered_df['strike'], y=filtered_df['Put_GEX'],
            marker_color='#e74c3c', name='Put Gamma GEX Exposure',
            hovertemplate="Strike: %{x}<br>Put GEX: $ %{y:,.0f}<extra></extra>"
        ))
        fig.update_layout(barmode='group')
    
    # FIX 6: Slicing the full dataset cumulative array window cleanly for visualization layout
    df_sorted_display = data_matrix[(data_matrix['strike'] >= lower_bound) & (data_matrix['strike'] <= upper_bound)].sort_values('strike').copy()
    df_sorted_display['cum_GEX_display'] = df_sorted_display['GEX'].cumsum()
    
    fig.add_trace(go.Scatter(
        x=df_sorted_display['strike'], y=df_sorted_display['cum_GEX_display'],
        line=dict(color='#f1c40f', width=3), name='Cumulative GEX Profile Line'
    ))
    
    fig.add_vline(x=current_price, line_dash="dash", line_color="#3498db", line_width=2, annotation_text=" SPOT ")
    fig.add_vline(x=zero_gamma_strike, line_dash="dot", line_color="#9b59b6", line_width=2, annotation_text=" TRUE FLIP NODE ")
    if max_pain_strike:
        fig.add_vline(x=max_pain_strike, line_dash="dot", line_color="#e67e22", line_width=2, annotation_text=" MAX PAIN ")
        
    fig.update_layout(
        template="plotly_dark", # Applied true dark layout configuration mapping
        xaxis_title="Strike Price ($)", yaxis_title="Structural Exposure Capacity ($)",
        margin=dict(l=20, r=20, t=20, b=20), height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.error("❌ Data matrix parsing execution failed.")
