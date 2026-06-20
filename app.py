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
st.caption("Institutional-grade mapping of options market maker hedging constraints.")

st.warning("⚠️ **Disclaimer:** This dashboard is for educational and research purposes only. Options trading involves significant risk. This is not financial, legal, or tax advice.")

# --- 2. HIGH-SPEED VECTORIZED QUANT ENGINE ---
def process_chain_vectorized(df, option_type, S, T, r_rate, dte_weight):
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
        
        gamma = np.nan_to_num(norm.pdf(d1) / (S * iv * np.sqrt(T)), nan=0.0, posinf=0.0)
        vanna = np.nan_to_num(-norm.pdf(d1) * (d2 / iv), nan=0.0, posinf=0.0)
        charm = np.nan_to_num(norm.pdf(d1) * (r_rate / (iv * np.sqrt(T)) - (d1 * d2) / (2 * T)) * (-1.0 / 365.0), nan=0.0, posinf=0.0)
        
        sign = 1.0 if option_type == 'call' else -1.0
        
    df['GEX'] = oi * gamma * 100 * (S**2) * dte_weight * sign
    df['Vanna'] = oi * vanna * 100 * dte_weight   
    df['Charm'] = oi * charm * 100 * dte_weight   
    df['IV_Raw'] = iv
    
    return df[['strike', 'GEX', 'Vanna', 'Charm', 'IV_Raw']]

# --- 3. VECTORIZED MAX PAIN ENGINE ---
def calculate_max_pain_vectorized(opt_chain):
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

# --- SIDEBAR CONTROLS & PERSISTENT ADHD REFERENCE ---
with st.sidebar:
    st.header("🎛️ Parameters")
    ticker_input = st.text_input(label="Target Ticker Symbol", value="BB").upper()
    
    st.markdown("---")
    st.subheader("⚙️ Model Settings")
    zoom_pct = st.slider("Chart Zoom Window (±%)", min_value=3, max_value=15, value=8, step=1) / 100.0
    risk_free_rate = st.number_input("Risk-Free Rate (r)", value=0.05, step=0.01)

    st.markdown("---")
    st.subheader("🧠 Playbook Cheat Sheet")
    # Added your complete requested explanations to the side panel as a permanent reference
    with st.container(border=True):
        st.markdown("""
        **🟢 Above 0 (Positive Gamma Zone):**
        The market has **"gravity."** If the stock price drops, market makers are forced to *buy shares* to hedge, pushing the price back up. Volatility is suppressed, and moves are slow.
        
        **🔴 Below 0 (Negative Gamma Zone):**
        **Gravity turns off**, and rocket boosters turn on in reverse. If the price drops past this point, market maker algorithms are forced to *sell shares* to hedge. Their selling forces it lower, causing more selling. Flash crashes happen here.
        
        **⚡ The Danger Zone:**
        When the **Blue Line (Price)** gets very close to the **Purple Line (Tipping Point)**, the market becomes highly unpredictable. Expect sudden, violent intraday whipsaws.
        """)

# --- 4. DATA ENGINE (FIXED CACHING BUG) ---
with st.spinner("Executing Vectorized Volatility Quant Matrices..."):
    try:
        stock = yf.Ticker(ticker_input)
        current_price = stock.fast_info.get('last_price') or stock.info.get('regularMarketPrice')
    except Exception:
        current_price = None

if current_price is None:
    st.error(f"❌ Failed to extract market price updates for {ticker_input}.")
    st.stop()

# FIX: Removed raw yfinance object from the cache return variables to avoid UnserializableReturnValueError
@st.cache_data(ttl=300)
def load_and_compute_gex_engine(ticker, r_rate, current_price):
    try:
        stock = yf.Ticker(ticker)
        expirations = stock.options
        if not expirations:
            return None, None, None, None, None
            
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
            
            if dte <= 0 or dte > 45:
                continue
            dte_weight = max(0.1, (46 - dte) / 45.0)
            T = dte / 365.0
            
            opt_chain = stock.option_chain(exp_str)
            call_res = process_chain_vectorized(opt_chain.calls, 'call', current_price, T, r_rate, dte_weight)
            put_res = process_chain_vectorized(opt_chain.puts, 'put', current_price, T, r_rate, dte_weight)
            
            if not call_res.empty: compiled_dfs.append(call_res)
            if not put_res.empty: compiled_dfs.append(put_res)
            
        if not compiled_dfs:
            return None, atm_iv_now, vix_delta_val, max_pain_val, "No Data"
            
        master_df = pd.concat(compiled_dfs, ignore_index=True)
        agg_df = master_df.groupby('strike').agg({
            'GEX': 'sum', 'Vanna': 'sum', 'Charm': 'sum', 'IV_Raw': 'mean'
        }).reset_index()
        
        raw_call_split = master_df[master_df['GEX'] >= 0].groupby('strike')['GEX'].sum().rename('Call_GEX').reset_index()
        raw_put_split = master_df[master_df['GEX'] < 0].groupby('strike')['GEX'].sum().rename('Put_GEX').reset_index()
        split_matrix = pd.merge(raw_call_split, raw_put_split, on='strike', how='outer').fillna(0.0)
        agg_df = pd.merge(agg_df, split_matrix, on='strike', how='left').fillna(0.0)
        
        est_tz = pytz.timezone('US/Eastern')
        fetch_timestamp = datetime.now(est_tz).strftime("%Y-%m-%d %I:%M:%S %p EST")
        
        return agg_df, fetch_timestamp, atm_iv_now, vix_delta_val, max_pain_val
        
    except Exception:
        return None, None, None, None, None

data_matrix, data_time, atm_iv, vix_delta, max_pain_strike = load_and_compute_gex_engine(ticker_input, risk_free_rate, current_price)

if data_matrix is not None:
    # --- TRUE ZERO-GAMMA COMPONENT ---
    agg_df_sorted = data_matrix.sort_values('strike').copy()
    agg_df_sorted['cumulative_GEX'] = agg_df_sorted['GEX'].cumsum()
    sign_changes = agg_df_sorted[(agg_df_sorted['cumulative_GEX'] * agg_df_sorted['cumulative_GEX'].shift(1) < 0)]
    zero_gamma_strike = sign_changes['strike'].iloc[0] if not sign_changes.empty else current_price

    lower_bound = current_price * (1.0 - zoom_pct)
    upper_bound = current_price * (1.0 + zoom_pct)
    filtered_df = data_matrix[(data_matrix['strike'] >= lower_bound) & (data_matrix['strike'] <= upper_bound)].copy()

    total_gex = filtered_df['GEX'].sum()
    total_charm = filtered_df['Charm'].sum()
    charm_shares = total_charm / current_price
    
    pct_from_flip = (abs(current_price - zero_gamma_strike) / current_price)
    is_approaching_zero = pct_from_flip <= 0.015

    def to_shorthand(value):
        abs_val = abs(value)
        sign_str = "-" if value < 0 else ""
        if abs_val >= 1_000_000_000: return f"{sign_str}${abs_val / 1_000_000_000:.2f}B"
        elif abs_val >= 1_000_000: return f"{sign_str}${abs_val / 1_000_000:.2f}M"
        elif abs_val >= 1_000: return f"{sign_str}${abs_val / 1_000:.2f}K"
        return f"{sign_str}${abs_val:.2f}"
        
    def to_shorthand_shares(value):
        abs_val = abs(value)
        sign_str = "-" if value < 0 else ""
        if abs_val >= 1_000_000: return f"{sign_str}{abs_val / 1_000_000:.1f}M shares"
        elif abs_val >= 1_000: return f"{sign_str}{abs_val / 1_000:.1f}K shares"
        return f"{sign_str}{abs_val:.0f} shares"

    # --- DISPLAY METRICS MATRIX ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label=f"Current {ticker_input} Price", value=f"${current_price:.2f}")
    with col2:
        st.metric(label="Total Gamma Exposure ($ GEX)", value=to_shorthand(total_gex))
    with col3:
        st.metric(
            label="Daily Charm Flow", 
            value=to_shorthand(total_charm), 
            delta=to_shorthand_shares(charm_shares),
            delta_color="off" if charm_shares >= 0 else "inverse"
        )
    with col4:
        state_label = "⚡ FLIP ZONE" if is_approaching_zero else ("🟢 POSITIVE GEX" if total_gex > 0 else "🔴 NEGATIVE GEX")
        st.metric(label="System Status", value=state_label)

    st.caption(f"🎯 **True Cumulative Zero-Gamma Strike:** ${zero_gamma_strike:.2f} | **Gravitational Max Pain Anchor:** ${max_pain_strike if max_pain_strike else 'N/A'}")
    st.divider()

    # Dynamic Danger-Close Notification
    if is_approaching_zero:
        st.markdown("### ⚡️ SYSTEM STATUS: Approaching the Tipping Point!")
        st.warning(f"**Heads Up:** The current price is only {pct_from_flip*100:.1f}% away from the Zero-Gamma line (${zero_gamma_strike:.2f}). Check the sidebar cheat sheet to see why volatility might get erratic here!")
        st.divider()

    # --- 7. PLOTLY CHART COMPONENT (CLEAN LEGEND PLACEMENT) ---
    st.subheader("📊 Cumulative Volatility Profile Architecture")

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
            marker_color='#2ecc71', name='Call Gamma GEX',
            hovertemplate="Strike: %{x}<br>Call GEX: $ %{y:,.0f}<extra></extra>"
        ))
        fig.add_trace(go.Bar(
            x=filtered_df['strike'], y=filtered_df['Put_GEX'],
            marker_color='#e74c3c', name='Put Gamma GEX',
            hovertemplate="Strike: %{x}<br>Put GEX: $ %{y:,.0f}<extra></extra>"
        ))
        fig.update_layout(barmode='group')
    
    df_sorted_display = data_matrix[(data_matrix['strike'] >= lower_bound) & (data_matrix['strike'] <= upper_bound)].sort_values('strike').copy()
    df_sorted_display['cum_GEX_display'] = df_sorted_display['GEX'].cumsum()
    
    fig.add_trace(go.Scatter(
        x=df_sorted_display['strike'], y=df_sorted_display['cum_GEX_display'],
        line=dict(color='#f1c40f', width=3), name='Cumulative GEX (Yellow Line)'
    ))
    
    fig.add_vline(x=current_price, line_dash="dash", line_color="#3498db", line_width=2.5)
    fig.add_vline(x=zero_gamma_strike, line_dash="dot", line_color="#9b59b6", line_width=2.5)
    if max_pain_strike:
        fig.add_vline(x=max_pain_strike, line_dash="dot", line_color="#e67e22", line_width=2)
        
    # FIX: Moved legend items explicitly into a clean vertical sidebar layout inside the chart container to resolve the overlapping grey boxes
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='#3498db', dash='dash'), name='🔵 BLUE LINE = Price Now'))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='#9b59b6', dash='dot'), name='🟣 PURPLE LINE = Tipping Point (0)'))
    if max_pain_strike:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='#e67e22', dash='dot'), name='🟠 ORANGE LINE = Max Pain Level'))
        
    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Strike Price ($)", yaxis_title="Exposure Capacity ($)",
        margin=dict(l=25, r=25, t=25, b=25), height=550,
        legend=dict(
            orientation="v", 
            yanchor="top", y=0.98, 
            xanchor="left", x=0.02, 
            bgcolor="rgba(20,20,20,0.9)",
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1
        )
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.error("❌ Data matrix parsing execution failed.")
