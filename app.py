import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from datetime import datetime
import pytz

# --- 1. SETUP & CONFIGURATION ---
st.set_page_config(page_title="GEX Flow Dashboard", layout="wide", initial_sidebar_state="expanded")
st.title("📊 SPY Structural Flow Dashboard")
st.caption("Tracking Options Market Maker Hedging Constraints.")

with st.sidebar:
    st.header("🎛️ Controls")
    ticker_input = st.text_input(label="Target Ticker Symbol", value="SPY").upper()
    st.markdown("---")
    st.subheader("⚙️ Settings")
    zoom_pct = st.slider("Chart Zoom Window (±%)", 3, 15, 6, 1) / 100.0
    risk_free_rate = st.number_input("Risk-Free Rate (r)", value=0.05, step=0.01)

# --- 2. VECTORIZED QUANT ENGINE ---
def process_chain_vectorized(df, option_type, S, T, r_rate, dte_weight):
    df = df[['strike', 'openInterest', 'impliedVolatility']].copy()
    df = df[(df['openInterest'] > 0) & (df['impliedVolatility'] > 0)].copy()
    if df.empty:
        return pd.DataFrame()
    K, iv, oi = df['strike'].values, df['impliedVolatility'].values, df['openInterest'].values
    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log(S / K) + (r_rate + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
        d2 = d1 - iv * np.sqrt(T)
        pdf_d1 = norm.pdf(d1)
        gamma = np.nan_to_num(pdf_d1 / (S * iv * np.sqrt(T)), nan=0.0)
        vanna = np.nan_to_num(-pdf_d1 * (d2 / iv), nan=0.0)
        charm = np.nan_to_num(pdf_d1 * (r_rate / (iv * np.sqrt(T)) - (d1 * d2) / (2 * T)) * (-1.0 / 365.0), nan=0.0)
        sign = 1.0 if option_type == 'call' else -1.0
    df['GEX'] = oi * gamma * 100 * (S**2) * dte_weight * sign
    df['Vanna'] = oi * vanna * 100 * dte_weight   
    df['Charm'] = oi * charm * 100 * dte_weight   
    df['IV_Raw'] = iv
    df['Option_Type'] = option_type
    return df[['strike', 'GEX', 'Vanna', 'Charm', 'IV_Raw', 'Option_Type']]

def calculate_max_pain_vectorized(opt_chain):
    try:
        calls = opt_chain.calls[['strike', 'openInterest']].dropna()
        puts = opt_chain.puts[['strike', 'openInterest']].dropna()
        strikes = np.array(sorted(set(calls['strike']) | set(puts['strike'])))
        if len(strikes) == 0: return None
        c_s, c_oi = calls['strike'].values, calls['openInterest'].values
        p_s, p_oi = puts['strike'].values, puts['openInterest'].values
        strikes_col = strikes[:, np.newaxis]
        call_loss = np.maximum(c_s - strikes_col, 0) * c_oi * 100
        put_loss = np.maximum(strikes_col - p_s, 0) * p_oi * 100
        return float(strikes[np.argmin(call_loss.sum(axis=1) + put_loss.sum(axis=1))])
    except: return None

# --- 3. METRIC SCALING COMPONENT ---
def format_scaled_exposure(val):
    abs_val = abs(val)
    sign = "-" if val < 0 else ""
    if abs_val >= 1e9: return f"{sign}${abs_val / 1e9:.2f}B"
    if abs_val >= 1e6: return f"{sign}${abs_val / 1e6:.2f}M"
    return f"{sign}${abs_val:,.0f}"

def format_scaled_shares(val):
    abs_val = abs(val)
    sign = "-" if val < 0 else ""
    if abs_val >= 1e6: return f"{sign}{abs_val / 1e6:.2f}M"
    return f"{sign}{abs_val:,.0f}"

# --- 4. DATA INGESTION ENGINE ---
with st.spinner("Executing Volatility Quant Matrices..."):
    try:
        stock = yf.Ticker(ticker_input)
        current_price = stock.fast_info.get('last_price')
        if current_price is None or np.isnan(current_price):
            current_price = stock.info.get('regularMarketPrice')
        if current_price is None or np.isnan(current_price):
            hist = stock.history(period="1d")
            if not hist.empty: current_price = float(hist['Close'].iloc[-1])
    except: current_price = None

if current_price is None or np.isnan(current_price):
    st.error(f"❌ Failed to extract price updates for {ticker_input}.")
    st.stop()

@st.cache_data(ttl=300)
def load_and_compute_gex_engine(ticker, r_rate, current_price):
    try:
        stock_obj = yf.Ticker(ticker)
        expirations = stock_obj.options
        if not expirations: return None, None, None
        
        near_chain = stock_obj.option_chain(expirations[0])
        atm_idx = (near_chain.calls['strike'] - current_price).abs().idxmin()
        atm_iv_now = float(near_chain.calls.loc[atm_idx, 'impliedVolatility'])
        max_pain_val = calculate_max_pain_vectorized(near_chain)
            
        compiled_dfs = []
        today = datetime.now().date()
        
        for exp_str in expirations:
            try:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                dte = (exp_date - today).days
                if dte <= 0 or dte > 45: continue
                dte_weight = max(0.1, (46 - dte) / 45.0)
                T = dte / 365.0
                opt_chain = stock_obj.option_chain(exp_str)
                c_res = process_chain_vectorized(opt_chain.calls, 'call', current_price, T, r_rate, dte_weight)
                p_res = process_chain_vectorized(opt_chain.puts, 'put', current_price, T, r_rate, dte_weight)
                if not c_res.empty: compiled_dfs.append(c_res)
                if not p_res.empty: compiled_dfs.append(p_res)
            except: continue
            
        if not compiled_dfs: return None, atm_iv_now, max_pain_val
        master_df = pd.concat(compiled_dfs, ignore_index=True)
        master_df = master_df[master_df.groupby('strike')['GEX'].transform('sum').abs() > 10000]
        
        agg_df = master_df.groupby('strike').agg({'GEX':'sum', 'Vanna':'sum', 'Charm':'sum', 'IV_Raw':'mean'}).reset_index()
        c_split = master_df[master_df['Option_Type']=='call'].groupby('strike')['GEX'].sum().rename('Call_GEX').reset_index()
        p_split = master_df[master_df['Option_Type']=='put'].groupby('strike')['GEX'].sum().rename('Put_GEX').reset_index()
        agg_df = pd.merge(agg_df, c_split, on='strike', how='left').fillna(0.0)
        agg_df = pd.merge(agg_df, p_split, on='strike', how='left').fillna(0.0)
        return agg_df, atm_iv_now, max_pain_val
    except: return None, None, None

data_matrix, atm_iv, max_pain_strike = load_and_compute_gex_engine(ticker_input, risk_free_rate, current_price)

# --- 5. DASHBOARD MAIN DISPLAY ---
if data_matrix is not None:
    agg_df_sorted = data_matrix.sort_values('strike').copy()
    agg_df_sorted['cumulative_GEX'] = agg_df_sorted['GEX'].cumsum()
    sign_changes = agg_df_sorted[(agg_df_sorted['cumulative_GEX'] * agg_df_sorted['cumulative_GEX'].shift(1) < 0)]
    zero_gamma_strike = sign_changes['strike'].iloc[0] if not sign_changes.empty else current_price

    lower_bound, upper_bound = current_price * (1.0 - zoom_pct), current_price * (1.0 + zoom_pct)
    filtered_df = data_matrix[(data_matrix['strike'] >= lower_bound) & (data_matrix['strike'] <= upper_bound)].copy()
    total_gex_dollar = data_matrix['GEX'].sum()
    total_gex_shares = total_gex_dollar / current_price
    pct_from_flip = (abs(current_price - zero_gamma_strike) / current_price)

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric(label=f"Price ({ticker_input})", value=f"${current_price:.2f}")
    col2.metric(label="Zero GEX Flip Node", value=f"${zero_gamma_strike:.2f}")
    col3.metric(label="Total Net GEX", value=format_scaled_exposure(total_gex_dollar))
    col4.metric(label="Total GEX (Shares)", value=format_scaled_shares(total_gex_shares))
    col5.metric(label=f"Implied Vol", value=f"{atm_iv * 100:.1f}%")
    
    with col6:
        if pct_from_flip <= 0.01: st.warning("⚡ TRANSITION")
        elif total_gex_dollar > 0: st.success("🟢 CALM REGIME")
        else: st.error("🔴 ACCELERATION")

    st.divider()
    st.subheader("📊 Cumulative Volatility Profile Architecture")
    chart_mode = st.radio("Display Profile Selection", ["Net GEX Profile", "Call / Put Split"], horizontal=True)
    
    fig = go.Figure()
    if chart_mode == "Net GEX Profile":
        fig.add_trace(go.Bar(
            x=filtered_df['strike'], y=filtered_df['GEX'],
            marker_color=np.where(filtered_df['GEX'] >= 0, '#2ecc71', '#e74c3c'),
            showlegend=False, hovertemplate="Strike: %{x}<br>GEX: $%{y:,.0f}<extra></extra>"
        ))
    else:
        fig.add_trace(go.Bar(
            x=filtered_df['strike'], y=filtered_df['Call_GEX'],
            marker_color='#2ecc71', name="Call GEX"
        ))
        fig.add_trace(go.Bar(
            x=filtered_df['strike'], y=filtered_df['Put_GEX'],
            marker_color='#e74c3c', name="Put GEX"
        ))
        fig.update_layout(barmode='group')
    
    df_s = filtered_df.sort_values('strike').copy()
    df_s['cum_GEX_display'] = df_s['GEX'].cumsum()
    fig.add_trace(go.Scatter(x=df_s['strike'], y=df_s['cum_GEX_display'], line=dict(color='#f1c40f', width=3), showlegend=False))
    
    fig.add_vline(x=current_price, line_dash="dash", line_color="#3498db", line_width=2)
    fig.add_vline(x=zero_gamma_strike, line_dash="dot", line_color="#9b59b6", line_width=2)
    if max_pain_strike is not None:
        fig.add_vline(x=max_pain_strike, line_dash="dot", line_color="#e67e22", line_width=2)
        
    fig.update_layout(template="plotly_dark", margin=dict(l=40, r=40, t=20, b=40), height=500)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.error("❌ Data matrix parsing execution failed.")
