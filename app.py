import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from datetime import datetime
import pytz

# --- 1. PAGE SETUP & CONFIGURATION ---
st.set_page_config(
    page_title="GEX Flow Dashboard", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- INJECTING ADHD CSS CUSTOM DESIGN THROUGH STREAMLIT SAFELY ---
st.markdown("""
<style>
    /* CSS custom styling rules are safely wrapped inside a Python string macro */
    .stAlert {
        animation: fadeIn 0.5s ease-in-out;
    }
    @keyframes fadeIn {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 SPY Structural Flow Dashboard")
st.caption("Tracking Options Market Maker Hedging Constraints.")

# --- 2. PERSISTENT LEFT SIDEBAR PANEL (WITH FINE-TUNING) ---
with st.sidebar:
    st.header("🎛️ Controls & Parameters")
    ticker_input = st.text_input(label="Target Ticker Symbol", value="SPY").upper()
    
    st.markdown("---")
    st.subheader("⚙️ Model Settings")
    zoom_pct = st.slider("Chart Zoom Window (±%)", min_value=3, max_value=15, value=6, step=1) / 100.0
    risk_free_rate = st.number_input("Risk-Free Rate (r)", value=0.05, step=0.01)
    
    st.markdown("### 🗓️ Expiration Tuning")
    min_dte = st.number_input("Minimum DTE Filter", min_value=0, max_value=10, value=0)
    max_dte = st.number_input("Maximum DTE Filter", min_value=11, max_value=90, value=45)

    # --- ADHD COGNITIVE OVERLAY / CHEAT SHEET PLAYBOOK ---
    st.markdown("---")
    st.subheader("🧠 2-Second Cheat Sheet Playbook")
    with st.container(border=True):
        st.markdown("""
        **🟢 Above Purple Line (Calm Zone):**
        * Market is stable.
        * Market maker programs buy dips and sell rallies.
        * Bias favors premium sellers and steady bounces.
        
        **🔴 Below Purple Line (Danger Zone):**
        * Slippery slope.
        * Market maker programs sell drops and follow momentum.
        * Expect wider swings and fast directional extension.
        
        **⚡ Stacking Directly On the Line:**
        * High variance zone.
        * Fast algorithmic cross-currents.
        * Hands off the keyboard until a breakout side is chosen!
        """)

# --- 3. HIGH-SPEED VECTORIZED QUANT ENGINE ---
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
        
        pdf_d1 = norm.pdf(d1)
        gamma = np.nan_to_num(pdf_d1 / (S * iv * np.sqrt(T)), nan=0.0, posinf=0.0)
        vanna = np.nan_to_num(-pdf_d1 * (d2 / iv), nan=0.0, posinf=0.0)
        charm = np.nan_to_num(
            pdf_d1 * (r_rate / (iv * np.sqrt(T)) - (d1 * d2) / (2 * T)) * (-1.0 / 365.0), 
            nan=0.0, posinf=0.0
        )
        
        sign = 1.0 if option_type == 'call' else -1.0
        
    df['GEX'] = oi * gamma * 100 * (S**2) * dte_weight * sign
    df['Vanna'] = oi * vanna * 100 * dte_weight   
    df['Charm'] = oi * charm * 100 * dte_weight   
    df['IV_Raw'] = iv
    df['Option_Type'] = option_type
    
    return df[['strike', 'GEX', 'Vanna', 'Charm', 'IV_Raw', 'Option_Type']]

# --- 4. VECTORIZED MAX PAIN ENGINE ---
def calculate_max_pain_vectorized(opt_chain):
    try:
        calls = opt_chain.calls[['strike', 'openInterest']].dropna()
        puts = opt_chain.puts[['strike', 'openInterest']].dropna()
        
        all_strikes = sorted(set(calls['strike']) | set(puts['strike']))
        strikes = np.array(all_strikes)
        if len(strikes) == 0:
            return None
            
        c_strikes, c_oi = calls['strike'].values, calls['openInterest'].values
        p_strikes, p_oi = puts['strike'].values, puts['openInterest'].values
        
        strikes_col = strikes[:, np.newaxis]
        call_loss = np.maximum(c_strikes - strikes_col, 0) * c_oi * 100
        put_loss = np.maximum(strikes_col - p_strikes, 0) * p_oi * 100
        
        total_loss = call_loss.sum(axis=1) + put_loss.sum(axis=1)
        return float(strikes[np.argmin(total_loss)])
    except Exception:
        return None

# --- 5. HUMAN-READABLE METRIC SCALING COMPONENT ---
def format_scaled_exposure(val):
    abs_val = abs(val)
    sign = "-" if val < 0 else ""
    if abs_val >= 1e9:
        return f"{sign}${abs_val / 1e9:.2f}B"
    elif abs_val >= 1e6:
        return f"{sign}${abs_val / 1e6:.2f}M"
    else:
        return f"{sign}${abs_val:,.0f}"

def format_scaled_shares(val):
    abs_val = abs(val)
    sign = "-" if val < 0 else ""
    if abs_val >= 1e6:
        return f"{sign}{abs_val / 1e6:.2f}M"
    else:
        return f"{sign}{abs_val:,.0f}"

# --- 6. DATA INGESTION ENGINE ---
with st.spinner("Executing Volatility Quant Matrices..."):
    try:
        stock = yf.Ticker(ticker_input)
        current_price = stock.fast_info.get('last_price')
        if current_price is None or np.isnan(current_price):
            current_price = stock.info.get('regularMarketPrice')
        if current_price is None or np.isnan(current_price):
            hist = stock.history(period="1d")
            if not hist.empty:
                current_price = float(hist['Close'].iloc[-1])
    except Exception:
        current_price = None

if current_price is None or np.isnan(current_price):
    st.error(f"❌ Failed to extract market price updates for {ticker_input}.")
    st.stop()

# FIX 3 — Round to 2dp so the cache key stays perfectly stable
price_key = round(current_price, 2)

@st.cache_data(ttl=300)
def load_and_compute_gex_engine(ticker, r_rate, target_price, min_d, max_d):
    try:
        stock_obj = yf.Ticker(ticker)
        expirations = stock_obj.options
        if not expirations:
            return None, None, None, None, 0.0
            
        try:
            near_chain = stock_obj.option_chain(expirations[0])
            atm_idx = (near_chain.calls['strike'] - target_price).abs().idxmin()
            atm_iv_now = float(near_chain.calls.loc[atm_idx, 'impliedVolatility'])
            max_pain_val = calculate_max_pain_vectorized(near_chain)
        except Exception:
            atm_iv_now = 0.20
            max_pain_val = None
            
        compiled_dfs = []
        today = datetime.now().date()
        
        for exp_str in expirations:
            try:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                dte = (exp_date - today).days
                if dte < min_d or dte > max_d:
                    continue
                    
                # FIX 1 — Scale weight dynamically based on user-defined max_dte
                dte_weight = max(0.1, (max_d + 1 - dte) / max(max_d, 1))
                
                # FIX 2 — Treat same-day/0-DTE options as 1-day minimum for Black-Scholes stability
                T = max(dte, 1) / 365.0
                
                opt_chain = stock_obj.option_chain(exp_str)
                call_res = process_chain_vectorized(opt_chain.calls, 'call', target_price, T, r_rate, dte_weight)
                put_res = process_chain_vectorized(opt_chain.puts, 'put', target_price, T, r_rate, dte_weight)
                
                if not call_res.empty: compiled_dfs.append(call_res)
                if not put_res.empty: compiled_dfs.append(put_res)
            except Exception:
                continue
            
        if not compiled_dfs:
            return None, atm_iv_now, max_pain_val, "No Compiled Data", 0.0
            
        master_df = pd.concat(compiled_dfs, ignore_index=True)
        
        # FIX 6 — Relative Noise Filter based dynamically on the asset's own GEX size floor
        gex_threshold = master_df.groupby('strike')['GEX'].transform('sum').abs()
        noise_floor = gex_threshold.max() * 0.001
        master_df = master_df[gex_threshold > noise_floor]
        
        agg_df = master_df.groupby('strike').agg({
            'GEX': 'sum', 'Vanna': 'sum', 'Charm': 'sum', 'IV_Raw': 'mean'
        }).reset_index()
        
        raw_call_split = master_df[master_df['Option_Type'] == 'call'].groupby('strike')['GEX'].sum().rename('Call_GEX').reset_index()
        raw_put_split = master_df[master_df['Option_Type'] == 'put'].groupby('strike')['GEX'].sum().rename('Put_GEX').reset_index()
        split_matrix = pd.merge(raw_call_split, raw_put_split, on='strike', how='outer').fillna(0.0)
        agg_df = pd.merge(agg_df, split_matrix, on='strike', how='left').fillna(0.0)
        
        est_tz = pytz.timezone('US/Eastern')
        fetch_timestamp = datetime.now(est_tz).strftime("%I:%M %p EST")
        
        # FIX 5 (Part A) — Retrieve Vanna historical indexing metrics
        try:
            vix = yf.Ticker("^VIX")
            vix_hist = vix.history(period="5d")
            vix_delta = float(vix_hist['Close'].iloc[-1] - vix_hist['Close'].iloc[0]) if len(vix_hist) >= 2 else 0.0
        except Exception:
            vix_delta = 0.0
            
        return agg_df, atm_iv_now, max_pain_val, fetch_timestamp, vix_delta
    except Exception:
        return None, None, None, None, 0.0

data_matrix, atm_iv, max_pain_strike, data_time, vix_delta_val = load_and_compute_gex_engine(
    ticker_input, risk_free_rate, price_key, min_dte, max_dte
)

# --- 7. DASHBOARD MAIN DISPLAY REGIME ---
if data_matrix is not None:
    # FIX 4 (Part A) — Compute unified full dataset cumulative sum before filtering
    agg_df_sorted = data_matrix.sort_values('strike').copy()
    agg_df_sorted['cumulative_GEX'] = agg_df_sorted['GEX'].cumsum()
    
    sign_changes = agg_df_sorted[(agg_df_sorted['cumulative_GEX'] * agg_df_sorted['cumulative_GEX'].shift(1) < 0)]
    zero_gamma_strike = sign_changes['strike'].iloc[0] if not sign_changes.empty else current_price

    lower_bound = current_price * (1.0 - zoom_pct)
    upper_bound = current_price * (1.0 + zoom_pct)
    
    # FIX 4 (Part B) — Slice the pre-calculated cumulative frame for direct chart distribution display
    filtered_df = agg_df_sorted[(agg_df_sorted['strike'] >= lower_bound) & (agg_df_sorted['strike'] <= upper_bound)].copy()

    total_gex_dollar = data_matrix['GEX'].sum()
    total_gex_shares = total_gex_dollar / current_price
    
    pct_from_flip = (abs(current_price - zero_gamma_strike) / current_price)
    is_approaching_zero = pct_from_flip <= 0.01

    # --- SCOREBOARD METRICS ROW ---
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric(label=f"Current {ticker_input} Price", value=f"${current_price:.2f}")
    with col2:
        st.metric(label="Tipping Point (Zero GEX)", value=f"${zero_gamma_strike:.2f}")
    with col3:
        st.metric(label="Total Net GEX", value=format_scaled_exposure(total_gex_dollar))
    with col4:
        st.metric(label="Total GEX (Shares)", value=format_scaled_shares(total_gex_shares))
    with col5:
        st.metric(label=f"Implied Vol (ATM)", value=f"{atm_iv * 100:.1f}%")
    with col6:
        if is_approaching_zero:
            st.warning("⚡ TRANSITION")
        elif total_gex_dollar > 0:
            st.success("🟢 CALM REGIME")
        else:
            st.error("🔴 ACCELERATION REGIME")

    # FIX 5 (Part B) — UI Vanna Tailwind/Headwind Structural Notification Bar
    total_vanna = data_matrix['Vanna'].sum()
    if vix_delta_val < -0.50 and total_vanna > 0:
        st.success("🚀 Vanna tailwind active — IV compressing, dealers buying delta mechanically")
    elif vix_delta_val > 0.50:
        st.error("⚠️ Vanna headwind — IV spiking, dealer delta unwinding in progress")

    st.divider()

    # --- ADHD VISUAL STRUCTURAL PLAYBOOK GUIDELINES ---
    st.subheader("🎯 Structural Playbook Guidelines")
    
    if is_approaching_zero:
        st.warning(f"🚨 **MODEL SIGNAL: TRANSITION ZONE RANGE INTRUSION.** Price is within {pct_from_flip*100:.1f}% of the calculated Zero-Gamma node (${zero_gamma_strike:.2f}).")
    else:
        st.info(f"ℹ️ **Tipping Point Proximity:** Price is currently {pct_from_flip*100:.1f}% away from the calculated Zero-Gamma Strike (${zero_gamma_strike:.2f}).")

    col_pb1, col_pb2 = st.columns(2)
    with col_pb1:
        st.markdown(f"""
        ### 🟢 ABOVE ESTIMATED ZERO-GAMMA (> ${zero_gamma_strike:.2f})
        * **The Alignment:** Historical bias favors lower systemic variance and compressed trading scales.
        * **Underlying Model Logic:** Options intermediaries balance inventory by counter-trend buying/selling.
        """)
    with col_pb2:
        st.markdown(f"""
        ### 🔴 BELOW ESTIMATED ZERO-GAMMA (< ${zero_gamma_strike:.2f})
        * **The Alignment:** Higher statistical tail risks and accelerated intraday expansion metrics.
        * **Underlying Model Logic:** Intermediaries chase momentum to maintain portfolio risk neutralization benchmarks.
        """)

    st.divider()

    # --- 8. PLOTLY CHART COMPONENT & LEGEND KEYS ---
    st.subheader("📊 Cumulative Volatility Profile Architecture")
    
    with st.container(border=True):
        st.markdown("""
        ### 🔑 Chart Legend Key
        * 🔵 **BLUE DASHED LINE** = Price Now: Current stock execution spot price.
        * 🟣 **PURPLE DOTTED LINE** = Estimated Zero-Gamma Node: Calculated across global option series.
        * 🟡 **YELLOW LINE** = Cumulative GEX Summation Profile: Tracking the running aggregate sum across strikes.
        """)
    
    chart_mode = st.radio("Display Profile Selection", ["Net GEX Profile", "Call / Put Distribution Split"], horizontal=True)
    
    fig = go.Figure()
    if chart_mode == "Net GEX Profile":
        fig.add_trace(go.Bar(
            x=filtered_df['strike'], 
            y=filtered_df['GEX'],
            marker_color=np.where(filtered_df['GEX'] >= 0, '#2ecc71', '#e74c3c'),
            showlegend=False, 
            hovertemplate="Strike: %{x}<br>Net GEX: $ %{y:,.0f}<extra></extra>"
        ))
    else:
        fig.add_trace(go.Bar(
            x=filtered_df['strike'], 
            y=filtered_df['Call_GEX'],
            marker_color='#2ecc71', 
            name="Call GEX",
            hovertemplate="Strike: %{x}<br>Call GEX: $ %{y:,.0f}<extra></extra>"
        ))
        fig.add_trace(go.Bar(
            x=filtered_df['strike'], 
            y=filtered_df['Put_GEX'],
            marker_color='#e74c3c', 
            name="Put GEX",
            hovertemplate="Strike: %{x}<br>Put GEX: $ %{y:,.0f}<extra></extra>"
        ))
        fig.update_layout(barmode='group')
    
    # FIX 4 (Part C) — Plot directly from pre-calculated global dataset frame
    fig.add_trace(go.Scatter(
        x=filtered_df['strike'], 
        y=filtered_df['cumulative_GEX'],
        line=dict(color='#f1c40f', width=3), 
        name="Cumulative GEX Summation"
    ))
    
    fig.add_vline(x=current_price, line_dash="dash", line_color="#3498db", line_width=2.5)
    fig.add_vline(x=zero_gamma_strike, line_dash="dot", line_color="#9b59b6", line_width=2.5)
    if max_pain_strike is not None:
        fig.add_vline(x=max_pain_strike, line_dash="dot", line_color="#e67e22", line_width=2.5)
        
    fig.update_layout(
        template="plotly_dark", 
        xaxis_title="Strike Price ($)", 
        yaxis_title="Running Exposure Capacity Sum ($)",
        margin=dict(l=40, r=40, t=20, b=40), 
        height=600, 
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- 9. QUANT BACKTESTING EXPORT ENGINE ---
    st.divider()
    st.subheader("💾 Export Flow Data For Offline Backtesting")
    
    csv_buffer = data_matrix.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export Full Structural Data Matrix to CSV",
        data=csv_buffer,
        file_name=f"{ticker_input}_GEX_Matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
else:
    st.error("❌ Data matrix parsing execution failed.")
