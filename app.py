import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from datetime import datetime
import pytz

# --- 1. PAGE SETUP ---
st.set_page_config(
    page_title="GEX Flow Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @media (prefers-reduced-motion: no-preference) {
        .stAlert { animation: fadeIn 0.5s ease-in-out; }
        @keyframes fadeIn { 0% { opacity: 0; } 100% { opacity: 1; } }
    }
</style>
""", unsafe_allow_html=True)

st.warning("⚠️ For educational and research purposes only. Not financial advice.")
st.title("📊 Structural Flow & GEX Dashboard")
st.caption("Tracking Options Market Maker Hedging Constraints (Gamma, Vanna, Charm).")

# --- 2. SIDEBAR ---
with st.sidebar:
    st.header("🎛️ Controls & Parameters")
    ticker_input = st.text_input(label="Target Ticker Symbol", value="SPY").upper()

    st.markdown("---")
    st.subheader("⚙️ Model Settings")
    zoom_pct = st.slider("Chart Zoom Window (±%)", min_value=3, max_value=15, value=8, step=1) / 100.0
    risk_free_rate = st.number_input("Risk-Free Rate (r)", value=0.05, step=0.01)

    st.markdown("### 🗓️ Expiration Tuning")
    min_dte = st.number_input("Minimum DTE Filter", min_value=0, max_value=10, value=0)
    max_dte = st.number_input("Maximum DTE Filter", min_value=11, max_value=90, value=45)

    st.markdown("---")
    st.subheader("🧠 2-Second Cheat Sheet")
    with st.container(border=True):
        st.markdown("""
        **🟢 Above Purple Line (Calm Zone):**
        * Dealers buy dips, sell rallies mechanically.
        * Compressed ranges. Favor premium sellers.

        **🔴 Below Purple Line (Danger Zone):**
        * Dealers sell drops, buy rips — momentum amplifies.
        * Wide swings. Avoid selling blind premium.

        **⚡ On The Line:**
        * Structural gravity is offline.
        * Wait for a confirmed directional break.
        """)

# --- 3. VECTORIZED BLACK-SCHOLES ENGINE ---
def process_chain_vectorized(df, option_type, S, T, r_rate, dte_weight):
    df = df[['strike', 'openInterest', 'impliedVolatility']].copy()
    df = df[(df['openInterest'] > 0) & (df['impliedVolatility'] > 0)].copy()
    if df.empty:
        return pd.DataFrame()

    K  = df['strike'].values
    iv = df['impliedVolatility'].values
    oi = df['openInterest'].values

    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log(S / K) + (r_rate + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
        d2 = d1 - iv * np.sqrt(T)
        pdf_d1 = norm.pdf(d1)

        gamma = np.nan_to_num(pdf_d1 / (S * iv * np.sqrt(T)),                                nan=0.0, posinf=0.0)
        vanna = np.nan_to_num(-pdf_d1 * (d2 / iv),                                           nan=0.0, posinf=0.0)
        charm = np.nan_to_num(
            pdf_d1 * (r_rate / (iv * np.sqrt(T)) - (d1 * d2) / (2 * T)) * (-1.0 / 365.0),
            nan=0.0, posinf=0.0
        )
        sign = 1.0 if option_type == 'call' else -1.0

    df['GEX']         = oi * gamma * 100 * (S**2) * dte_weight * sign
    df['Vanna']       = oi * vanna * 100 * dte_weight
    df['Charm']       = oi * charm * 100 * dte_weight
    df['IV_Raw']      = iv
    df['Option_Type'] = option_type

    return df[['strike', 'GEX', 'Vanna', 'Charm', 'IV_Raw', 'Option_Type']]

# --- 4. VECTORIZED MAX PAIN ---
def calculate_max_pain_vectorized(opt_chain):
    try:
        calls = opt_chain.calls[['strike', 'openInterest']].dropna()
        puts  = opt_chain.puts[['strike', 'openInterest']].dropna()
        all_strikes = sorted(set(calls['strike']) | set(puts['strike']))
        strikes = np.array(all_strikes)
        if len(strikes) == 0:
            return None

        c_strikes, c_oi = calls['strike'].values, calls['openInterest'].values
        p_strikes, p_oi = puts['strike'].values,  puts['openInterest'].values

        strikes_col = strikes[:, np.newaxis]
        call_loss   = np.maximum(c_strikes - strikes_col, 0) * c_oi * 100
        put_loss    = np.maximum(strikes_col - p_strikes, 0) * p_oi * 100
        total_loss  = call_loss.sum(axis=1) + put_loss.sum(axis=1)
        return float(strikes[np.argmin(total_loss)])
    except Exception:
        return None

# --- 5. FORMATTING HELPERS ---
def format_scaled_exposure(val):
    abs_val = abs(val)
    sign = "-" if val < 0 else ""
    if abs_val >= 1e9:   return f"{sign}${abs_val / 1e9:.2f}B"
    elif abs_val >= 1e6: return f"{sign}${abs_val / 1e6:.2f}M"
    else:                return f"{sign}${abs_val:,.0f}"

def format_scaled_shares(val):
    abs_val = abs(val)
    sign = "-" if val < 0 else ""
    if abs_val >= 1e6: return f"{sign}{abs_val / 1e6:.2f}M"
    else:              return f"{sign}{abs_val:,.0f}"

# --- 6. PRICE FETCH WITH TRIPLE FALLBACK ---
with st.spinner("Fetching live market price..."):
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

if not current_price or (isinstance(current_price, float) and np.isnan(current_price)):
    st.error(f"❌ Failed to retrieve a valid market price for '{ticker_input}'. Check the ticker and try again.")
    st.stop()

price_key = round(current_price, 2)

# --- 7. MAIN CACHED GEX ENGINE ---
@st.cache_data(ttl=300)
def load_and_compute_gex_engine(ticker, r_rate, target_price, min_d, max_d):
    try:
        stock_obj   = yf.Ticker(ticker)
        expirations = stock_obj.options
        if not expirations:
            return None, None, None, None, 0.0

        chain_cache = {}

        # ATM IV + Max Pain from nearest expiry
        try:
            near_chain  = stock_obj.option_chain(expirations[0])
            chain_cache[expirations[0]] = near_chain
            atm_idx     = (near_chain.calls['strike'] - target_price).abs().idxmin()
            atm_iv_now  = float(near_chain.calls.loc[atm_idx, 'impliedVolatility'])
            max_pain_val = calculate_max_pain_vectorized(near_chain)
        except Exception:
            atm_iv_now   = 0.20
            max_pain_val = None

        compiled_dfs = []
        today = datetime.now().date()

        for exp_str in expirations:
            try:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                dte = (exp_date - today).days
                if dte < min_d or dte > max_d:
                    continue

                dte_weight = max(0.1, (max_d + 1 - dte) / max(max_d, 1))
                T = max(dte, 1) / 365.0

                opt_chain = chain_cache.get(exp_str) or stock_obj.option_chain(exp_str)

                call_res = process_chain_vectorized(opt_chain.calls, 'call', target_price, T, r_rate, dte_weight)
                put_res  = process_chain_vectorized(opt_chain.puts,  'put',  target_price, T, r_rate, dte_weight)

                if not call_res.empty: compiled_dfs.append(call_res)
                if not put_res.empty:  compiled_dfs.append(put_res)
            except Exception:
                continue

        if not compiled_dfs:
            return None, atm_iv_now, max_pain_val, "No Compiled Data", 0.0

        master_df = pd.concat(compiled_dfs, ignore_index=True)

        # Relative noise filter — adapts to ticker's own GEX scale
        gex_per_strike = master_df.groupby('strike')['GEX'].transform('sum').abs()
        noise_floor    = max(gex_per_strike.max() * 0.0001, 1.0)
        master_df      = master_df[gex_per_strike > noise_floor]

        agg_df = master_df.groupby('strike').agg({
            'GEX': 'sum', 'Vanna': 'sum', 'Charm': 'sum', 'IV_Raw': 'mean'
        }).reset_index()

        # Call / put split for grouped bar mode
        call_split   = master_df[master_df['Option_Type'] == 'call'].groupby('strike')['GEX'].sum().rename('Call_GEX').reset_index()
        put_split    = master_df[master_df['Option_Type'] == 'put'].groupby('strike')['GEX'].sum().rename('Put_GEX').reset_index()
        split_matrix = pd.merge(call_split, put_split, on='strike', how='outer').fillna(0.0)
        agg_df       = pd.merge(agg_df, split_matrix, on='strike', how='left').fillna(0.0)

        est_tz          = pytz.timezone('US/Eastern')
        fetch_timestamp = datetime.now(est_tz).strftime("%I:%M %p EST")

        # VIX 5-day trend — requires 3 trading rows minimum
        try:
            vix      = yf.Ticker("^VIX")
            vix_hist = vix.history(period="5d")
            vix_delta = float(vix_hist['Close'].iloc[-1] - vix_hist['Close'].iloc[0]) \
                        if len(vix_hist) >= 3 else 0.0
        except Exception:
            vix_delta = 0.0

        return agg_df, atm_iv_now, max_pain_val, fetch_timestamp, vix_delta

    except Exception:
        return None, None, None, None, 0.0

# --- 8. EXECUTE ENGINE ---
with st.spinner("Executing Vectorized GEX Quant Matrix..."):
    data_matrix, atm_iv, max_pain_strike, data_time, vix_delta_val = load_and_compute_gex_engine(
        ticker_input, risk_free_rate, price_key, min_dte, max_dte
    )

if data_matrix is None:
    st.error("❌ No options data matched your DTE filter range. Try widening Min/Max DTE in the sidebar.")
    st.stop()

# --- 9. DYNAMIC IV-SCALED ZERO-GAMMA FLIP DETECTION ---
# FIX: Use ATM IV to scale the strike window dynamically per ticker
# This prevents deep OTM puts from poisoning the cumulative sum
# e.g. SPY IV=15% → ±30% window | TSLA IV=55% → ±40% window (capped)
iv_window = max(0.10, min(0.40, atm_iv * 2.0))

agg_df_sorted = data_matrix.sort_values('strike').copy()

# Apply IV-scaled strike filter BEFORE computing cumulative sum
agg_df_sorted = agg_df_sorted[
    (agg_df_sorted['strike'] >= current_price * (1 - iv_window)) &
    (agg_df_sorted['strike'] <= current_price * (1 + iv_window))
]

agg_df_sorted['cumulative_GEX'] = agg_df_sorted['GEX'].cumsum()

# Find sign change closest to spot price (not just first sign change)
sign_changes = agg_df_sorted[
    (agg_df_sorted['cumulative_GEX'] * agg_df_sorted['cumulative_GEX'].shift(1) < 0) &
    agg_df_sorted['cumulative_GEX'].shift(1).notna()
]

if not sign_changes.empty:
    closest_flip_idx  = (sign_changes['strike'] - current_price).abs().idxmin()
    zero_gamma_strike = sign_changes.loc[closest_flip_idx, 'strike']
else:
    # Fallback: strike where cumulative GEX is closest to zero
    closest_idx       = agg_df_sorted['cumulative_GEX'].abs().idxmin()
    zero_gamma_strike = agg_df_sorted.loc[closest_idx, 'strike']

# Chart display window uses zoom slider independently of cumsum window
lower_bound = current_price * (1.0 - zoom_pct)
upper_bound = current_price * (1.0 + zoom_pct)
filtered_df = agg_df_sorted[
    (agg_df_sorted['strike'] >= lower_bound) &
    (agg_df_sorted['strike'] <= upper_bound)
].copy()

total_gex_dollar = data_matrix['GEX'].sum()
total_gex_shares = total_gex_dollar / price_key
total_vanna      = data_matrix['Vanna'].sum()
total_charm      = data_matrix['Charm'].sum()

pct_from_flip       = abs(current_price - zero_gamma_strike) / current_price
is_approaching_zero = pct_from_flip <= 0.01

# --- 10. HEADER INFO BAR ---
st.info(
    f"📅 Data as of {data_time}  |  "
    f"{ticker_input} ATM IV: {atm_iv * 100:.1f}%  |  "
    f"VIX 5d Δ: {vix_delta_val:+.2f}pts  |  "
    f"IV Strike Window: ±{iv_window*100:.0f}%"
)

# --- 11. SCOREBOARD METRICS ---
col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
    st.metric(label=f"{ticker_input} Spot Price",  value=f"${current_price:.2f}")
with col2:
    st.metric(label="Zero-GEX Flip Strike",        value=f"${zero_gamma_strike:.2f}")
with col3:
    st.metric(label="Total Net GEX ($)",           value=format_scaled_exposure(total_gex_dollar))
with col4:
    st.metric(label="Total GEX (Shares)",          value=format_scaled_shares(total_gex_shares))
with col5:
    st.metric(label="ATM Implied Vol",             value=f"{atm_iv * 100:.1f}%")
with col6:
    if is_approaching_zero:
        st.warning("⚡ TRANSITION")
    elif total_gex_dollar > 0:
        st.success("🟢 CALM REGIME")
    else:
        st.error("🔴 ACCEL REGIME")

st.divider()

# --- 12. VANNA BANNER WITH EXPLANATION ---
if vix_delta_val < -0.50 and total_vanna > 0:
    st.success(
        f"🚀 **Vanna Tailwind Active** — VIX dropped {abs(vix_delta_val):.2f}pts this week.\n\n"
        f"**What this means:** Vanna measures how much a dealer's delta exposure changes when "
        f"implied volatility moves. When IV *falls*, dealers who sold options find their hedges "
        f"are now over-hedged — they must *buy* the underlying stock to rebalance. "
        f"This creates a mechanical, reflexive buying bid underneath the market that can "
        f"lift prices even without fundamental news. The bigger the IV drop, the stronger the tailwind."
    )
elif vix_delta_val > 0.50:
    st.error(
        f"⚠️ **Vanna Headwind Active** — VIX rose {abs(vix_delta_val):.2f}pts this week.\n\n"
        f"**What this means:** When IV *rises*, dealers' hedges are now under-hedged — they must "
        f"*sell* the underlying stock to rebalance their delta exposure. This creates mechanical, "
        f"reflexive selling pressure that amplifies drops. Rising IV + negative Vanna flow is "
        f"one of the most dangerous structural combinations — it turns a normal selloff into a cascade."
    )




# --- 13. CHARM DAILY FLOW WITH EXPLANATION ---
charm_display   = format_scaled_exposure(total_charm)
charm_direction = "buying" if total_charm > 0 else "selling"

st.caption(
    f"⏱️ Daily Charm Flow: {charm_display} — "
    f"Charm measures how much dealer delta decays from time passing alone "
    f"(independent of price or vol moves). "
    f"Today's time decay forces dealers to mechanically {charm_direction} an estimated "
    f"{charm_display} worth of underlying stock purely to stay delta-neutral. "
    f"This flow happens every trading day on autopilot regardless of market direction."
)

# --- 14. STRUCTURAL PLAYBOOK ---
st.subheader("🎯 Structural Playbook")

if is_approaching_zero:
    st.warning(
        f"🚨 **TRANSITION ZONE** — Price is within {pct_from_flip*100:.2f}% of the Zero-Gamma node "
        f"(${zero_gamma_strike:.2f}). Structural gravity is offline. "
        f"Stand aside until price confirms a directional break above or below this level."
    )
else:
    regime = "above" if current_price > zero_gamma_strike else "below"
    st.info(
        f"ℹ️ Price is {pct_from_flip*100:.2f}% {regime} the Zero-Gamma Strike (${zero_gamma_strike:.2f})."
    )

col_pb1, col_pb2 = st.columns(2)
with col_pb1:
    st.markdown(f"""
    ### 🟢 Above Zero-Gamma (> ${zero_gamma_strike:.2f})
    * **What dealers do:** Net long gamma — they buy every dip and sell every rip to stay hedged.
    * **Market effect:** Price gets "pinned" — ranges compress, rebounds are quick, trends fade.
    * **Approved plays:** Sell OTM puts, cash-secured puts, iron condors, covered calls.
    """)
with col_pb2:
    st.markdown(f"""
    ### 🔴 Below Zero-Gamma (< ${zero_gamma_strike:.2f})
    * **What dealers do:** Net short gamma — they sell every drop and buy every rip to stay hedged.
    * **Market effect:** Moves get amplified — drops accelerate, bounces are weak and short-lived.
    * **Approved plays:** Long puts, long VIX calls, reduce or fully hedge long exposure.
    """)

st.divider()

# --- 15. GAMMA WALL CHART ---
st.subheader("📊 GEX Gamma Wall Profile")

with st.container(border=True):
    st.markdown("""
    **Chart Legend:**
    * 🔵 **Blue dashed** = Current spot price
    * 🟣 **Purple dotted** = Zero-gamma flip node (where cumulative GEX crosses zero — the structural tipping point)
    * 🟠 **Orange dotted** = Max pain strike (strike causing maximum loss to option buyers at expiry — acts as gravitational pin)
    * 🟡 **Yellow line** = Cumulative GEX summation (running total across strikes — zero crossing = flip node)
    """)

chart_mode = st.radio(
    "Display Mode",
    ["Net GEX Profile", "Call / Put Distribution Split"],
    horizontal=True
)

fig = go.Figure()

if chart_mode == "Net GEX Profile":
    fig.add_trace(go.Bar(
        x=filtered_df['strike'],
        y=filtered_df['GEX'],
        marker_color=np.where(filtered_df['GEX'] >= 0, '#2ecc71', '#e74c3c'),
        showlegend=False,
        hovertemplate="Strike: $%{x}<br>Net GEX: $%{y:,.0f}<extra></extra>"
    ))
else:
    fig.add_trace(go.Bar(
        x=filtered_df['strike'],
        y=filtered_df['Call_GEX'],
        marker_color='#2ecc71',
        name="Call GEX",
        hovertemplate="Strike: $%{x}<br>Call GEX: $%{y:,.0f}<extra></extra>"
    ))
    fig.add_trace(go.Bar(
        x=filtered_df['strike'],
        y=filtered_df['Put_GEX'],
        marker_color='#e74c3c',
        name="Put GEX",
        hovertemplate="Strike: $%{x}<br>Put GEX: $%{y:,.0f}<extra></extra>"
    ))
    fig.update_layout(barmode='group')

fig.add_trace(go.Scatter(
    x=filtered_df['strike'],
    y=filtered_df['cumulative_GEX'],
    line=dict(color='#f1c40f', width=3),
    name="Cumulative GEX"
))

fig.add_vline(x=current_price,     line_dash="dash", line_color="#3498db", line_width=2.5,
              annotation_text=" SPOT ", annotation_position="top right")
fig.add_vline(x=zero_gamma_strike, line_dash="dot",  line_color="#9b59b6", line_width=2.5,
              annotation_text=" FLIP NODE ", annotation_position="top left")
if max_pain_strike is not None:
    fig.add_vline(x=max_pain_strike, line_dash="dot", line_color="#e67e22", line_width=2.5,
                  annotation_text=" MAX PAIN ", annotation_position="bottom right")

fig.update_layout(
    template="plotly_dark",
    xaxis_title="Strike Price ($)",
    yaxis_title="GEX Exposure ($)",
    margin=dict(l=40, r=40, t=40, b=40),
    height=600,
    showlegend=True
)
st.plotly_chart(fig, use_container_width=True)

# --- 16. CSV EXPORT ---
st.divider()
st.subheader("💾 Export Data for Backtesting")

csv_buffer = data_matrix.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Export Full GEX Data Matrix to CSV",
    data=csv_buffer,
    file_name=f"{ticker_input}_GEX_Matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    mime="text/csv"
)
