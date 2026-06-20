This is the most complete version yet. Let me do a full evaluation.
  ::view-transition-group(*),
  ::view-transition-old(*),
  ::view-transition-new(*) {
    animation-duration: 0.25s;
    animation-timing-function: cubic-bezier(0.19, 1, 0.22, 1);
  }
VvisualizeVvisualize show_widgetHere are the concrete fixes for the four critical bugs:
Fix 1 — DTE weight breaks when user sets max_dte above 45
python# Current — goes negative when dte=60, max_dte=90
dte_weight = max(0.1, (46 - dte) / 45.0)

# Fixed — scales dynamically to whatever the user sets
dte_weight = max(0.1, (max_d + 1 - dte) / max(max_d, 1))
Fix 2 — T=0 causes silent BS division by zero
python# Current — dte=0 produces T=0, nan_to_num silently zeros all greeks
T = dte / 365.0

# Fixed — treat same-day as 1-day minimum for BS stability
T = max(dte, 1) / 365.0
Fix 3 — Float price busts cache every run
python# Current — float equality is fragile as a cache key
data_matrix, atm_iv, max_pain_strike, data_time = load_and_compute_gex_engine(
    ticker_input, risk_free_rate, current_price, min_dte, max_dte
)

# Fixed — round to 2dp so the cache key is stable
price_key = round(current_price, 2)
data_matrix, atm_iv, max_pain_strike, data_time = load_and_compute_gex_engine(
    ticker_input, risk_free_rate, price_key, min_dte, max_dte
)
Fix 4 — Cumulative GEX scope mismatch (flip line vs yellow line)
python# Current — flip computed on full data_matrix, chart line on filtered_df
# These cross zero at different strikes — the purple line and yellow line disagree

# Fixed — compute full cumsum once, then slice for display
agg_df_sorted = data_matrix.sort_values('strike').copy()
agg_df_sorted['cumulative_GEX'] = agg_df_sorted['GEX'].cumsum()

# Find flip on the full dataset
sign_changes = agg_df_sorted[
    agg_df_sorted['cumulative_GEX'] * agg_df_sorted['cumulative_GEX'].shift(1) < 0
]
zero_gamma_strike = sign_changes['strike'].iloc[0] if not sign_changes.empty else current_price

# Slice for chart display — carries the same cumulative values
df_display = agg_df_sorted[
    (agg_df_sorted['strike'] >= lower_bound) & 
    (agg_df_sorted['strike'] <= upper_bound)
].copy()
# Now df_display['cumulative_GEX'] already has the full-dataset cumsum — plot directly
Fix 5 — Restore Vanna tailwind (regression from v5)
This was the most theoretically authentic piece of the dashboard and was silently dropped. Add back to the data engine:
python# At end of load_and_compute_gex_engine, add:
try:
    vix = yf.Ticker("^VIX")
    vix_hist = vix.history(period="5d")
    vix_delta = float(vix_hist['Close'].iloc[-1] - vix_hist['Close'].iloc[0]) \
                if len(vix_hist) >= 2 else 0.0
except Exception:
    vix_delta = 0.0

return agg_df, atm_iv_now, max_pain_val, fetch_timestamp, vix_delta

# Then in the UI section, above the playbook:
total_vanna = data_matrix['Vanna'].sum()
if vix_delta < -0.50 and total_vanna > 0:
    st.success("🚀 Vanna tailwind active — IV compressing, dealers buying delta mechanically")
elif vix_delta > 0.50:
    st.error("⚠️ Vanna headwind — IV spiking, dealer delta unwinding in progress")
Fix 6 — Relative noise filter
python# Current — $10k is meaningless for SPY (billions in GEX) but kills small caps
master_df = master_df[master_df.groupby('strike')['GEX'].transform('sum').abs() > 10000]

# Fixed — relative to the ticker's own GEX scale
gex_threshold = master_df.groupby('strike')['GEX'].transform('sum').abs()
noise_floor = gex_threshold.max() * 0.001  # drop an
