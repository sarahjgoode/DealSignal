import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- FIX 1: Set Matplotlib Backend for Cloud ---
# This prevents "TclError" or display errors on headless servers
import matplotlib
matplotlib.use('Agg')

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DealSignal: Take-Private Screener",
    page_icon="signal",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. DATA LOADING (Updated for Robustness)
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('enhanced_candidates.csv')
        
        # --- FIX 2: Force Numeric Types ---
        # Ensure 'Conviction_Score' is a number (coercing errors to NaN)
        # This prevents TypeError in background_gradient
        cols_to_numeric = ['Conviction_Score', 'Implied EV', 'EBITDA', 
                           'Buyout_Prob', 'Analyst_Upside', 'Valuation/Revenue', 
                           'Rule_of_40', 'Rev_Growth', 'EBITDA Margin %']
        
        for col in cols_to_numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Calculate Display Metric
        df['EV_EBITDA_Display'] = df.apply(
            lambda row: row['Implied EV'] / row['EBITDA'] if (row['EBITDA'] > 0 and pd.notnull(row['EBITDA'])) else np.nan, 
            axis=1
        )
        
        # Label for charts
        df['Label'] = df['Ticker'] + " (" + df['Company Name'] + ")"
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_data()

if df is None:
    st.stop()

# -----------------------------------------------------------------------------
# 3. SIDEBAR FILTERS
# -----------------------------------------------------------------------------
st.sidebar.title("DealSignal Settings")

# Filter by Conviction Score
min_score = st.sidebar.slider("Min. Conviction Score", 0.0, 1.0, 0.5, 0.05)

# Filter by EV
min_ev, max_ev = st.sidebar.select_slider(
    "Implied EV Range ($B)",
    options=[0.1, 0.5, 1, 2, 5, 10, 20, 50, 100],
    value=(0.1, 100)
)

mask = (
    (df['Conviction_Score'] >= min_score) & 
    (df['Implied EV'] >= min_ev * 1e9) & 
    (df['Implied EV'] <= max_ev * 1e9)
)
filtered_df = df[mask].copy()

# -----------------------------------------------------------------------------
# 4. MAIN DASHBOARD
# -----------------------------------------------------------------------------
st.title("DealSignal: Take-Private Screener")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Companies Screened", len(df))
col2.metric("Candidates Found", len(filtered_df))
col3.metric("Avg Probability", f"{filtered_df['Buyout_Prob'].mean():.1%}")
col4.metric("Median Upside", f"{filtered_df['Analyst_Upside'].median():.1%}")

tab1, tab2, tab3 = st.tabs(["🏆 Top Candidates", "📊 Visual Analysis", "🔍 Deep Dive"])

with tab1:
    st.subheader("Top Candidates")
    
    display_cols = [
        'Ticker', 'Company Name', 'Conviction_Score', 'Buyout_Prob', 
        'Analyst_Upside', 'Implied EV', 'EV_EBITDA_Display', 
        'Valuation/Revenue', 'Rule_of_40'
    ]
    
    # Ensure we only select columns that exist
    valid_cols = [c for c in display_cols if c in filtered_df.columns]
    table_df = filtered_df.sort_values(by='Conviction_Score', ascending=False)[valid_cols]

    # --- FIX 3: Safe Styling ---
    # We apply the gradient, but if it fails (TypeError), we fallback to plain table
    try:
        st.dataframe(
            table_df.style.format({
                'Conviction_Score': '{:.2f}',
                'Buyout_Prob': '{:.1%}',
                'Analyst_Upside': '{:.1%}',
                'Implied EV': '${:,.0f}',
                'EV_EBITDA_Display': '{:.1f}x',
                'Valuation/Revenue': '{:.1f}x',
                'Rule_of_40': '{:.1f}'
            }).background_gradient(subset=['Conviction_Score'], cmap="Greens"),
            use_container_width=True,
            height=600
        )
    except Exception as e:
        # Fallback if styling fails
        st.warning(f"Could not apply color styling: {e}")
        st.dataframe(table_df, use_container_width=True)

with tab2:
    col_chart_1, col_chart_2 = st.columns(2)
    with col_chart_1:
        st.subheader("Valuation vs. Profitability")
        fig_scatter = px.scatter(
            filtered_df,
            x="EBITDA Margin %", y="Valuation/Revenue", size="Implied EV",
            color="Conviction_Score", hover_name="Company Name",
            color_continuous_scale="RdYlGn", title="EV/Revenue vs. EBITDA Margin"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with col_chart_2:
        st.subheader("Distribution of Valuation")
        fig_hist = px.histogram(filtered_df, x="Valuation/Revenue", nbins=20, title="EV / Revenue Distribution")
        st.plotly_chart(fig_hist, use_container_width=True)

with tab3:
    if not filtered_df.empty:
        # Sort so the highest score is the default
        sorted_candidates = filtered_df.sort_values('Conviction_Score', ascending=False)
        top_ticker = sorted_candidates.iloc[0]['Ticker']
        
        selected_ticker = st.selectbox("Select Company", sorted_candidates['Ticker'].unique())
        comp = filtered_df[filtered_df['Ticker'] == selected_ticker].iloc[0]
        
        st.markdown(f"## {comp['Company Name']} ({comp['Ticker']})")
        
        dp1, dp2, dp3 = st.columns(3)
        with dp1:
            st.metric("Implied EV", f"${comp['Implied EV']/1e9:,.2f}B")
            st.metric("EV / Revenue", f"{comp['Valuation/Revenue']:.2f}x")
        with dp2:
            st.metric("Revenue (LTM)", f"${comp['Revenue']/1e6:,.0f}M")
            st.metric("EBITDA Margin", f"{comp['EBITDA Margin %']:.1%}")
        with dp3:
            st.metric("Conviction Score", f"{comp['Conviction_Score']:.2f}")
            st.progress(max(0.0, min(1.0, float(comp['Buyout_Prob']))), text="Buyout Prob")
    else:
        st.write("No candidates match filters.")
