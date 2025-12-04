import streamlit as st
import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# 1. SETUP & CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DealSignal: Take-Private Screener",
    page_icon="signal",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. DATA LOADING
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    try:
        # We read the standard file. Make sure you renamed your new CSV 
        # to 'enhanced_candidates.csv' before uploading to GitHub!
        df = pd.read_csv('enhanced_candidates.csv')
        
        # --- CLEANING: Force numeric types ---
        cols_to_clean = ['Conviction_Score', 'Implied EV', 'EBITDA', 'Buyout_Prob', 
                         'Analyst_Upside', 'Valuation/Revenue', 'Rule_of_40', 
                         'Rev_Growth', 'EBITDA Margin %']
        
        for col in cols_to_clean:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # --- CALCULATION: Safe EV/EBITDA ---
        df['EV_EBITDA_Display'] = df.apply(
            lambda row: row['Implied EV'] / row['EBITDA'] if (pd.notnull(row['EBITDA']) and row['EBITDA'] > 0) else np.nan, 
            axis=1
        )
        return df
    except Exception as e:
        st.error(f"Data Loading Error: {e}")
        return None

df = load_data()

if df is None:
    st.stop()

# -----------------------------------------------------------------------------
# 3. SIDEBAR FILTERS
# -----------------------------------------------------------------------------
st.sidebar.title("DealSignal Settings")

min_score = st.sidebar.slider("Min. Conviction Score", 0.0, 1.0, 0.5, 0.05)
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

# --- CUSTOM PROJECT OVERVIEW ---
st.markdown(
    """
    **Project Overview:** Identifying U.S. publicly listed software companies that exhibit 
    financial and market characteristics similar to firms taken private between 2018-2024.
    """
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Companies Screened", len(df))
col2.metric("Candidates Found", len(filtered_df))
col3.metric("Avg Probability", f"{filtered_df['Buyout_Prob'].mean():.1%}")
col4.metric("Median Upside", f"{filtered_df['Analyst_Upside'].median():.1%}")

tab1, tab2, tab3 = st.tabs(["🏆 Top Candidates", "📊 Visual Analysis", "🔍 Company Deep Dive"])

# --- TAB 1: TOP CANDIDATES ---
with tab1:
    st.subheader("Top Candidates")
    
    display_cols = [
        'Ticker', 'Company Name', 'Conviction_Score', 'Buyout_Prob', 
        'Analyst_Upside', 'Implied EV', 'EV_EBITDA_Display', 
        'Valuation/Revenue', 'Rule_of_40'
    ]
    
    valid_cols = [c for c in display_cols if c in filtered_df.columns]
    table_df = filtered_df.sort_values(by='Conviction_Score', ascending=False)[valid_cols]

    st.dataframe(
        table_df,
        use_container_width=True,
        height=600,
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", width="small"),
            "Company Name": st.column_config.TextColumn("Company", width="medium"),
            "Conviction_Score": st.column_config.ProgressColumn("Conviction Score", format="%.2f", min_value=0, max_value=1),
            "Buyout_Prob": st.column_config.NumberColumn("Buyout Prob", format="%.1f%%"),
            "Analyst_Upside": st.column_config.NumberColumn("Analyst Upside", format="%.1f%%"),
            "Implied EV": st.column_config.NumberColumn("Implied EV", format="$%.0f"),
            "EV_EBITDA_Display": st.column_config.NumberColumn("EV/EBITDA", format="%.1fx"),
            "Valuation/Revenue": st.column_config.NumberColumn("EV/Revenue", format="%.1fx"),
            "Rule_of_40": st.column_config.NumberColumn("Rule of 40", format="%.1f")
        },
        hide_index=True
    )

# --- TAB 2: VISUAL ANALYSIS ---
with tab2:
    import plotly.express as px
    col_chart_1, col_chart_2 = st.columns(2)
    
    with col_chart_1:
        st.subheader("Valuation vs. Profitability")
        if not filtered_df.empty:
            fig_scatter = px.scatter(
                filtered_df,
                x="EBITDA Margin %", y="Valuation/Revenue", size="Implied EV",
                color="Conviction_Score", hover_name="Company Name",
                color_continuous_scale="RdYlGn", title="EV/Revenue vs. EBITDA Margin"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
    with col_chart_2:
        st.subheader("Distribution of Valuation")
        if not filtered_df.empty:
            fig_hist = px.histogram(
                filtered_df, x="Valuation/Revenue", nbins=20, 
                title="EV / Revenue Distribution"
            )
            st.plotly_chart(fig_hist, use_container_width=True)

# --- TAB 3: DEEP DIVE ---
with tab3:
    if not filtered_df.empty:
        # Sort so highest conviction is first
        sorted_candidates = filtered_df.sort_values('Conviction_Score', ascending=False)
        
        # Selector
        selected_ticker = st.selectbox("Select Company", sorted_candidates['Ticker'].unique())
        
        # Get Data
        comp = filtered_df[filtered_df['Ticker'] == selected_ticker].iloc[0]
        
        # --- HEADER: Displays "Varonis Systems (VRNS)" ---
        # Because we are using the new CSV, comp['Company Name'] is already the full name!
        st.markdown(f"## {comp['Company Name']} ({comp['Ticker']})")
        
        dp1, dp2, dp3 = st.columns(3)
        with dp1:
            st.metric("Implied EV", f"${comp['Implied EV']/1e9:,.2f}B")
            st.metric("EV / Revenue", f"{comp['Valuation/Revenue']:.2f}x")
        with dp2:
            st.metric("Revenue", f"${comp['Revenue']/1e6:,.0f}M")
            st.metric("EBITDA Margin", f"{comp['EBITDA Margin %']:.1%}")
        with dp3:
            st.metric("Conviction Score", f"{comp['Conviction_Score']:.2f}")
            prob = max(0.0, min(1.0, float(comp['Buyout_Prob'])))
            st.progress(prob, text=f"Buyout Prob: {prob:.1%}")
    else:
        st.write("No candidates match filters.")
