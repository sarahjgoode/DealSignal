import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- 1. SETUP & CONFIGURATION ---
st.set_page_config(
    page_title="DealSignal: Take-Private Screener",
    page_icon="signal",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. ROBUST DATA LOADING ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('enhanced_candidates.csv')
        
        # CLEANING: Force numeric columns to be numbers (not strings)
        # This fixes the "TypeError" in the background_gradient
        cols_to_clean = ['Conviction_Score', 'Implied EV', 'EBITDA', 'Buyout_Prob', 
                         'Analyst_Upside', 'Valuation/Revenue', 'Rule_of_40', 
                         'Rev_Growth', 'EBITDA Margin %']
        
        for col in cols_to_clean:
            if col in df.columns:
                # 'coerce' turns non-numbers into NaN (safe for math)
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # CALCULATION: Create EV/EBITDA safely
        df['EV_EBITDA_Display'] = df.apply(
            lambda row: row['Implied EV'] / row['EBITDA'] if (pd.notnull(row['EBITDA']) and row['EBITDA'] > 0) else np.nan, 
            axis=1
        )
        
        # LABEL: Create a label for charts
        df['Label'] = df['Ticker'] + " (" + df['Company Name'] + ")"
        
        return df
    except Exception as e:
        st.error(f"Data Loading Error: {e}")
        return None

df = load_data()

if df is None:
    st.stop()

# --- 3. SIDEBAR FILTERS ---
st.sidebar.title("DealSignal Settings")

# Filter by Conviction Score (Default 0.5)
min_score = st.sidebar.slider("Min. Conviction Score", 0.0, 1.0, 0.5, 0.05)

# Filter by EV (Billions)
min_ev, max_ev = st.sidebar.select_slider(
    "Implied EV Range ($B)",
    options=[0.1, 0.5, 1, 2, 5, 10, 20, 50, 100],
    value=(0.1, 100)
)

# Apply Filters
mask = (
    (df['Conviction_Score'] >= min_score) & 
    (df['Implied EV'] >= min_ev * 1e9) & 
    (df['Implied EV'] <= max_ev * 1e9)
)
filtered_df = df[mask].copy()

# --- 4. MAIN DASHBOARD ---
st.title("DealSignal: Take-Private Screener")

# Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Companies Screened", len(df))
col2.metric("Candidates Found", len(filtered_df))
col3.metric("Avg Probability", f"{filtered_df['Buyout_Prob'].mean():.1%}")
col4.metric("Median Upside", f"{filtered_df['Analyst_Upside'].median():.1%}")

# Tabs
tab1, tab2, tab3 = st.tabs(["🏆 Top Candidates", "📊 Visual Analysis", "🔍 Company Deep Dive"])

# --- TAB 1: ROBUST TOP CANDIDATES ---
with tab1:
    st.subheader("Top Candidates")
    
    # Columns to display
    display_cols = [
        'Ticker', 'Company Name', 'Conviction_Score', 'Buyout_Prob', 
        'Analyst_Upside', 'Implied EV', 'EV_EBITDA_Display', 
        'Valuation/Revenue', 'Rule_of_40'
    ]
    
    # Only keep columns that actually exist in the CSV
    valid_cols = [c for c in display_cols if c in filtered_df.columns]
    table_df = filtered_df.sort_values(by='Conviction_Score', ascending=False)[valid_cols]

    # STYLING LOGIC: "Try" to style it, but if it breaks, just show plain table
    try:
        # Define formatting
        format_dict = {
            'Conviction_Score': '{:.2f}',
            'Buyout_Prob': '{:.1%}',
            'Analyst_Upside': '{:.1%}',
            'Implied EV': '${:,.0f}',
            'EV_EBITDA_Display': '{:.1f}x',
            'Valuation/Revenue': '{:.1f}x',
            'Rule_of_40': '{:.1f}'
        }
        
        # Apply formatting
        styled_df = table_df.style.format(format_dict)
        
        # Try adding the color gradient
        # (This is the part that usually crashes on Cloud)
        styled_df = styled_df.background_gradient(subset=['Conviction_Score'], cmap="Greens")
        
        st.dataframe(styled_df, use_container_width=True, height=600)
        
    except Exception as e:
        # FALLBACK: If styling fails, show the plain table with a warning
        # st.warning(f"Note: Table styling disabled due to data format issue ({e})")
        st.dataframe(table_df, use_container_width=True, height=600)

# --- TAB 2: VISUAL ANALYSIS ---
with tab2:
    col_chart_1, col_chart_2 = st.columns(2)
    
    with col_chart_1:
        st.subheader("Valuation vs. Profitability")
        # Ensure we have data before plotting
        if not filtered_df.empty:
            fig_scatter = px.scatter(
                filtered_df,
                x="EBITDA Margin %", 
                y="Valuation/Revenue", 
                size="Implied EV",
                color="Conviction_Score", 
                hover_name="Company Name",
                color_continuous_scale="RdYlGn", 
                title="EV/Revenue vs. EBITDA Margin"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("No data to plot with current filters.")
        
    with col_chart_2:
        st.subheader("Distribution of Valuation")
        if not filtered_df.empty:
            fig_hist = px.histogram(
                filtered_df, 
                x="Valuation/Revenue", 
                nbins=20, 
                title="EV / Revenue Distribution"
            )
            st.plotly_chart(fig_hist, use_container_width=True)

# --- TAB 3: DEEP DIVE ---
with tab3:
    if not filtered_df.empty:
        # Sort so highest conviction is first
        sorted_candidates = filtered_df.sort_values('Conviction_Score', ascending=False)
        top_ticker = sorted_candidates.iloc[0]['Ticker']
        
        # Selector
        selected_ticker = st.selectbox("Select Company", sorted_candidates['Ticker'].unique())
        
        # Get Data
        comp = filtered_df[filtered_df['Ticker'] == selected_ticker].iloc[0]
        
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
            # Safe progress bar
            prob = max(0.0, min(1.0, float(comp['Buyout_Prob'])))
            st.progress(prob, text=f"Buyout Prob: {prob:.1%}")
    else:
        st.write("No candidates match filters.")
