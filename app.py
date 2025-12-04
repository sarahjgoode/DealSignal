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
        # 1. Load the new CSV
        df = pd.read_csv('candidates_with_annual_returns.csv')
        
        # 2. Clean Numeric Columns
        # Added 'Annual Return %' to the list
        cols_to_clean = ['Conviction_Score', 'Implied EV', 'EBITDA', 'Buyout_Prob', 
                         'Analyst_Upside', 'Valuation/Revenue', 'Rule_of_40', 
                         'Rev_Growth', 'EBITDA Margin %', 'Annual Return %']
        
        for col in cols_to_clean:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 3. Normalize Return % (if data is like 50.5 for 50.5%, convert to 0.505)
        # Assuming the CSV has whole numbers (e.g. -30.5) for percentages
        if 'Annual Return %' in df.columns:
             df['Annual Return %'] = df['Annual Return %'] / 100.0

        # 4. Calculate EV/EBITDA
        df['EV_EBITDA_Display'] = df.apply(
            lambda row: row['Implied EV'] / row['EBITDA'] if (pd.notnull(row['EBITDA']) and row['EBITDA'] > 0) else np.nan, 
            axis=1
        )

        # 5. Create Label
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

# Filter by Conviction Score
min_score = st.sidebar.slider("Min. Conviction Score", 0.0, 1.0, 0.5, 0.05)

# Filter by EV (Billions)
min_ev, max_ev = st.sidebar.select_slider(
    "Implied EV Range ($B)",
    options=[0.1, 0.5, 1, 2, 5, 10, 20, 50, 100],
    value=(0.1, 100)
)

# Filter by Annual Return %
# Handle min/max for the slider even if data is missing (fillna)
col_return = 'Annual Return %'
if col_return in df.columns:
    min_ret_data = df[col_return].min() if df[col_return].notna().any() else -1.0
    max_ret_data = df[col_return].max() if df[col_return].notna().any() else 1.0

    min_return, max_return = st.sidebar.slider(
        "Annual Return % Range",
        min_value=float(min_ret_data),
        max_value=float(max_ret_data),
        value=(float(min_ret_data), float(max_ret_data)),
        format="%.2f"
    )
else:
    # Fallback if column missing
    min_return, max_return = -1.0, 1.0

# Apply Filters
mask = (
    (df['Conviction_Score'] >= min_score) & 
    (df['Implied EV'] >= min_ev * 1e9) & 
    (df['Implied EV'] <= max_ev * 1e9)
)

if col_return in df.columns:
    mask = mask & (
        (df[col_return].fillna(0) >= min_return) & 
        (df[col_return].fillna(0) <= max_return)
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

# --- TAB 1: TOP CANDIDATES (Sorted by Conviction) ---
with tab1:
    st.subheader("Top Candidates")
    
    display_cols = [
        'Ticker', 'Company Name', 'Conviction_Score', 'Buyout_Prob', 
        'Analyst_Upside', 'Annual Return %', 'Implied EV', 'EV_EBITDA_Display', 
        'Valuation/Revenue', 'Rule_of_40'
    ]
    
    valid_cols = [c for c in display_cols if c in filtered_df.columns]
    
    # Sort by Conviction Score for the table
    table_df = filtered_df.sort_values(by='Conviction_Score', ascending=False)[valid_cols]

    try:
        format_dict = {
            'Conviction_Score': '{:.2f}',
            'Buyout_Prob': '{:.1%}',
            'Analyst_Upside': '{:.1%}',
            'Annual Return %': '{:.1%}',
            'Implied EV': '${:,.0f}',
            'EV_EBITDA_Display': '{:.1f}x',
            'Valuation/Revenue': '{:.1f}x',
            'Rule_of_40': '{:.1f}'
        }
        
        styled_df = table_df.style.format(format_dict)
        styled_df = styled_df.background_gradient(subset=['Conviction_Score'], cmap="Greens")
        st.dataframe(styled_df, use_container_width=True, height=600)
        
    except Exception as e:
        st.dataframe(table_df, use_container_width=True, height=600)

# --- TAB 2: VISUAL ANALYSIS ---
with tab2:
    col_chart_1, col_chart_2 = st.columns(2)
    
    with col_chart_1:
        st.subheader("Valuation vs. Profitability")
        if not filtered_df.empty:
            hover_cols = ["Annual Return %"] if "Annual Return %" in filtered_df.columns else []
            fig_scatter = px.scatter(
                filtered_df,
                x="EBITDA Margin %", 
                y="Valuation/Revenue", 
                size="Implied EV",
                color="Conviction_Score", 
                hover_name="Company Name",
                hover_data=hover_cols,
                color_continuous_scale="RdYlGn", 
                title="Valuation/Revenue vs. EBITDA Margin"
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

# --- TAB 3: DEEP DIVE (Sorted Alphabetically) ---
with tab3:
    if not filtered_df.empty:
        # 1. Sort ALPHABETICALLY for the dropdown
        alphabetical_candidates = filtered_df.sort_values('Company Name', ascending=True)
        
        # 2. Selector
        selected_ticker_display = st.selectbox(
            "Select Company", 
            alphabetical_candidates['Label'].unique()
        )
        
        # 3. Extract Ticker
        selected_ticker = selected_ticker_display.split(" (")[0]
        
        # 4. Get Data
        comp = filtered_df[filtered_df['Ticker'] == selected_ticker].iloc[0]
        
        st.markdown(f"## {comp['Company Name']} ({comp['Ticker']})")
        
        dp1, dp2, dp3 = st.columns(3)
        with dp1:
            st.metric("Implied EV", f"${comp['Implied EV']/1e9:,.2f}B")
            st.metric("EV / Revenue", f"{comp['Valuation/Revenue']:.2f}x")
            
            if 'Annual Return %' in comp:
                st.metric("Annual Return", f"{comp['Annual Return %']:.1%}")
                
        with dp2:
            st.metric("Revenue", f"${comp['Revenue']/1e6:,.0f}M")
            st.metric("EBITDA Margin", f"{comp['EBITDA Margin %']:.1%}")
            st.metric("Analyst Upside", f"{comp['Analyst_Upside']:.1%}")
        with dp3:
            st.metric("Conviction Score", f"{comp['Conviction_Score']:.2f}")
            prob = max(0.0, min(1.0, float(comp['Buyout_Prob'])))
            st.progress(prob, text=f"Buyout Prob: {prob:.1%}")
    else:
        st.write("No candidates match filters.")
