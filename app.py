import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

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
# 2. DATA LOADING
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    """
    Loads the enhanced candidates CSV and performs minor cleaning/calculations
    for visualization purposes.
    """
    # Load the data provided in the prompt
    df = pd.read_csv('enhanced_candidates.csv')
    
    # Calculate EV/EBITDA for display (handling divide by zero/negatives)
    # We use a lambda to only calculate if EBITDA > 0, else NaN
    df['EV_EBITDA'] = df.apply(
        lambda row: row['Implied EV'] / row['EBITDA'] if row['EBITDA'] > 0 else np.nan, 
        axis=1
    )
    
    # Create a formatted label for the scatter plots
    df['Label'] = df['Ticker'] + " (" + df['Company Name'] + ")"
    
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("File 'enhanced_candidates.csv' not found. Please place the CSV file in the same directory.")
    st.stop()

# -----------------------------------------------------------------------------
# 3. SIDEBAR FILTERS
# -----------------------------------------------------------------------------
st.sidebar.title("DealSignal Settings")
st.sidebar.markdown("Filter the universe of software companies.")

# Filter by Conviction Score
min_score = st.sidebar.slider(
    "Min. Conviction Score",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05
)

# Filter by Market Cap / EV (Log EV in data, but let's use Implied EV for display)
min_ev, max_ev = st.sidebar.select_slider(
    "Implied EV Range ($B)",
    options=[0.1, 0.5, 1, 5, 10, 20, 50, 100],
    value=(0.1, 100)
)

# Apply Filters
# Note: Implied EV is in actual dollars, slider is in Billions
mask = (
    (df['Conviction_Score'] >= min_score) & 
    (df['Implied EV'] >= min_ev * 1e9) & 
    (df['Implied EV'] <= max_ev * 1e9)
)
filtered_df = df[mask].copy()

st.sidebar.markdown("---")
st.sidebar.info(
    "**Methodology Note**:\n\n"
    "**Conviction Score** = \n"
    "0.75 × Buyout Model Prob \n"
    "+ 0.25 × Analyst Upside Rank\n\n"
    ""
)

# -----------------------------------------------------------------------------
# 4. DASHBOARD LAYOUT
# -----------------------------------------------------------------------------

# Title Section
st.title("DealSignal: Future Take-Private Candidates")
st.markdown(
    """
    **Project Overview**: Identifying U.S. publicly listed software companies that exhibit 
    financial and market characteristics similar to firms taken private between 2018-2024.
    """
)

# Top Level Metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Companies Screened", f"{len(df)}")
with col2:
    st.metric("Candidates (Filtered)", f"{len(filtered_df)}")
with col3:
    avg_prob = filtered_df['Buyout_Prob'].mean()
    st.metric("Avg Buyout Probability", f"{avg_prob:.1%}")
with col4:
    med_upside = filtered_df['Analyst_Upside'].median()
    st.metric("Median Analyst Upside", f"{med_upside:.1%}")

st.markdown("---")

# TABS for different views
tab1, tab2, tab3 = st.tabs(["🏆 Top Candidates", "📊 Visual Analysis", "🔍 Company Deep Dive"])

# -----------------------------------------------------------------------------
# TAB 1: TOP CANDIDATES (The "Deal Screener" List)
# -----------------------------------------------------------------------------
with tab1:
    st.subheader(f"Top {len(filtered_df)} Candidates by Conviction Score")
    st.markdown("Matches the **'Top 25 Candidates'** format from the proposal.")

    # Prepare table for display
    display_cols = [
        'Ticker', 'Company Name', 'Conviction_Score', 'Buyout_Prob', 
        'Analyst_Upside', 'Implied EV', 'EV_EBITDA', 
        'Valuation/Revenue', 'Rule_of_40'
    ]
    
    # Sort by Conviction Score descending
    table_df = filtered_df.sort_values(by='Conviction_Score', ascending=False)[display_cols]
    
    # Formatting columns
    format_dict = {
        'Conviction_Score': '{:.2f}',
        'Buyout_Prob': '{:.1%}',
        'Analyst_Upside': '{:.1%}',
        'Implied EV': '${:,.0f}',
        'EV_EBITDA': '{:.1f}x',
        'Valuation/Revenue': '{:.1f}x',
        'Rule_of_40': '{:.1f}'
    }
    
    st.dataframe(
        table_df.style.format(format_dict).background_gradient(subset=['Conviction_Score'], cmap="Greens"),
        use_container_width=True,
        height=600
    )

# -----------------------------------------------------------------------------
# TAB 2: VISUAL ANALYSIS (Scatter Plots & Distributions)
# -----------------------------------------------------------------------------
with tab2:
    col_chart_1, col_chart_2 = st.columns(2)
    
    with col_chart_1:
        st.subheader("Valuation vs. Profitability")
        st.markdown("Replicating **'EV vs. EBITDA Margin'** Hypothesis")
        
        # Scatter: EV/Revenue vs EBITDA Margin (Common SaaS metric)
        # Sizing points by Implied EV to show scale
        fig_scatter = px.scatter(
            filtered_df,
            x="EBITDA Margin %",
            y="Valuation/Revenue",
            size="Implied EV",
            color="Conviction_Score",
            hover_name="Company Name",
            hover_data=["Ticker", "Buyout_Prob", "Rule_of_40"],
            color_continuous_scale="RdYlGn",
            title="EV/Revenue vs. EBITDA Margin (Bubble Size = EV)"
        )
        # Add a reference line for "fair value" or Rule of 40 trade-off could be added here
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with col_chart_2:
        st.subheader("Distribution of Valuation")
        st.markdown("Comparing Valuation Multiples in the identified set.")
        
        fig_hist = px.histogram(
            filtered_df,
            x="Valuation/Revenue",
            nbins=20,
            title="Distribution of EV / Revenue Multiples",
            color_discrete_sequence=['#3366CC']
        )
        fig_hist.add_vline(x=filtered_df['Valuation/Revenue'].median(), line_dash="dash", annotation_text="Median")
        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")
    
    st.subheader("Rule of 40 Analysis")
    st.markdown("Companies above the line are balancing Growth and Profitability effectively.")
    
    # Rule of 40 Scatter: Growth vs Margin
    fig_rule40 = px.scatter(
        filtered_df,
        x="EBITDA Margin %",
        y="Rev_Growth",
        color="Rule_of_40",
        hover_name="Ticker",
        title="Rule of 40: Revenue Growth vs. EBITDA Margin",
        labels={"Rev_Growth": "Revenue Growth (%)", "EBITDA Margin %": "EBITDA Margin (%)"}
    )
    # Add the Rule of 40 line (x + y = 0.40)
    # Note: Data in CSV seems to be decimals (0.10) or integers (40)? 
    # Checking CSV snippet: Rev_Growth is 0.10 (10%), Rule_of_40 is ~10-40.
    # We will assume decimal scale for axes.
    x_range = np.linspace(filtered_df['EBITDA Margin %'].min(), filtered_df['EBITDA Margin %'].max(), 100)
    y_rule40 = 0.40 - x_range
    
    fig_rule40.add_trace(go.Scatter(x=x_range, y=y_rule40, mode='lines', name='Rule of 40 Line', line=dict(dash='dash', color='red')))
    
    st.plotly_chart(fig_rule40, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 3: COMPANY DEEP DIVE (The "Deal Profile" View)
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("Deal Profile & Metrics")
    
    # Select box to choose a company
    # defaulting to the highest conviction score
    top_ticker = filtered_df.sort_values('Conviction_Score', ascending=False).iloc[0]['Ticker']
    selected_ticker = st.selectbox("Select a Company:", filtered_df['Ticker'].unique(), index=list(filtered_df['Ticker']).index(top_ticker))
    
    # Get data for selected company
    company_data = filtered_df[filtered_df['Ticker'] == selected_ticker].iloc[0]
    
    # Header
    st.markdown(f"## {company_data['Company Name']} ({company_data['Ticker']})")
    
    # 3-Column Layout for Key Stats (as seen in Proposal PDF "Deal Profile")
    dp_col1, dp_col2, dp_col3 = st.columns(3)
    
    with dp_col1:
        st.markdown("### 💰 Valuation")
        st.metric("Implied EV", f"${company_data['Implied EV']:,.0f}")
        st.metric("EV / Revenue", f"{company_data['Valuation/Revenue']:.2f}x")
        st.metric("EV / EBITDA", f"{company_data['EV_EBITDA']:.2f}x" if not pd.isna(company_data['EV_EBITDA']) else "N/A")

    with dp_col2:
        st.markdown("### 📈 Profitability & Growth")
        st.metric("Revenue", f"${company_data['Revenue']:,.0f}")
        st.metric("EBITDA Margin", f"{company_data['EBITDA Margin %']:.1%}")
        st.metric("Revenue Growth", f"{company_data['Rev_Growth']:.1%}")
        st.metric("Rule of 40", f"{company_data['Rule_of_40']:.1f}")

    with dp_col3:
        st.markdown("### 🎯 Model Output")
        st.progress(company_data['Buyout_Prob'], text=f"Buyout Probability: {company_data['Buyout_Prob']:.1%}")
        st.progress(company_data['Analyst_Upside'] if company_data['Analyst_Upside'] > 0 else 0, text=f"Analyst Upside: {company_data['Analyst_Upside']:.1%}")
        st.metric("Final Conviction Score", f"{company_data['Conviction_Score']:.2f}", delta="Rank Metric")

    st.markdown("---")
    st.info(f"**Investment Thesis Logic**: {company_data['Company Name']} has a buyout probability of **{company_data['Buyout_Prob']:.1%}** driven by its valuation compression and profitability profile, combined with an analyst upside of **{company_data['Analyst_Upside']:.1%}**.")