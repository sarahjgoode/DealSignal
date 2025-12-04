import streamlit as st
import os

# 1. Print a message immediately to prove the app is running
st.title("Debug Mode: DealSignal")
st.write("✅ App successfully started.")

# 2. Try importing libraries one by one
try:
    import pandas as pd
    st.write("✅ Pandas imported successfully.")
    import plotly.express as px
    st.write("✅ Plotly imported successfully.")
    import numpy as np
    st.write("✅ Numpy imported successfully.")
except ImportError as e:
    st.error(f"❌ Library Import Failed: {e}")
    st.stop()

# 3. Check if the CSV file exists
file_name = 'enhanced_candidates.csv'
if os.path.exists(file_name):
    st.write(f"✅ File '{file_name}' found.")
else:
    st.error(f"❌ File '{file_name}' NOT found. Please check the spelling and capitalization in GitHub.")
    st.write("Files in current directory:", os.listdir())
    st.stop()

# 4. Try loading the data
try:
    df = pd.read_csv(file_name)
    st.write(f"✅ Data loaded. Rows: {len(df)}")
    st.dataframe(df.head())
except Exception as e:
    st.error(f"❌ Data Load Error: {e}")
    st.stop()

# 5. Try the basic Chart
try:
    st.write("Testing Chart...")
    fig = px.scatter(df, x=df.columns[0], y=df.columns[1], title="Test Chart")
    st.plotly_chart(fig)
    st.write("✅ Chart rendered successfully.")
except Exception as e:
    st.error(f"❌ Chart Error: {e}")
