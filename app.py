import streamlit as st
import pandas as pd
from classify import classify
import os
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Configure page settings
st.set_page_config(
    page_title="Log Message Classification App",
    layout="wide",
    page_icon="🧠"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .st-bw {background-color: #ffffff;}
    .reportview-container .main .block-container {padding-top: 1rem;}
    .sidebar .sidebar-content {background-color: #ffffff;}
    h1 {color: #2b3e50;}
    .stDownloadButton button {width: 100%;}
    </style>
    """, unsafe_allow_html=True)

def show_visualizations(df):
    """Display interactive data visualizations"""
    st.subheader("📊 Classification Insights")
    
    # Create two columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Category Distribution")
        fig, ax = plt.subplots(figsize=(8,4))
        sns.countplot(
            y='target_label', 
            data=df, 
            order=df['target_label'].value_counts().index,
            palette="viridis"
        )
        plt.xlabel('Count')
        plt.ylabel('Category')
        st.pyplot(fig)

    with col2:
        st.markdown("### Source System Distribution")
        source_counts = df['source'].value_counts()
        fig, ax = plt.subplots(figsize=(8,4))
        ax.pie(
            source_counts, 
            labels=source_counts.index,
            autopct='%1.1f%%',
            colors=sns.color_palette("pastel")
        )
        st.pyplot(fig)

def debug_section(df, processing_time):
    """Show debug information"""
    with st.expander("🔧 Debug Details"):
        st.markdown("### Model Information")
        st.code("""
        Classification Pipeline:
        - Regex Pattern Matching (First Pass)
        - BERT Model: all-MiniLM-L6-v2 (embedding with sklearn algorithms)
        - LLM Model: deepseek-r1-distill-llama-70b (via Groq)
        """)
        
        st.markdown("### Performance Metrics")
        col1, col2 = st.columns(2)
        col1.metric("Total Logs Processed", len(df))
        col2.metric("Processing Time", f"{processing_time:.2f} sec")
        
        st.markdown("### Sample Classifications")
        st.dataframe(df[['source', 'log_message', 'target_label']].head(5))

# Main app layout
st.title("🧠 Log Message Classification App")
st.markdown("Classify log messages using a combination of Regex, BERT, and LLM techniques.")

# Sidebar configuration
with st.sidebar:
    st.header("📘 Instructions")
    st.markdown("""
    - Upload a CSV file with two columns: **`source`** and **`log_message`**
    - Click **Classify Logs** to run predictions
    - Download the output CSV once classification is complete
    - Try real-time classification below!
    """)
    st.info("Ensure your CSV file is properly formatted.", icon="✅")
    
    # Debug mode toggle
    debug_mode = st.checkbox("Enable Debug Mode", False)

# File upload section
uploaded_file = st.file_uploader("📤 Upload CSV File", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        if {"source", "log_message"}.issubset(df.columns):
            st.success("File uploaded successfully!")
            st.dataframe(df.head(), use_container_width=True)

            if st.button("🚀 Classify Logs"):
                start_time = time.time()
                
                with st.spinner("Classifying logs..."):
                    df["target_label"] = classify(list(zip(df["source"], df["log_message"])))
                    processing_time = time.time() - start_time
                    
                    # Save results
                    output_path = "resources/output.csv"
                    os.makedirs("resources", exist_ok=True)
                    df.to_csv(output_path, index=False)

                st.success("Classification complete!")
                
                # Show visualizations
                show_visualizations(df)
                
                # Download section
                st.download_button(
                    "📥 Download Result CSV",
                    data=open(output_path, "rb"),
                    file_name="classified_logs.csv",
                    use_container_width=True
                )

                # Debug information
                if debug_mode:
                    debug_section(df, processing_time)
        else:
            st.error("CSV must contain 'source' and 'log_message' columns.")
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

# Real-time classification
st.markdown("---")
st.header("🧪 Try Real-time Classification")

col1, col2 = st.columns(2)
with col1:
    sample_source = st.selectbox(
        "Source System",
        ["ModernCRM", "BillingSystem", "AnalyticsEngine", "ThirdPartyAPI", "ModernHR", "LegacyCRM"]
    )

with col2:
    sample_log = st.text_area("Enter a log message to classify", height=100)

if st.button("🔎 Predict Label"):
    if sample_log.strip():
        start_time = time.time()
        label = classify([(sample_source, sample_log.strip())])[0]
        processing_time = time.time() - start_time
        
        st.success(f"Predicted Label: **{label}**")
        
        if debug_mode:
            with st.expander("🔧 Debug Details"):
                st.markdown(f"""
                **Classification Pipeline:**
                - Source System: {sample_source}
                - Processing Time: {processing_time:.2f}s
                - Final Label: {label}
                """)
    else:
        st.warning("Please enter a log message to classify.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; font-size: 15px;'>
    Developed by <strong>Muhammad Hamza</strong> | Powered by LLMs, BERT, Regex<br><br> 
    <a href='https://github.com/mrhamxo' target='_blank'>
        <img src='https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg' width='30' style='margin-right: 10px;' />
    </a>
    <a href='https://www.linkedin.com/in/muhammad-hamza-khattak/' target='_blank'>
        <img src='https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg' width='30' style='margin-right: 10px;' />
    </a>
    <a href='mailto:mr.hamxa942@gmail.com'>
        <img src='https://upload.wikimedia.org/wikipedia/commons/4/4e/Mail_%28iOS%29.svg' width='30' />
    </a>
</div>
""", unsafe_allow_html=True)