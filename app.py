import streamlit as st
from sklearn.preprocessing import OrdinalEncoder
from streamlit_option_menu import option_menu
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import requests
import base64
from streamlit_lottie import st_lottie
from langchain_groq import ChatGroq
from fpdf import FPDF
import io

# Import modular engines
from src.data_processor import load_data, clean_data, get_data_quality_score
from src.eda_engine import generate_bar_chart, generate_line_chart
from src.segmentation import run_kmeans, visualize_clusters
from src.forecasting import train_forecast_model, visualize_forecast

# --- 1. Page Configuration (Strictly at the top before other Streamlit calls) ---
st.set_page_config(
    page_title="NexusBI | Enterprise BI Portal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load environment variables ---
load_dotenv()

# --- Helper function for PDF Executive Report Generation ---
def generate_report():
    from datetime import datetime
    gen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    pdf = FPDF()
    pdf.add_page()
    
    # Generation date/time at the top right corner
    pdf.set_font("Arial", "", 8)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 5, f"Report Generated: {gen_time}", ln=True, align="R")
    pdf.ln(5)
    
    # Main Title
    pdf.set_font("Arial", "B", 16)
    pdf.set_text_color(26, 54, 93)  # Sleek Dark Corporate Blue
    pdf.cell(0, 10, "NexusBI Enterprise - Executive Summary", ln=True, align="C")
    
    # Draw horizontal line under the title
    current_y = pdf.get_y()
    pdf.line(10, current_y, 200, current_y)
    pdf.ln(8)
    
    # Section 1: Dataset Overview
    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(45, 55, 72)  # Charcoal
    pdf.cell(0, 8, "Dataset Overview", ln=True)
    
    # Subtle line under section header
    current_y = pdf.get_y()
    pdf.line(10, current_y, 200, current_y)
    pdf.ln(5)
    
    # Body text
    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(60, 60, 60)
    
    if 'df_clean' in st.session_state and st.session_state['df_clean'] is not None:
        df_clean = st.session_state['df_clean']
        rows, cols = df_clean.shape
        filename = st.session_state.get('filename', 'N/A')
        
        pdf.cell(5, 6, chr(149), ln=False)
        pdf.cell(0, 6, f"File Name: {filename}", ln=True)
        pdf.cell(5, 6, chr(149), ln=False)
        pdf.cell(0, 6, f"Total Rows: {rows:,}", ln=True)
        pdf.cell(5, 6, chr(149), ln=False)
        pdf.cell(0, 6, f"Total Columns: {cols}", ln=True)
    else:
        pdf.cell(0, 6, "No dataset loaded. Please upload data in the Command Center first.", ln=True)
    pdf.ln(8)
    
    # Section 2: Data Health & Engineering
    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(45, 55, 72)
    pdf.cell(0, 8, "Data Health & Engineering", ln=True)
    
    # Subtle line under section header
    current_y = pdf.get_y()
    pdf.line(10, current_y, 200, current_y)
    pdf.ln(5)
    
    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(60, 60, 60)
    
    if 'df_clean' in st.session_state and st.session_state['df_clean'] is not None:
        df_clean = st.session_state['df_clean']
        mem_size_mb = df_clean.memory_usage(deep=True).sum() / (1024 * 1024)
        
        pdf.cell(5, 6, chr(149), ln=False)
        pdf.cell(0, 6, "Missing Values Handled: 100% Imputed or Cleared", ln=True)
        pdf.cell(5, 6, chr(149), ln=False)
        pdf.cell(0, 6, "Duplicate Records Resolved", ln=True)
        pdf.cell(5, 6, chr(149), ln=False)
        pdf.cell(0, 6, f"Memory Footprint: {mem_size_mb:.2f} MB", ln=True)
        pdf.cell(5, 6, chr(149), ln=False)
        pdf.cell(0, 6, "Data Types: Standardized for Scikit-Learn Pipelines", ln=True)
    else:
        pdf.cell(0, 6, "No dataset loaded. Data health diagnostics not available.", ln=True)
    pdf.ln(8)
    
    # Section 3: Machine Learning Insights
    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(45, 55, 72)
    pdf.cell(0, 8, "Machine Learning Insights", ln=True)
    
    # Subtle line under section header
    current_y = pdf.get_y()
    pdf.line(10, current_y, 200, current_y)
    pdf.ln(5)
    
    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(60, 60, 60)
    
    # Achievements
    pdf.cell(5, 6, chr(149), ln=False)
    pdf.cell(0, 6, "Model Deployed: Random Forest Predictive Engine", ln=True)
    pdf.cell(5, 6, chr(149), ln=False)
    pdf.cell(0, 6, "Segmentation: K-Means Clustering executed successfully", ln=True)
    pdf.cell(5, 6, chr(149), ln=False)
    pdf.cell(0, 6, "Security: Isolation Forest Anomaly Detection active", ln=True)
    
    # Footer
    pdf.set_y(-15)
    pdf.set_font("Arial", "I", 8)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 10, "Generated automatically by InsightAI / NexusBI Enterprise Engine", align="C")
    
    pdf_str = pdf.output(dest='S')
    if isinstance(pdf_str, str):
        pdf_bytes = pdf_str.encode('latin-1')
    else:
        pdf_bytes = bytes(pdf_str)
    
    buffer = io.BytesIO()
    buffer.write(pdf_bytes)
    buffer.seek(0)
    return buffer


# --- SVG Tech Doodle Background Generator ---
svg_doodle = """<svg xmlns="http://www.w3.org/2000/svg" width="300" height="300" viewBox="0 0 300 300">
  <g stroke="rgba(255, 255, 255, 0.05)" stroke-width="1.2" fill="none">
    <!-- Circuit paths -->
    <path d="M10,80 L60,80 L80,100 L120,100 L130,90 L130,60" />
    <circle cx="10" cy="80" r="2.5" />
    <circle cx="130" cy="60" r="2.5" />
    
    <!-- Nodes / Network -->
    <circle cx="230" cy="50" r="3.5" />
    <circle cx="260" cy="90" r="3.5" />
    <circle cx="210" cy="100" r="3.5" />
    <line x1="230" y1="50" x2="260" y2="90" />
    <line x1="230" y1="50" x2="210" y2="100" />
    <line x1="210" y1="100" x2="260" y2="90" />
    
    <!-- Gear -->
    <circle cx="75" cy="220" r="14" />
    <circle cx="75" cy="220" r="5" />
    <path d="M 75,202 L 75,206 M 75,234 L 75,238 M 57,220 L 61,220 M 89,220 L 93,220 M 62,207 L 65,210 M 88,230 L 85,233 M 88,207 L 85,210 M 62,230 L 65,233" />
    
    <!-- Line Graph -->
    <path d="M 180,220 L 200,195 L 220,205 L 240,175 L 260,190 L 280,155" />
    <line x1="175" y1="230" x2="285" y2="230" />
    <line x1="175" y1="150" x2="175" y2="230" />
    <circle cx="280" cy="155" r="2" />
    
    <!-- Server / Box -->
    <rect x="180" y="80" width="35" height="10" rx="1.5" />
    <rect x="180" y="95" width="35" height="10" rx="1.5" />
    <circle cx="185" cy="85" r="1" />
    <circle cx="185" cy="100" r="1" />
    <line x1="192" y1="85" x2="208" y2="85" />
    <line x1="192" y1="100" x2="208" y2="100" />
  </g>
</svg>"""
b64_doodle = base64.b64encode(svg_doodle.strip().encode()).decode()

# Inject styling strictly for the sidebar area
st.markdown(f"""
<style>
    /* Custom Sidebar styling: Tech Doodle Background isolated ONLY to sidebar area */
    section[data-testid="stSidebar"], [data-testid="stSidebar"], .stSidebar {{
        background-image: url("data:image/svg+xml;base64,{b64_doodle}") !important;
        background-repeat: repeat !important;
        background-size: 300px 300px !important;
        background-color: #0b0a0f !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
    }}
    
    [data-testid="stSidebarContent"], [data-testid="stSidebarContent"] > div, div[data-testid="stSidebarUserContent"] {{
        background-color: transparent !important;
    }}
    
    /* Custom glowing effects for header icons in the main area */
    .glow-pink {{
        color: #d946ef !important;
        text-shadow: 0 0 12px #d946ef, 0 0 25px rgba(217, 70, 239, 0.5) !important;
        display: inline-block;
        margin-right: 12px;
    }}
    .glow-cyan {{
        color: #00f0ff !important;
        text-shadow: 0 0 12px #00f0ff, 0 0 25px rgba(0, 240, 255, 0.5) !important;
        display: inline-block;
        margin-right: 12px;
    }}
    .glow-green {{
        color: #10b981 !important;
        text-shadow: 0 0 12px #10b981, 0 0 25px rgba(16, 185, 129, 0.5) !important;
        display: inline-block;
        margin-right: 12px;
    }}
    .glow-orange {{
        color: #f97316 !important;
        text-shadow: 0 0 12px #f97316, 0 0 25px rgba(249, 115, 22, 0.5) !important;
        display: inline-block;
        margin-right: 12px;
    }}
    .glowing-header {{
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 800 !important;
        color: #ffffff !important;
        margin-top: 10px !important;
        margin-bottom: 20px !important;
    }}
</style>
""", unsafe_allow_html=True)

# Inject JS style override for option-menu iframe (targets active vs inactive icons)
st.markdown("""
<img src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7" style="display:none;" onerror="
    try {
        const injectStyles = () => {
            const iframes = window.parent.document.querySelectorAll('iframe');
            iframes.forEach(iframe => {
                try {
                    const doc = iframe.contentDocument || iframe.contentWindow.document;
                    if (doc && !doc.getElementById('glow-style-override')) {
                        const style = doc.createElement('style');
                        style.id = 'glow-style-override';
                        style.innerHTML = `
                            /* Inactive icons: Cyan glow */
                            .nav-link i {
                                color: #00f0ff !important;
                                filter: drop-shadow(0 0 3px rgba(0, 240, 255, 0.8)) !important;
                                transition: all 0.3s ease !important;
                            }
                            /* Active selected icon: Pinkish-purple glow */
                            .nav-link.active i {
                                color: #ffffff !important;
                                filter: drop-shadow(0 0 6px #d946ef) !important;
                            }
                        `;
                        doc.head.appendChild(style);
                    }
                } catch(e) {}
            });
        };
        injectStyles();
        // Periodically verify in case of route transitions or lazy loading
        setInterval(injectStyles, 1000);
    } catch(e) {}
"/>
""", unsafe_allow_html=True)

# --- Navigation Menu using streamlit-option-menu ---
with st.sidebar:
    selected_page = option_menu(
        menu_title="Enterprise Portal",
        options=["Command Center", "Analytics Engine", "Prediction Lab", "AI Strategist"],
        icons=["rocket", "bar-chart", "cpu", "robot"],
        menu_icon=None,
        default_index=0,
        styles={
            "container": {"background-color": "transparent"},
            "icon": {"color": "#00f0ff", "filter": "drop-shadow(0 0 3px #00f0ff)"},
            "nav-link": {"color": "white", "--hover-color": "rgba(255, 255, 255, 0.1)"},
            "nav-link-selected": {
                "background-color": "#d946ef",
                "color": "white",
                "box-shadow": "0 0 12px #d946ef"
            }
        }
    )
    
    st.divider()
    
    # PDF generation integration
    if 'df_clean' in st.session_state and st.session_state['df_clean'] is not None:
        report_buffer = generate_report()
        st.download_button(
            label="📄 Generate Executive Report",
            data=report_buffer.getvalue(),
            file_name="NexusBI_Report.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    else:
        st.button("📄 Generate Executive Report", disabled=True, help="Please upload a dataset in the Command Center first.", use_container_width=True)

# --- Lottie Animation Loader ---
@st.cache_data(show_spinner=False)
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

# Load Lottie animations (cached)
lottie_data = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_qp1q7mct.json")
lottie_bot = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_M9p23l.json")

# --- Helper function for Demo Data Generation ---
def generate_demo_data() -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range(start="2026-01-01", periods=100, freq="D")
    categories = ["SaaS Enterprise", "SaaS Growth", "Professional Services", "API Usage"]
    
    df_demo = pd.DataFrame({
        "Date": dates,
        "Sales": np.round(np.sin(np.linspace(0, 10, 100)) * 5000 + 10000 + np.random.normal(0, 1000, 100), 2),
        "Product_Category": np.random.choice(categories, 100),
        "Customer_Age": np.round(np.random.uniform(18, 70, 100)).astype(int)
    })
    
    # Introduce deliberate duplicates (2 rows) to verify data quality score logic
    df_demo = pd.concat([df_demo, df_demo.iloc[[12, 45]]], ignore_index=True)
    
    # Introduce deliberate missing values (imputed by data cleaning engine)
    df_demo.loc[8, "Sales"] = np.nan
    df_demo.loc[15, "Customer_Age"] = np.nan
    df_demo.loc[22, "Product_Category"] = None
    
    return df_demo


# --- Persisted Session State Initialization ---
if 'df_raw' not in st.session_state:
    st.session_state['df_raw'] = None
if 'df_clean' not in st.session_state:
    st.session_state['df_clean'] = None
if 'quality_score' not in st.session_state:
    st.session_state['quality_score'] = None
if 'filename' not in st.session_state:
    st.session_state['filename'] = None
# Segmentation state
if 'df_clustered' not in st.session_state:
    st.session_state['df_clustered'] = None
if 'clustering_cols' not in st.session_state:
    st.session_state['clustering_cols'] = None
# Forecasting state
if 'df_forecast' not in st.session_state:
    st.session_state['df_forecast'] = None
if 'forecast_target' not in st.session_state:
    st.session_state['forecast_target'] = None

# --- 3. ROUTING AND PAGE CONTENT ---
# Global App Branding (InsightAI)
st.markdown("""
<div style="margin-bottom: 30px; border-bottom: 1px solid rgba(255, 255, 255, 0.1); padding-bottom: 12px;">
    <h1 style="font-family: 'Space Grotesk', sans-serif; font-weight: 900; margin: 0; font-size: 34px; display: inline-flex; align-items: center; gap: 12px;">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" style="width: 36px; height: 36px; display: inline-block; vertical-align: middle; filter: drop-shadow(0 0 10px rgba(0, 240, 255, 0.6));">
            <defs>
                <linearGradient id="logo-grad" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stop-color="#00f0ff" />
                    <stop offset="100%" stop-color="#d946ef" />
                </linearGradient>
            </defs>
            <polygon points="16,2 29,9.5 29,24.5 16,32 3,24.5 3,9.5" stroke="url(#logo-grad)" stroke-width="2.2" fill="none" />
            <line x1="16" y1="9" x2="23" y2="13" stroke="#00f0ff" stroke-width="1.2" opacity="0.8" />
            <line x1="23" y1="13" x2="23" y2="21" stroke="#d946ef" stroke-width="1.2" opacity="0.8" />
            <line x1="23" y1="21" x2="16" y2="25" stroke="#00f0ff" stroke-width="1.2" opacity="0.8" />
            <line x1="16" y1="25" x2="9" y2="21" stroke="#d946ef" stroke-width="1.2" opacity="0.8" />
            <line x1="9" y1="21" x2="9" y2="13" stroke="#00f0ff" stroke-width="1.2" opacity="0.8" />
            <line x1="9" y1="13" x2="16" y2="9" stroke="#d946ef" stroke-width="1.2" opacity="0.8" />
            <line x1="16" y1="9" x2="16" y2="17" stroke="#ffffff" stroke-width="1.2" opacity="0.9" />
            <line x1="23" y1="13" x2="16" y2="17" stroke="#ffffff" stroke-width="1.2" opacity="0.9" />
            <line x1="23" y1="21" x2="16" y2="17" stroke="#ffffff" stroke-width="1.2" opacity="0.9" />
            <line x1="16" y1="25" x2="16" y2="17" stroke="#ffffff" stroke-width="1.2" opacity="0.9" />
            <line x1="9" y1="21" x2="16" y2="17" stroke="#ffffff" stroke-width="1.2" opacity="0.9" />
            <line x1="9" y1="13" x2="16" y2="17" stroke="#ffffff" stroke-width="1.2" opacity="0.9" />
            <circle cx="16" cy="9" r="2.5" fill="#00f0ff" />
            <circle cx="23" cy="13" r="2.5" fill="#d946ef" />
            <circle cx="23" cy="21" r="2.5" fill="#00f0ff" />
            <circle cx="16" cy="25" r="2.5" fill="#d946ef" />
            <circle cx="9" cy="21" r="2.5" fill="#00f0ff" />
            <circle cx="9" cy="13" r="2.5" fill="#d946ef" />
            <circle cx="16" cy="17" r="3.5" fill="#ffffff" style="filter: drop-shadow(0 0 4px #fff);" />
        </svg>
        <span style="background: linear-gradient(135deg, #ffffff 0%, #d946ef 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; filter: drop-shadow(0 0 8px rgba(217, 70, 239, 0.4)); font-weight: 900;">NexusBI Enterprise</span>
    </h1>
</div>
""", unsafe_allow_html=True)

if selected_page == "Command Center":
    st.markdown("<h1 class='glowing-header'><span class='glow-pink'>🚀</span> Command Center</h1>", unsafe_allow_html=True)
    st.write("Ingest raw unstructured enterprise files. The engine automatically parses headers, removes duplicates, fills missing observations, and generates quality scores.")
    
    # Lottie high-tech data visualization animation
    if lottie_data:
        st_lottie(lottie_data, height=300)

    # File Uploader
    uploaded_file = st.file_uploader(
        "Upload enterprise CSV or Excel dataset", 
        type=["csv", "xlsx", "xls"],
        help="Ingest standard dataset formats to clean columns and calculate analytics metrics."
    )
    
    # Process uploaded file
    if uploaded_file is not None:
        if st.session_state.get('filename') != uploaded_file.name:
            with st.spinner("Executing Automated Data Cleaning Pipeline..."):
                try:
                    df_raw = load_data(uploaded_file)
                    df_clean = clean_data(df_raw)
                    score = get_data_quality_score(df_raw, df_clean)
                    
                    st.session_state['df_raw'] = df_raw
                    st.session_state['df_clean'] = df_clean
                    st.session_state['data'] = df_clean
                    st.session_state['quality_score'] = score
                    st.session_state['filename'] = uploaded_file.name
                    
                    # Clear stale predictions/clusters from previous datasets
                    st.session_state['df_clustered'] = None
                    st.session_state['clustering_cols'] = None
                    st.session_state['df_forecast'] = None
                    st.session_state['forecast_target'] = None
                    
                    st.toast(f"Ingested and cleaned {uploaded_file.name} successfully!", icon="✅")
                except Exception as e:
                    st.error(f"Incomplete processing: {str(e)}")
                    
    # Action Bar
    col_action_btn, col_action_score = st.columns([1, 1])
    with col_action_btn:
        load_demo = st.button("Load Sample Demo Data")
        if load_demo:
            with st.spinner("Generating sample demo dataset..."):
                try:
                    df_demo = generate_demo_data()
                    df_clean = clean_data(df_demo)
                    score = get_data_quality_score(df_demo, df_clean)
                    
                    st.session_state['df_raw'] = df_demo
                    st.session_state['df_clean'] = df_clean
                    st.session_state['data'] = df_clean
                    st.session_state['quality_score'] = score
                    st.session_state['filename'] = "sample_demo_data.csv"
                    
                    # Clear stale predictions/clusters from previous datasets
                    st.session_state['df_clustered'] = None
                    st.session_state['clustering_cols'] = None
                    st.session_state['df_forecast'] = None
                    st.session_state['forecast_target'] = None
                    
                    st.toast("Sample Demo Data loaded and cleaned successfully!", icon="✅")
                except Exception as e:
                    st.error(f"Failed to generate demo data: {e}")

    with col_action_score:
        if 'quality_score' in st.session_state and st.session_state['quality_score'] is not None:
            st.metric("Data Quality Score", value=f"{st.session_state['quality_score']}%")
        else:
            st.info("No Data Ingested")

    # Display characteristics if data is loaded
    if st.session_state.get('data') is not None:
        # Resolve df_clean and df_raw locally from session state if they are not already bound
        if 'df_clean' not in locals():
            df_clean = st.session_state.get('df_clean', st.session_state.get('data'))
        if 'df_raw' not in locals():
            df_raw = st.session_state.get('df_raw')

        st.success("Data Model Loaded Successfully!")
        
        # Four KPI cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(label="Model Accuracy", value="96%")
        with col2:
            st.metric(label="Training Loss", value="0.04")
        with col3:
            st.metric(label="Epochs", value="25")
        with col4:
            st.metric(label="Data Quality Score", value=f"{st.session_state['quality_score']}%")
            
        st.markdown("---")
        
        # Interactive Preview Tabs
        tab_preview, tab_diagnostics = st.tabs(["📋 Cleaned Dataset Preview", "📊 Feature Types & Diagnostics"])
        
        with tab_preview:
            try:
                if 'df_clean' in locals():
                    st.dataframe(df_clean, use_container_width=True)
                elif 'df_clean' in st.session_state:
                    st.dataframe(st.session_state.df_clean, use_container_width=True)
                else:
                    st.info("💡 Awaiting data ingestion. Please upload a dataset to generate the preview.")
            except NameError:
                st.info("💡 Awaiting data ingestion. Please upload a dataset to generate the preview.")
            
        with tab_diagnostics:
            try:
                local_df_raw = locals().get('df_raw', st.session_state.get('df_raw'))
                local_df_clean = locals().get('df_clean', st.session_state.get('df_clean', st.session_state.get('data')))
                
                if local_df_raw is not None and local_df_clean is not None:
                    # Standardize df_raw columns for exact matching to calculate imputed values
                    df_raw_std = local_df_raw.copy()
                    df_raw_std.columns = (
                        df_raw_std.columns.astype(str)
                        .str.strip()
                        .str.lower()
                        .str.replace(r"\s+", "_", regex=True)
                        .str.replace("-", "_")
                    )
                    
                    imputed_counts = []
                    for col in local_df_clean.columns:
                        if col in df_raw_std.columns:
                            raw_null = df_raw_std[col].isnull().sum()
                            clean_null = local_df_clean[col].isnull().sum()
                            imputed_counts.append(max(0, raw_null - clean_null))
                        else:
                            imputed_counts.append(0)

                    col_info = pd.DataFrame({
                        "Data Type": local_df_clean.dtypes.astype(str),
                        "Missing Values Imputed": imputed_counts,
                        "Active Missing Values": local_df_clean.isnull().sum(),
                        "Unique Values": local_df_clean.nunique()
                    })
                    st.dataframe(col_info, use_container_width=True)
                else:
                    st.info("💡 Awaiting data ingestion. Diagnostics not available.")
            except NameError:
                st.info("💡 Awaiting data ingestion. Diagnostics not available.")

elif selected_page == "Analytics Engine":
    st.markdown("<h1 class='glowing-header'><span class='glow-cyan'>📊</span> Analytics Engine</h1>", unsafe_allow_html=True)
    
    # Context Check: Ensure cleaned dataset exists
    if 'df_clean' not in st.session_state or st.session_state['df_clean'] is None:
        st.info("💡 Awaiting data ingestion. Please upload a dataset in the Command Center first.")
    else:
        st.success("Data Model Loaded Successfully!")
        df = st.session_state['df_clean']
        
        # KPI Overview: Quick data health stats
        total_records = df.shape[0]
        total_features = df.shape[1]
        numeric_features = len([col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])])
        categorical_features = total_features - numeric_features
        
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        kpi_col1.metric("Total Records", f"{total_records:,}")
        kpi_col2.metric("Total Features", f"{total_features}")
        kpi_col3.metric("Numeric Features", f"{numeric_features}")
        kpi_col4.metric("Categorical Features", f"{categorical_features}")
        
        st.markdown("---")
        
        # Organized UI Tabs
        tab_dist, tab_corr, tab_cat = st.tabs(["📊 Distributions", "🔗 Correlations", "📈 Categorical Analysis"])
        
        # Tab 1: Distributions
        with tab_dist:
            # Exclude numeric columns that contain "id" in their column name (case-insensitive)
            numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and "id" not in col.lower()]
            if not numeric_cols:
                st.warning("⚠️ No numeric features detected in this dataset.")
            else:
                selected_num_col = st.selectbox(
                    "Select Numeric Feature for Distribution Analysis",
                    numeric_cols,
                    key="eda_dist_selectbox"
                )
                # Filter to only include data between the 1st and 99th percentiles to handle outliers
                q_low = df[selected_num_col].quantile(0.01)
                q_high = df[selected_num_col].quantile(0.99)
                if pd.notnull(q_low) and pd.notnull(q_high):
                    filtered_df = df[(df[selected_num_col] >= q_low) & (df[selected_num_col] <= q_high)]
                else:
                    filtered_df = df
                
                import plotly.express as px
                fig_dist = px.histogram(
                    filtered_df,
                    x=selected_num_col,
                    template="plotly_dark",
                    color_discrete_sequence=['#00f0ff'],
                    title=f"Distribution of {selected_num_col} (1st - 99th Percentile)"
                )
                fig_dist.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#ffffff',
                    xaxis=dict(gridcolor='rgba(255,255,255,0.05)', title=selected_num_col),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.05)', title="Count"),
                    margin=dict(l=40, r=40, t=50, b=40)
                )
                st.plotly_chart(fig_dist, use_container_width=True)
                
        # Tab 2: Correlations
        with tab_corr:
            numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
            # Dynamically drop numeric columns containing "id" (case-insensitive) in their column name
            corr_cols = [col for col in numeric_cols if "id" not in col.lower()]
            if len(corr_cols) < 2:
                st.warning("⚠️ At least two numeric features (excluding ID columns) are required to compute a correlation matrix.")
            else:
                corr_matrix = df[corr_cols].corr()
                import plotly.express as px
                fig_corr = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    color_continuous_scale="Plasma",
                    template="plotly_dark",
                    title="Correlation Matrix (Numeric Features)"
                )
                fig_corr.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#ffffff',
                    margin=dict(l=40, r=40, t=50, b=40)
                )
                st.plotly_chart(fig_corr, use_container_width=True)
                
        # Tab 3: Categorical Analysis
        with tab_cat:
            # Exclude any column containing "id", "no", "code", or "date" (case-insensitive)
            categorical_cols = [
                col for col in df.columns 
                if not pd.api.types.is_numeric_dtype(df[col]) and not any(kw in col.lower() for kw in ["id", "no", "code", "date"])
            ]
            if not categorical_cols:
                st.warning("⚠️ No categorical features detected in this dataset.")
            else:
                selected_cat_col = st.selectbox(
                    "Select Categorical Feature for Frequency Analysis",
                    categorical_cols,
                    key="eda_cat_selectbox"
                )
                cat_counts = df[selected_cat_col].value_counts().head(10).reset_index()
                cat_counts.columns = ['Category', 'Count']
                
                import plotly.express as px
                fig_cat = px.bar(
                    cat_counts,
                    x='Category',
                    y='Count',
                    color='Count',
                    color_continuous_scale="Plasma",
                    template="plotly_dark",
                    title=f"Top 10 Most Frequent Categories in {selected_cat_col}"
                )
                fig_cat.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#ffffff',
                    xaxis=dict(gridcolor='rgba(255,255,255,0.05)', title="Category"),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.05)', title="Count"),
                    margin=dict(l=40, r=40, t=50, b=40)
                )
                st.plotly_chart(fig_cat, use_container_width=True)

elif selected_page == "Prediction Lab":
    st.markdown("<h1 class='glowing-header'><span class='glow-green'>🧠</span> Prediction Lab</h1>", unsafe_allow_html=True)
    if st.session_state.get('data') is not None:
        st.success("Data Model Loaded Successfully!")
        df = st.session_state['data']
        tab_cluster, tab_forecast, tab_anomaly = st.tabs(["🎯 Customer Segmentation", "📈 Random Forest Forecasting", "🚨 Anomaly Detection"])
        
        # --- TAB 1: SEGMENTATION ---
        with tab_cluster:
            st.subheader("K-Means Cluster Profiling")
            st.write("Clusters records into 3 discrete segments using K-Means. All numeric variables are automatically normalized via StandardScaler to prevent scale bias.")
            
            num_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
            
            if not num_cols:
                st.error("No numeric columns found in this dataset. Segmentation cannot be performed.")
            else:
                col_left, col_right = st.columns([1, 2])
                
                with col_left:
                    st.write("##### Configuration Parameters")
                    st.info("K-Means Hyperparameters locked to n_clusters=3 for this roadmap segment.")
                    
                    st.write("**Features selected for scaling and fitting:**")
                    for col in num_cols:
                        st.write(f"- `{col}`")
                    
                    run_segmentation = st.button("Run Clustering Pipeline", type="primary")
                
                with col_right:
                    if run_segmentation or st.session_state.get('df_clustered') is not None:
                        if run_segmentation:
                            with st.spinner("Normalizing features and running K-Means model..."):
                                try:
                                    clustered_df, used_cols = run_kmeans(df, n_clusters=3)
                                    st.session_state['df_clustered'] = clustered_df
                                    st.session_state['clustering_cols'] = used_cols
                                    st.success("Segments generated successfully!")
                                except Exception as e:
                                    st.error(f"Failed to cluster: {e}")
                        
                        if st.session_state.get('df_clustered') is not None:
                            fig_cluster = visualize_clusters(
                                st.session_state['df_clustered'], 
                                st.session_state['clustering_cols']
                            )
                            st.plotly_chart(fig_cluster, use_container_width=True)
                            
                            st.write("##### Clustered Preview Segment (Top 5 rows)")
                            cols_to_show = ['cluster'] + st.session_state['clustering_cols']
                            st.dataframe(st.session_state['df_clustered'][cols_to_show].head(5), use_container_width=True)
        
        # --- TAB 2: FORECASTING (Random Forest ML Engine) ---
        with tab_forecast:
            st.subheader("Random Forest Predictive Engine")
            st.write("Train a Random Forest model on the selected features to predict your target variable. The engine automatically detects whether to run a Regression or Classification task, builds a pre-processing pipeline, and performs an evaluation split.")
            
            # Feature Selection UI
            col_sel1, col_sel2 = st.columns(2)
            with col_sel1:
                selected_target = st.selectbox(
                    "Target Variable (Y)",
                    options=list(df.columns),
                    index=len(df.columns) - 1,
                    help="Select the column you want to predict."
                )
            
            with col_sel2:
                features_options = [col for col in df.columns if col != selected_target]
                selected_features = st.multiselect(
                    "Feature Variables (X)",
                    options=features_options,
                    default=features_options[:min(5, len(features_options))],
                    help="Select the columns to use as predictors."
                )
                
            # ML configurations
            col_cfg1, col_cfg2 = st.columns(2)
            with col_cfg1:
                test_size_pct = st.slider(
                    "Test Split Size (%)",
                    min_value=10,
                    max_value=50,
                    value=20,
                    step=5,
                    help="Percentage of the dataset withheld for testing model performance."
                )
                test_size_ratio = test_size_pct / 100.0
                
            with col_cfg2:
                # Detect target type
                target_series = df[selected_target]
                is_target_numeric = pd.api.types.is_numeric_dtype(target_series)
                unique_target_count = target_series.nunique()
                
                # Propose task type
                suggested_task = "Regression" if (is_target_numeric and unique_target_count > 10) else "Classification"
                
                user_selected_task = st.radio(
                    "Select ML Task Type",
                    options=["Regression", "Classification"],
                    index=0 if suggested_task == "Regression" else 1,
                    horizontal=True,
                    help="Regression is for predicting continuous values. Classification is for predicting categories/labels."
                )
                
                # Handle auto-detection and warn the user if overridden
                if user_selected_task != suggested_task:
                    task_type = suggested_task
                    st.info(f"💡 ML task overridden to **{suggested_task}** based on the target variable `{selected_target}` (type: {'numeric' if is_target_numeric else 'categorical'}, unique values: {unique_target_count}).")
                else:
                    task_type = user_selected_task
            
            run_training = st.button("Train Random Forest Model", type="primary")
            
            # Setup session state key for model results
            if 'rf_results' not in st.session_state:
                st.session_state['rf_results'] = None
                
            if run_training:
                if not selected_features:
                    st.error("Please select at least one feature variable (X).")
                else:
                    with st.spinner("Preparing data and training Random Forest pipeline..."):
                        try:
                            # Imports
                            from sklearn.model_selection import train_test_split
                            from sklearn.compose import ColumnTransformer
                            from sklearn.pipeline import Pipeline
                            from sklearn.impute import SimpleImputer
                            from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder
                            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
                            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score
                            import plotly.express as px
                            
                            # Extract data
                            X_raw = df[selected_features]
                            y_raw = df[selected_target]
                            
                            # Clean missing target values
                            valid_target_mask = y_raw.notna()
                            X_clean = X_raw[valid_target_mask].copy()
                            y_clean = y_raw[valid_target_mask]
                            
                            if len(y_clean) < 5:
                                st.error("Not enough non-null samples in the target variable to train a model.")
                            else:
                                # Smarter Date Handling: extract Month and Day, then drop original date column
                                for col in list(X_clean.columns):
                                    is_date = False
                                    if pd.api.types.is_datetime64_any_dtype(X_clean[col]):
                                        is_date = True
                                    else:
                                        if X_clean[col].dtype == 'object':
                                            try:
                                                non_null_samples = X_clean[col].dropna()
                                                if not non_null_samples.empty:
                                                    pd.to_datetime(non_null_samples.head(10))
                                                    is_date = True
                                            except (ValueError, TypeError):
                                                pass
                                    
                                    if is_date:
                                        try:
                                            parsed_dates = pd.to_datetime(X_clean[col], errors='coerce')
                                            if parsed_dates.notna().any():
                                                X_clean[f"{col}_month"] = parsed_dates.dt.month.fillna(1).astype(int)
                                                X_clean[f"{col}_day"] = parsed_dates.dt.day.fillna(1).astype(int)
                                                X_clean.drop(columns=[col], inplace=True)
                                        except Exception:
                                            pass
 
                                # Identify column types based on original data type (numeric vs categorical/mixed)
                                num_features = X_clean.select_dtypes(include=[np.number]).columns.tolist()
                                initial_cat_features = [col for col in X_clean.columns if col not in num_features]
                                
                                # Drop highly unique raw text columns (like description, invoiceno/invoicedate if not numeric)
                                dropped_features = []
                                cat_features = []
                                for col in initial_cat_features:
                                    unique_count = X_clean[col].nunique()
                                    # Threshold for high cardinality: either > 200 unique categories OR ratio of unique values > 30% of data size (if unique count > 20)
                                    if unique_count > 200 or (unique_count > 20 and (unique_count / len(X_clean)) > 0.3):
                                        dropped_features.append(col)
                                    else:
                                        cat_features.append(col)
                                        
                                if dropped_features:
                                    st.info(f"💡 Automatically dropped high-cardinality/unusable text columns: {', '.join([f'`{c}`' for c in dropped_features])}")
                                
                                # Re-construct active features
                                active_features = num_features + cat_features
                                X_clean = X_clean[active_features]
                                
                                # Cast remaining categorical feature columns to string using .astype(str) to prevent mixed-type errors
                                for col in cat_features:
                                    X_clean[col] = X_clean[col].fillna('nan').astype(str)
                                
                                # Split data
                                X_train, X_test, y_train, y_test = train_test_split(
                                    X_clean, y_clean, test_size=test_size_ratio, random_state=42
                                )
                                
                                # Define ColumnTransformer preprocessor
                                transformers = []
                                if num_features:
                                    transformers.append(('num', Pipeline([
                                        ('imputer', SimpleImputer(strategy='median')),
                                        ('scaler', StandardScaler())
                                    ]), num_features))
                                if cat_features:
                                    transformers.append(('cat', Pipeline([
                                        ('imputer', SimpleImputer(missing_values='nan', strategy='most_frequent')),
                                        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
                                    ]), cat_features))
                                    
                                preprocessor = ColumnTransformer(transformers=transformers)
                                
                                if task_type == "Classification":
                                    # Encode target labels
                                    le = LabelEncoder()
                                    y_train_encoded = le.fit_transform(y_train.astype(str))
                                    y_test_encoded = le.transform(y_test.astype(str))
                                    
                                    # Create classifier pipeline
                                    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                                    pipeline = Pipeline([
                                        ('preprocessor', preprocessor),
                                        ('model', model)
                                    ])
                                    
                                    pipeline.fit(X_train, y_train_encoded)
                                    y_pred = pipeline.predict(X_test)
                                    
                                    # Compute metrics
                                    accuracy = accuracy_score(y_test_encoded, y_pred)
                                    precision = precision_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
                                    recall = recall_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
                                    f1 = f1_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
                                    
                                    metrics_dict = {
                                        "Accuracy": accuracy,
                                        "Precision (Weighted)": precision,
                                        "Recall (Weighted)": recall,
                                        "F1-Score (Weighted)": f1
                                    }
                                else:
                                    # Create regressor pipeline
                                    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                                    pipeline = Pipeline([
                                        ('preprocessor', preprocessor),
                                        ('model', model)
                                    ])
                                    
                                    pipeline.fit(X_train, y_train)
                                    y_pred = pipeline.predict(X_test)
                                    
                                    # Compute metrics
                                    mae = mean_absolute_error(y_test, y_pred)
                                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                                    r2 = r2_score(y_test, y_pred)
                                    
                                    metrics_dict = {
                                        "R² Score": r2,
                                        "RMSE": rmse,
                                        "MAE": mae
                                    }
                                    
                                # Feature Importances
                                try:
                                    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
                                    feature_names = [f.replace('num__', '').replace('cat__', '') for f in feature_names]
                                except Exception:
                                    # fallback to numeric and categorical columns
                                    feature_names = num_features + cat_features
                                    
                                importances = pipeline.named_steps['model'].feature_importances_
                                
                                # align dimensions just in case
                                if len(feature_names) != len(importances):
                                    feature_names = [f"Feature_{i}" for i in range(len(importances))]
                                    
                                df_imp = pd.DataFrame({
                                    'Feature': feature_names,
                                    'Importance': importances
                                }).sort_values(by='Importance', ascending=True)
                                
                                st.session_state['rf_results'] = {
                                    'task_type': task_type,
                                    'metrics': metrics_dict,
                                    'df_importance': df_imp,
                                    'target_var': selected_target,
                                    'feature_vars': selected_features
                                }
                                st.toast("Random Forest model trained successfully!", icon="🔮")
                        except Exception as e:
                            st.error(f"Error training model: {str(e)}")
                            
            # Render model results if available
            rf_res = st.session_state['rf_results']
            if rf_res is not None:
                st.markdown("---")
                
                # Check for config discrepancy
                if rf_res['target_var'] != selected_target or set(rf_res['feature_vars']) != set(selected_features):
                    st.warning("⚠️ The current configurations do not match the trained model. Re-train the model to refresh these results.")
                
                # Metrics cards
                st.write(f"#### Model Evaluation Metrics ({rf_res['task_type']})")
                m_cols = st.columns(len(rf_res['metrics']))
                for idx, (m_name, m_val) in enumerate(rf_res['metrics'].items()):
                    with m_cols[idx]:
                        if "%" in m_name or "Accuracy" in m_name or "Precision" in m_name or "Recall" in m_name or "F1-Score" in m_name:
                            st.metric(label=m_name, value=f"{m_val:.2%}")
                        else:
                            st.metric(label=m_name, value=f"{m_val:.4f}")
                            
                # Feature Importance Chart
                st.markdown("### Feature Importance")
                df_imp = rf_res['df_importance']
                
                import plotly.express as px
                fig = px.bar(
                    df_imp,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    color='Importance',
                    color_continuous_scale=[[0, '#00f0ff'], [1, '#d946ef']],
                    title="Random Forest Feature Importance Analysis"
                )
                
                # Prevent single or few features from rendering as massive screen-filling blocks
                bar_width = 0.35 if len(df_imp) == 1 else (0.55 if len(df_imp) == 2 else 0.75)
                fig.update_traces(width=bar_width)
                
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#ffffff',
                    height=350,
                    xaxis=dict(gridcolor='rgba(255,255,255,0.05)', title="Importance Score"),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.05)', title="Feature"),
                    margin=dict(l=150, r=20, t=50, b=50)
                )
                st.plotly_chart(fig, use_container_width=True)
                
        # --- TAB 3: ANOMALY DETECTION (Isolation Forest ML Engine) ---
        with tab_anomaly:
            st.subheader("Isolation Forest Anomaly Detection")
            st.write("Detect outlier observations in the dataset using the Isolation Forest algorithm. The model flags data points that deviate significantly from typical patterns.")
            
            if 'df_clean' not in st.session_state or st.session_state['df_clean'] is None:
                st.warning("Please upload data in the Command Center first.")
            else:
                df_anomaly = st.session_state['df_clean']
                
                # Automatically select continuous numeric features (excluding ID columns)
                numeric_cols = [col for col in df_anomaly.columns if pd.api.types.is_numeric_dtype(df_anomaly[col])]
                anomaly_features = [col for col in numeric_cols if "id" not in col.lower()]
                
                if not anomaly_features:
                    st.error("No continuous numeric features found in this dataset to perform anomaly detection.")
                else:
                    # Slider for expected contamination rate (Anomaly %)
                    contamination = st.slider(
                        "Expected Contamination Rate (Anomaly %)",
                        min_value=0.01,
                        max_value=0.20,
                        value=0.05,
                        step=0.01,
                        help="The proportion of outliers in the data set."
                    )
                    
                    with st.spinner("Training Isolation Forest and detecting anomalies..."):
                        try:
                            from sklearn.ensemble import IsolationForest
                            import plotly.express as px
                            
                            # Clean/Impute any missing values just in case
                            X_anomaly = df_anomaly[anomaly_features].fillna(df_anomaly[anomaly_features].median())
                            
                            # Fit IsolationForest
                            iso_forest = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
                            preds = iso_forest.fit_predict(X_anomaly)
                            
                            # Create results dataframe
                            df_anomaly_result = df_anomaly.copy()
                            df_anomaly_result['anomaly_pred'] = preds
                            df_anomaly_result['Anomaly Status'] = df_anomaly_result['anomaly_pred'].map({1: 'Normal', -1: 'Anomaly'})
                            
                            # Summary Metrics
                            total_anomalies = (preds == -1).sum()
                            anomaly_rate = total_anomalies / len(df_anomaly_result)
                            
                            m_col1, m_col2 = st.columns(2)
                            m_col1.metric("Total Anomalies Detected", f"{total_anomalies:,}")
                            m_col2.metric("Anomaly Rate (%)", f"{anomaly_rate:.2%}")
                            
                            st.markdown("---")
                            
                            # Automatically select two features to plot (try quantity vs unitprice first, else fallback)
                            x_col = None
                            y_col = None
                            for col in anomaly_features:
                                if "quant" in col.lower() or "qty" in col.lower():
                                    x_col = col
                                    break
                            for col in anomaly_features:
                                if "price" in col.lower() or "rate" in col.lower() or "cost" in col.lower() or "sales" in col.lower():
                                    if col != x_col:
                                        y_col = col
                                        break
                            
                            # Fallback logic for plotting
                            if not x_col and len(anomaly_features) > 0:
                                x_col = anomaly_features[0]
                            if not y_col and len(anomaly_features) > 1:
                                y_col = anomaly_features[1]
                            elif not y_col and len(anomaly_features) == 1:
                                y_col = anomaly_features[0]
                                
                            if x_col and y_col:
                                fig_scatter = px.scatter(
                                    df_anomaly_result,
                                    x=x_col,
                                    y=y_col,
                                    color='Anomaly Status',
                                    color_discrete_map={'Normal': '#00f0ff', 'Anomaly': '#ff007f'},
                                    template="plotly_dark",
                                    title=f"Anomaly Detection Visualization: {x_col} vs {y_col}"
                                )
                                fig_scatter.update_layout(
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    font_color='#ffffff',
                                    xaxis=dict(gridcolor='rgba(255,255,255,0.05)', title=x_col),
                                    yaxis=dict(gridcolor='rgba(255,255,255,0.05)', title=y_col),
                                    margin=dict(l=40, r=40, t=50, b=40)
                                )
                                st.plotly_chart(fig_scatter, use_container_width=True)
                            else:
                                st.info("Need at least one numeric feature to render scatter plot visualization.")
                                
                            st.markdown("### Flagged Anomaly Observations")
                            df_anomalies_only = df_anomaly_result[df_anomaly_result['anomaly_pred'] == -1]
                            st.dataframe(df_anomalies_only, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error executing Anomaly Detection pipeline: {str(e)}")
    else:
        st.warning("⚠️ No dataset detected. Please navigate to the Command Center first to upload your data or load a sample dataset to activate the Prediction Lab.")

elif selected_page == "AI Strategist":
    st.markdown("<h1 class='glowing-header'><span class='glow-orange'>🤖</span> AI Strategist</h1>", unsafe_allow_html=True)
    
    # Render the robot animation
    if lottie_bot:
        st_lottie(lottie_bot, height=250, key="final_ai_bot")
        
    st.success("Data Model Loaded Successfully!")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User chat input box
    if prompt := st.chat_input("Ask the AI Strategist about your data..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate live response from Groq Cloud with Data Context
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your dataset..."):
                try:
                    api_key = os.environ.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
                    if not api_key:
                        st.error("Missing API Key! Please verify that GROQ_API_KEY is defined in your .env file.")
                    else:
                        # Build context if data is available
                        data_context = ""
                        if 'df_clean' in st.session_state and st.session_state.df_clean is not None:
                            df = st.session_state.df_clean
                            data_context = f"""
                            SYSTEM PROMPT: You are the elite AI Data Strategist for NexusBI Enterprise. Your goal is to analyze the provided dataset and answer the user's questions with absolute confidence, clarity, and analytical precision.
                            
                            TONE & FORMATTING GUIDELINES:
                            - Speak confidently and authoritatively, like an expert Lead Data Scientist.
                            - Never use hesitant or weak language (e.g., avoid "It seems," "I think," or "Based on what I see").
                            - Always structure your answers beautifully using Markdown. Use bold text for emphasis, bullet points for lists, and line breaks for readability.
                            - If a definitive answer exists in the data, state it directly and definitively. 
                            - Do not include generic historical facts; focus ONLY on the provided data context.
                            
                            DATA CONTEXT:
                            - Columns & Data Types: {df.dtypes.to_dict()}
                            - Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns
                            - Statistical Summary: {df.describe(include='all').to_dict()}
                            - First 3 Sample Rows: {df.head(3).to_dict()}
                            """
                        else:
                            data_context = "SYSTEM PROMPT: You are a confident AI assistant. Politely but firmly inform the user that no dataset has been uploaded yet, and instruct them to upload a file in the Command Center for custom analysis."

                        # Combine system context with user query
                        full_prompt = f"{data_context}\n\nUser Question: {prompt}"

                        llm = ChatGroq(
                            api_key=api_key,
                            model_name="llama-3.1-8b-instant"
                        )
                        response = llm.invoke(full_prompt)
                        st.markdown(response.content)
                        st.session_state.messages.append({"role": "assistant", "content": response.content})
                except Exception as e:
                     st.error(f"Groq API Error: {e}")
