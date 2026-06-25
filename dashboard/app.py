import streamlit as st
import sys
import os

# Add project root to sys.path to allow absolute imports of dashboard package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 1. Page Configuration (Must be first Streamlit command)
st.set_page_config(
    page_title="LLM Evaluation Platform Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply sleek styling adjustments (custom margins & styling indicators)
st.markdown("""
<style>
    .reportview-container {
        background-color: #fafbfc;
    }
    .stAlert {
        border-radius: 6px;
    }
    .st-emotion-cache-1kyxssb {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# 2. Imports after set_page_config
from dashboard.components.sidebar import render_sidebar
from dashboard.utils import fetch_experiments

# Import pages
from dashboard.pages.home import show_home_page
from dashboard.pages.experiments import show_experiments_page
from dashboard.pages.reports import show_reports_page
from dashboard.pages.comparisons import show_comparisons_page
from dashboard.pages.regression import show_regression_page
from dashboard.pages.failure_analysis import show_failure_analysis_page

def main():
    # Fetch all experiments (handles online REST API or fallback data automatically)
    experiments = fetch_experiments()
    
    # Render Sidebar and get currently selected page view
    selected_page = render_sidebar()
    
    # Routing Logic
    if selected_page == "🏠 Home":
        show_home_page(experiments)
    elif selected_page == "🧪 Experiments":
        show_experiments_page(experiments)
    elif selected_page == "📄 Reports":
        show_reports_page(experiments)
    elif selected_page == "📊 Comparisons":
        show_comparisons_page(experiments)
    elif selected_page == "⚠️ Regression Detection":
        show_regression_page(experiments)
    elif selected_page == "🤖 Failure Analysis":
        show_failure_analysis_page(experiments)

if __name__ == "__main__":
    main()
