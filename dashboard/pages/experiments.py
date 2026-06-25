import streamlit as st
from typing import List
from backend.experiments.models import Experiment
from dashboard.components.tables import render_experiments_overview, render_test_cases_table
from dashboard.components.charts import render_pass_fail_distribution
from dashboard.components.filters import render_experiment_selector

def show_experiments_page(experiments: List[Experiment]):
    """Renders the Experiments log and drill-down details view."""
    st.title("🧪 Experiment Logs")
    st.markdown("Browse and search completed LLM evaluations. Drill down into individual runs to inspect outputs and test cases.")
    st.markdown("---")
    
    if not experiments:
        st.info("No experiments found in workspace storage.")
        return
        
    st.subheader("All Evaluation Runs")
    render_experiments_overview(experiments)
    
    st.markdown("---")
    
    st.subheader("🔍 Experiment Drill-down")
    selected_exp = render_experiment_selector(experiments, key_prefix="logs_drilldown")
    
    if selected_exp:
        # Display metadata
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**Experiment ID:** `{selected_exp.experiment_id}`")
            st.markdown(f"**Dataset:** `{selected_exp.dataset_name}`")
        with col2:
            st.markdown(f"**Provider:** `{selected_exp.provider.upper()}`")
            st.markdown(f"**Model:** `{selected_exp.model}`")
        with col3:
            st.markdown(f"**Timestamp:** `{selected_exp.timestamp.strftime('%Y-%m-%d %H:%M:%S')}`")
            
        st.markdown("**Configuration parameters:**")
        st.json(selected_exp.evaluation_configuration)
        
        st.markdown("---")
        
        # Display distribution & results table
        col_chart, col_empty = st.columns([1, 1])
        with col_chart:
            render_pass_fail_distribution(selected_exp)
            
        st.subheader("Evaluation Entries")
        render_test_cases_table(selected_exp.evaluation_results)
