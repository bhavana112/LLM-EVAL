import streamlit as st
from typing import List
from backend.experiments.models import Experiment
from dashboard.components.filters import render_experiment_selector
from dashboard.utils import get_experiment_report, get_regression_comparison

def show_regression_page(experiments: List[Experiment]):
    """Renders the Regression and Improvement Detection view."""
    st.title("⚠️ Regression Detection")
    st.markdown("Compare two evaluation runs to detect metric improvements, regressions, and response latency drifts.")
    st.markdown("---")
    
    if len(experiments) < 2:
        st.info("At least two experiments must be completed to perform regression detection analysis.")
        return
        
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Current Run")
        curr_exp = st.selectbox(
            label="Select Current Run",
            options=experiments,
            format_func=lambda e: f"{e.experiment_id} ({e.provider} - {e.model})",
            key="reg_current"
        )
    with col2:
        st.subheader("Benchmark / Reference Run")
        # Default to next experiment in list
        default_index = 1 if len(experiments) > 1 else 0
        ref_options = [e for e in experiments if e.experiment_id != curr_exp.experiment_id]
        
        if not ref_options:
            st.warning("No alternative runs available to set as benchmark.")
            return
            
        ref_exp = st.selectbox(
            label="Select Benchmark Run",
            options=ref_options,
            format_func=lambda e: f"{e.experiment_id} ({e.provider} - {e.model})",
            key="reg_reference"
        )
        
    st.markdown("---")
    
    if curr_exp and ref_exp:
        # Check dataset compatibility
        if curr_exp.dataset_name != ref_exp.dataset_name:
            st.error(
                f"**Incompatible Datasets!** Current run evaluated dataset `{curr_exp.dataset_name}` "
                f"but benchmark run evaluated `{ref_exp.dataset_name}`. Comparison cannot be calculated."
            )
            return
            
        try:
            curr_report = get_experiment_report(curr_exp)
            ref_report = get_experiment_report(ref_exp)
            
            comp = get_regression_comparison(curr_report, ref_report)
            
            # Display verdict banner
            verdict = comp.performance_verdict
            if verdict == "Better":
                st.success(f"### Performance Verdict: Better (🟢 Improved)")
                st.balloons()
            elif verdict == "Worse":
                st.error(f"### Performance Verdict: Worse (🔴 Regressed)")
            else:
                st.info(f"### Performance Verdict: Stable (⚪ Approximately the same)")
                
            st.markdown("---")
            
            # Display score deltas
            st.subheader("Comparison Deltas")
            d_col1, d_col2, d_col3 = st.columns(3)
            with d_col1:
                st.metric(
                    label="Overall Score Delta",
                    value=f"{comp.score_difference:+.2%}",
                    delta=f"{comp.score_difference:+.2%}"
                )
            with d_col2:
                st.metric(
                    label="Pass Rate Delta",
                    value=f"{comp.pass_rate_difference:+.2%}",
                    delta=f"{comp.pass_rate_difference:+.2%}"
                )
            with d_col3:
                st.metric(
                    label="Avg Latency Delta",
                    value=f"{comp.latency_difference:+.1f} ms",
                    delta=f"{comp.latency_difference:+.1f} ms",
                    delta_color="inverse"
                )
                
            st.markdown("---")
            
            # Detail change log
            st.subheader("Summary of Changes Detected")
            if comp.change_summary:
                for change in comp.change_summary:
                    if "regressed" in change or "increased" in change or "missing" in change:
                        st.markdown(f"🔴 {change}")
                    elif "improved" in change or "decreased" in change or "added" in change:
                        st.markdown(f"🟢 {change}")
                    else:
                        st.markdown(f"ℹ️ {change}")
            else:
                st.write("No performance deviations detected beyond tolerance limits.")
                
        except Exception as e:
            st.error(f"Failed to generate regression report: {str(e)}")
