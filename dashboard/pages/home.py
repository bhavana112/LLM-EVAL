import streamlit as st
from typing import List
from backend.experiments.models import Experiment
from dashboard.components.metrics import render_kpi_cards
from dashboard.components.charts import render_score_trends, render_latency_trends
from dashboard.utils import get_experiment_report, get_regression_comparison

def show_home_page(experiments: List[Experiment]):
    """Renders the Home dashboard view."""
    st.title("🏠 Overview Dashboard")
    st.markdown("A high-level summary of LLM evaluation experiments, metrics performance, and system latencies.")
    st.markdown("---")
    
    if not experiments:
        st.info("No experiments found. Run evaluations or start the backend to populate data.")
        return

    # 1. Compute summary stats across all runs
    total_exps = len(experiments)
    latest_exp = experiments[0]  # sorted reverse-chronological by default
    
    # Average score
    all_scores = []
    for e in experiments:
        vals = list(e.evaluation_metrics.values())
        if vals:
            all_scores.append(sum(vals) / len(vals))
    avg_overall_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
    
    # Average latency
    avg_latency = sum(e.average_latency for e in experiments) / total_exps
    total_cases = sum(e.total_number_of_test_cases for e in experiments)
    
    # Delta comparison for latest experiment against the baseline run (previous run)
    score_delta = 0.0
    latency_delta = 0.0
    if len(experiments) >= 2:
        prev_exp = experiments[1]
        
        latest_vals = list(latest_exp.evaluation_metrics.values())
        latest_score = sum(latest_vals) / len(latest_vals) if latest_vals else 0.0
        
        prev_vals = list(prev_exp.evaluation_metrics.values())
        prev_score = sum(prev_vals) / len(prev_vals) if prev_vals else 0.0
        
        score_delta = latest_score - prev_score
        latency_delta = latest_exp.average_latency - prev_exp.average_latency

    # Render KPI cards
    st.subheader("Key Performance Indicators")
    render_kpi_cards(
        overall_score=avg_overall_score,
        pass_rate=latest_exp.passed_test_cases / latest_exp.total_number_of_test_cases if latest_exp.total_number_of_test_cases > 0 else 0.0,
        avg_latency_ms=avg_latency,
        total_cases=total_cases,
        score_delta=score_delta,
        latency_delta=latency_delta
    )
    
    st.markdown("---")
    
    # System Trends
    col1, col2 = st.columns(2)
    with col1:
        render_score_trends(experiments)
    with col2:
        render_latency_trends(experiments)
        
    st.markdown("---")
    
    # Recent Alert Section
    st.subheader("⚠️ Recent Alerts")
    if len(experiments) >= 2:
        try:
            curr_report = get_experiment_report(latest_exp)
            prev_report = get_experiment_report(experiments[1])
            comp = get_regression_comparison(curr_report, prev_report)
            
            if comp.regression_status:
                st.error(f"**Regression Detected!** Run `{latest_exp.experiment_id}` performed WORSE than previous run `{experiments[1].experiment_id}`.")
                for change in comp.change_summary:
                    if "regressed" in change or "increased" in change:
                        st.write(f"- {change}")
            elif comp.improvement_status:
                st.success(f"**Improvement Detected!** Run `{latest_exp.experiment_id}` performed BETTER than previous run `{experiments[1].experiment_id}`.")
                for change in comp.change_summary:
                    if "improved" in change or "decreased" in change:
                        st.write(f"- {change}")
            else:
                st.info(f"System status stable. Performance of run `{latest_exp.experiment_id}` is approximately the same as `{experiments[1].experiment_id}`.")
        except Exception as e:
            st.error(f"Failed to load regression alert: {str(e)}")
    else:
        st.info("Additional experiment runs are required to compute regression comparisons and alerts.")
