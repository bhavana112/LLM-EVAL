import streamlit as st
from typing import List
from backend.experiments.models import Experiment
from dashboard.components.filters import render_experiment_selector
from dashboard.utils import get_experiment_report

def show_reports_page(experiments: List[Experiment]):
    """Renders the detailed Experiment Reports view."""
    st.title("📄 Evaluation Reports")
    st.markdown("Detailed breakdown reports generated for individual evaluation runs, summarizing metrics and latency.")
    st.markdown("---")
    
    if not experiments:
        st.info("No experiments available to generate reports.")
        return
        
    selected_exp = render_experiment_selector(experiments, key_prefix="reports")
    
    if selected_exp:
        try:
            report = get_experiment_report(selected_exp)
            
            # Show summary block
            st.info(f"📝 **Summary:** {report.evaluation_summary}")
            
            st.subheader("Performance & Quality Metrics")
            
            # Show KPI stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Overall Score", f"{report.overall_score:.2%}")
            with col2:
                st.metric("Pass Rate", f"{report.pass_rate:.2%}")
            with col3:
                st.metric("Successful Cases", f"{report.successful_evaluations} / {report.total_number_of_test_cases}")
                
            st.markdown("---")
            
            # Metric averages breakdown
            st.subheader("Metric Scores Breakdown")
            if report.average_score_for_every_evaluation_metric:
                cols = st.columns(len(report.average_score_for_every_evaluation_metric))
                for idx, (metric_name, score) in enumerate(report.average_score_for_every_evaluation_metric.items()):
                    with cols[idx]:
                        st.metric(metric_name.replace("_", " ").title(), f"{score:.2f}")
            else:
                st.write("No metrics computed for this evaluation.")
                
            st.markdown("---")
            
            # Latency statistics
            st.subheader("⏱️ Latency Stats")
            l_col1, l_col2, l_col3, l_col4 = st.columns(4)
            with l_col1:
                st.metric("Average Latency", f"{report.average_latency:.1f} ms")
            with l_col2:
                st.metric("Minimum Latency", f"{report.minimum_latency:.1f} ms")
            with l_col3:
                st.metric("Maximum Latency", f"{report.maximum_latency:.1f} ms")
            with l_col4:
                st.metric("Total Exec Time", f"{report.total_execution_time_ms / 1000:.2f} s")
                
            st.markdown("---")
            
            # Failed Test Cases List
            st.subheader("❌ Failed Cases")
            failed_cases = [r for r in selected_exp.evaluation_results if not r.passed or not r.success]
            if failed_cases:
                for case in failed_cases:
                    with st.expander(f"Test Case ID: {case.test_case_id}"):
                        st.markdown(f"**Prompt:** {case.prompt}")
                        st.markdown(f"**Expected Output:** `{case.expected_output or 'N/A'}`")
                        st.markdown(f"**Generated Output:** `{case.generated_output}`")
                        
                        # Scores
                        st.write("**Evaluation Scores:**")
                        st.write(case.scores)
                        if case.error_message:
                            st.error(f"Error Message: {case.error_message}")
            else:
                st.success("All test cases passed successfully in this run!")
                
        except Exception as e:
            st.error(f"Failed to generate evaluation report: {str(e)}")
            logger.error(f"Report rendering error: {str(e)}")
