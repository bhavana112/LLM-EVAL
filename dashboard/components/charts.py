import streamlit as st
import pandas as pd
from typing import List
from backend.experiments.models import Experiment

def render_score_trends(experiments: List[Experiment]):
    """Renders a line chart displaying overall score trends across experiments."""
    st.caption("Overall Score Trend (Chronological)")
    if not experiments:
        st.info("No experiments to display trends.")
        return
        
    # Order experiments chronologically for trends
    sorted_exps = sorted(experiments, key=lambda x: x.timestamp)
    
    data = {
        "Experiment ID": [e.experiment_id for e in sorted_exps],
        "Overall Score": [e.evaluation_metrics.get("faithfulness", 0.0) if "faithfulness" in e.evaluation_metrics else e.average_latency for e in sorted_exps] # Fallback calculation
    }
    
    # Recalculate true overall score as average of available metrics
    scores = []
    for e in sorted_exps:
        vals = list(e.evaluation_metrics.values())
        avg_score = sum(vals) / len(vals) if vals else 0.0
        scores.append(avg_score)
        
    df = pd.DataFrame({
        "Experiment": [e.experiment_id[:15] + "..." if len(e.experiment_id) > 15 else e.experiment_id for e in sorted_exps],
        "Score": scores
    })
    
    st.line_chart(df.set_index("Experiment"))


def render_latency_trends(experiments: List[Experiment]):
    """Renders a bar chart displaying average latencies across experiments."""
    st.caption("Average Latency Trend (ms)")
    if not experiments:
        st.info("No experiments to display latencies.")
        return
        
    sorted_exps = sorted(experiments, key=lambda x: x.timestamp)
    df = pd.DataFrame({
        "Experiment": [e.experiment_id[:15] + "..." if len(e.experiment_id) > 15 else e.experiment_id for e in sorted_exps],
        "Average Latency (ms)": [e.average_latency for e in sorted_exps]
    })
    
    st.bar_chart(df.set_index("Experiment"))


def render_pass_fail_distribution(experiment: Experiment):
    """Renders a simple stacked horizontal bar representing pass vs fail ratio."""
    st.caption("Pass vs. Failed Case Distribution")
    total = experiment.total_number_of_test_cases
    passed = experiment.passed_test_cases
    failed = experiment.failed_test_cases
    
    if total == 0:
        st.info("No test cases evaluated.")
        return
        
    pass_pct = passed / total
    fail_pct = failed / total
    
    df = pd.DataFrame({
        "Status": ["Passed", "Failed"],
        "Count": [passed, failed],
        "Percentage": [f"{pass_pct:.1%}", f"{fail_pct:.1%}"]
    })
    
    st.bar_chart(df.set_index("Status")["Count"])


def render_multi_metric_comparison(experiments: List[Experiment]):
    """Compares average scores per metric for the selected runs."""
    st.caption("Metric Breakdown Comparison")
    
    metric_names = set()
    for e in experiments:
        metric_names.update(e.evaluation_metrics.keys())
        
    if not metric_names:
        st.info("No metrics found in selected experiments.")
        return
        
    chart_data = []
    for e in experiments:
        row = {"Experiment": e.experiment_id}
        for m in metric_names:
            row[m] = e.evaluation_metrics.get(m, 0.0)
        chart_data.append(row)
        
    df = pd.DataFrame(chart_data)
    st.bar_chart(df.set_index("Experiment"))
