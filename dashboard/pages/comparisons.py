import streamlit as st
from typing import List
from backend.experiments.models import Experiment
from dashboard.components.filters import render_multiple_experiments_selector
from dashboard.components.tables import render_config_comparison_table
from dashboard.components.charts import render_multi_metric_comparison

def show_comparisons_page(experiments: List[Experiment]):
    """Renders side-by-side configuration and performance comparisons of multiple runs."""
    st.title("📊 Configuration Comparisons")
    st.markdown("Select two or more experiments to compare performance metrics, latency statistics, and prompt/generation parameters.")
    st.markdown("---")
    
    if not experiments:
        st.info("No experiments available for comparison.")
        return
        
    selected_exps = render_multiple_experiments_selector(experiments, key_prefix="comparisons")
    
    if len(selected_exps) >= 2:
        # Determine the best performing configuration based on overall score
        best_exp = None
        best_score = -1.0
        
        for e in selected_exps:
            vals = list(e.evaluation_metrics.values())
            overall_score = sum(vals) / len(vals) if vals else 0.0
            if overall_score > best_score:
                best_score = overall_score
                best_exp = e
                
        if best_exp:
            st.success(
                f"🏆 **Best Configuration:** Run `{best_exp.experiment_id}` using **{best_exp.model}** "
                f"via **{best_exp.provider.upper()}** achieved the highest overall score of **{best_score:.2%}**."
            )
            
        st.subheader("Side-by-Side Configuration & Score Comparison")
        render_config_comparison_table(selected_exps)
        
        st.markdown("---")
        
        # Metric-wise breakdown chart
        st.subheader("Metric Performance Chart")
        render_multi_metric_comparison(selected_exps)
        
        st.markdown("---")
        
        # Latency side-by-side comparison
        st.subheader("Latency Profiling")
        l_cols = st.columns(len(selected_exps))
        for idx, e in enumerate(selected_exps):
            with l_cols[idx]:
                st.metric(
                    label=f"Avg Latency: {e.model}",
                    value=f"{e.average_latency:.1f} ms",
                    delta=f"{e.average_latency - best_exp.average_latency:.1f} ms" if e != best_exp else "Best Score",
                    delta_color="inverse"
                )
    else:
        st.info("Please select at least 2 experiments to perform comparisons.")
        
        # If they only have one selected, we show a friendly hint
        if len(selected_exps) == 1:
            st.warning("Multi-run comparison requires selecting additional experiments from the list above.")
