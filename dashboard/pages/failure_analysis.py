import streamlit as st
import asyncio
import pandas as pd
from typing import List
from backend.experiments.models import Experiment
from dashboard.components.filters import render_experiment_selector
from dashboard.utils import get_failure_analysis

def show_failure_analysis_page(experiments: List[Experiment]):
    """Renders the AI-Assisted Failure Analysis report page."""
    st.title("🤖 AI-Assisted Failure Analysis")
    st.markdown("Automatically groups evaluation failures, identifies recurring patterns, and generates root cause insights and actionable recommendations.")
    st.markdown("---")
    
    if not experiments:
        st.info("No experiments available for failure analysis.")
        return
        
    selected_exp = render_experiment_selector(experiments, key_prefix="failure_analysis")
    
    if selected_exp:
        if selected_exp.failed_test_cases == 0:
            st.success(
                f"🎉 **Perfect Run!** Experiment `{selected_exp.experiment_id}` has **0 failed test cases**. "
                "No failure analysis is required."
            )
            return
            
        st.subheader("Analysis Summary")
        
        # Resolve async get_failure_analysis call
        try:
            with st.spinner("Generating failure analysis report from LLM provider..."):
                analysis_result = asyncio.run(get_failure_analysis(selected_exp))
                
            # Render root cause summary
            st.info(f"🧬 **Root Cause Verdict:** {analysis_result.ai_generated_summary}")
            
            # Show categories counts
            col_stats, col_chart = st.columns([1, 1])
            with col_stats:
                st.subheader("Failed Categories Count")
                for cat, count in analysis_result.category_counts.items():
                    st.markdown(f"- **{cat}:** {count} case(s)")
            with col_chart:
                if analysis_result.category_counts:
                    df = pd.DataFrame({
                        "Category": list(analysis_result.category_counts.keys()),
                        "Failures": list(analysis_result.category_counts.values())
                    })
                    st.bar_chart(df.set_index("Category"))
                    
            st.markdown("---")
            
            # Recurring failure patterns
            st.subheader("Identified Recurring Patterns")
            if analysis_result.identified_patterns:
                for idx, pattern in enumerate(analysis_result.identified_patterns):
                    st.write(f"{idx+1}. ⚠️ {pattern}")
            else:
                st.write("No specific patterns detected.")
                
            st.markdown("---")
            
            # Grouped Failures details
            st.subheader("Grouped Failures List")
            for cat, groups in analysis_result.grouped_failures.items():
                st.markdown(f"#### Category: {cat}")
                for idx, group in enumerate(groups):
                    with st.expander(f"Theme {idx+1}: {group.description[:80]}...", expanded=False):
                        st.write(f"**Failure Theme:** {group.description}")
                        st.write("**Impacted Case IDs:**")
                        st.write(", ".join([f"`{cid}`" for cid in group.test_case_ids]))
                        
            st.markdown("---")
            
            # Recommendations
            st.subheader("📋 Recommendations for Improvement")
            if analysis_result.recommendations:
                for rec in analysis_result.recommendations:
                    st.checkbox(label=rec, value=False, key=f"rec_{rec[:30]}")
            else:
                st.write("No specific recommendations generated.")
                
            st.caption(f"Analysis provider: {analysis_result.provider_used_for_analysis.upper()}")
            
        except Exception as e:
            st.error(f"Failed to generate failure analysis: {str(e)}")
            st.info("Tip: Double-check your Gemini API key or network connection if running in live mode.")
            logger.error(f"Failure analysis rendering error: {str(e)}")
