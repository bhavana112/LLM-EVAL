import streamlit as st

def render_kpi_cards(
    overall_score: float,
    pass_rate: float,
    avg_latency_ms: float,
    total_cases: int,
    score_delta: float = 0.0,
    latency_delta: float = 0.0
):
    """
    Renders clean summary KPI metric cards side-by-side.
    """
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Overall Score",
            value=f"{overall_score:.2%}",
            delta=f"{score_delta:+.2%}" if score_delta != 0.0 else None
        )
        
    with col2:
        st.metric(
            label="Pass Rate",
            value=f"{pass_rate:.2%}"
        )
        
    with col3:
        st.metric(
            label="Avg Latency",
            value=f"{avg_latency_ms:.2f} ms",
            delta=f"{latency_delta:+.2f} ms" if latency_delta != 0.0 else None,
            delta_color="inverse"  # Lower is better for latency!
        )
        
    with col4:
        st.metric(
            label="Total Test Cases",
            value=total_cases
        )
