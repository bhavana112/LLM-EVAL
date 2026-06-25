import streamlit as st
from typing import List, Optional
from backend.experiments.models import Experiment

def render_experiment_selector(experiments: List[Experiment], key_prefix: str = "") -> Optional[Experiment]:
    """Renders a dropdown box to select a single experiment from a list."""
    if not experiments:
        st.warning("No experiments available to select.")
        return None
        
    options = {f"{e.experiment_id} ({e.provider} - {e.model})": e for e in experiments}
    selected_label = st.selectbox(
        label="Select Experiment",
        options=list(options.keys()),
        key=f"{key_prefix}_single_selector"
    )
    return options.get(selected_label)


def render_multiple_experiments_selector(experiments: List[Experiment], key_prefix: str = "") -> List[Experiment]:
    """Renders a multi-select box to choose multiple experiments for comparison."""
    if not experiments:
        st.warning("No experiments available for comparison.")
        return []
        
    options = {f"{e.experiment_id} ({e.provider} - {e.model})": e for e in experiments}
    selected_labels = st.multiselect(
        label="Select Experiments to Compare",
        options=list(options.keys()),
        default=list(options.keys())[:min(2, len(options))],
        key=f"{key_prefix}_multi_selector"
    )
    return [options.get(label) for label in selected_labels]
