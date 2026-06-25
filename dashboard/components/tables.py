import streamlit as st
import pandas as pd
from typing import List
from backend.experiments.models import Experiment, EvaluationResultEntry

def render_experiments_overview(experiments: List[Experiment]):
    """Renders a clean summary table listing all experiments."""
    table_rows = []
    for e in experiments:
        # Calculate overall score as average of metrics
        vals = list(e.evaluation_metrics.values())
        overall_score = sum(vals) / len(vals) if vals else 0.0
        
        pass_rate = e.passed_test_cases / e.total_number_of_test_cases if e.total_number_of_test_cases > 0 else 0.0
        
        table_rows.append({
            "Experiment ID": e.experiment_id,
            "Timestamp": e.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "Provider": e.provider.upper(),
            "Model": e.model,
            "Dataset": e.dataset_name,
            "Overall Score": f"{overall_score:.2%}",
            "Pass Rate": f"{pass_rate:.2%}",
            "Avg Latency": f"{e.average_latency:.2f} ms"
        })
        
    df = pd.DataFrame(table_rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_test_cases_table(results: List[EvaluationResultEntry]):
    """Renders details of each evaluated test case."""
    rows = []
    for r in results:
        scores_str = ", ".join([f"{k}: {v:.2f}" for k, v in r.scores.items()])
        rows.append({
            "Test Case ID": r.test_case_id,
            "Prompt": r.prompt,
            "Expected": r.expected_output or "N/A",
            "Generated": r.generated_output,
            "Scores": scores_str if scores_str else "None",
            "Latency": f"{r.latency_ms:.1f} ms",
            "Status": "✅ Passed" if r.passed else "❌ Failed"
        })
        
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_config_comparison_table(experiments: List[Experiment]):
    """Renders a comparison table of configuration settings side by side."""
    data = {}
    
    # Rows we want to compare
    attributes = [
        "Provider",
        "Model",
        "Temperature",
        "Max Output Tokens",
        "System Instruction",
        "Overall Score",
        "Pass Rate",
        "Avg Latency"
    ]
    
    for attr in attributes:
        data[attr] = []
        
    for e in experiments:
        vals = list(e.evaluation_metrics.values())
        overall_score = sum(vals) / len(vals) if vals else 0.0
        pass_rate = e.passed_test_cases / e.total_number_of_test_cases if e.total_number_of_test_cases > 0 else 0.0
        
        gen_config = e.evaluation_configuration.get("generation_config", {})
        temp = gen_config.get("temperature", "Default")
        max_t = gen_config.get("max_tokens", "Default")
        sys_inst = e.evaluation_configuration.get("system_instruction") or "None"
        
        data["Provider"].append(e.provider)
        data["Model"].append(e.model)
        data["Temperature"].append(str(temp))
        data["Max Output Tokens"].append(str(max_t))
        data["System Instruction"].append(sys_inst[:40] + "..." if len(sys_inst) > 40 else sys_inst)
        data["Overall Score"].append(f"{overall_score:.2%}")
        data["Pass Rate"].append(f"{pass_rate:.2%}")
        data["Avg Latency"].append(f"{e.average_latency:.2f} ms")
        
    df = pd.DataFrame(data, index=[e.experiment_id for e in experiments]).T
    st.dataframe(df, use_container_width=True)
