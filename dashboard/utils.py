import httpx
import logging
import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from backend.experiments.models import Experiment, EvaluationResultEntry
from backend.reporting.report_generator import ReportGenerator
from backend.reporting.regression_detector import RegressionDetector
from backend.reporting.models import ExperimentReport, RegressionComparisonReport
from backend.analysis.failure_analyzer import FailureAnalyzer
from backend.analysis.models import FailureAnalysisResult, GroupedFailure
from backend.providers.provider_factory import ProviderFactory
from backend.providers.response import LLMResponse

logger = logging.getLogger("dashboard.utils")

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


# --- Health Check ---
def check_backend_health() -> bool:
    """Checks if the FastAPI backend server is online."""
    try:
        response = httpx.get(f"{BACKEND_URL}/health", timeout=1.0)
        return response.status_code == 200 and response.json().get("status") == "healthy"
    except Exception:
        return False


# --- Model Adapter ---
def adapt_run_to_experiment(run: Dict[str, Any]) -> Experiment:
    """
    Translates API ExperimentRun model dictionary representation 
    into the modern Pydantic Experiment schema.
    """
    config = run.get("config", {}) or {}
    results_raw = run.get("results", []) or []
    
    eval_results = []
    passed_count = 0
    failed_count = 0
    total_latency = 0.0
    
    for r in results_raw:
        scores = r.get("metrics", {}) or {}
        # Assume pass if all metric scores are >= 0.8 (or default True if no scores)
        passed = all(val >= 0.8 for val in scores.values()) if scores else True
        if passed:
            passed_count += 1
        else:
            failed_count += 1
            
        latency = r.get("latency_ms", 0.0)
        total_latency += latency
        
        # Parse timestamp
        t_str = r.get("timestamp")
        dt = datetime.now(timezone.utc)
        if t_str:
            try:
                dt = datetime.fromisoformat(t_str.replace("Z", "+00:00"))
            except ValueError:
                pass
        
        eval_results.append(
            EvaluationResultEntry(
                test_case_id=r.get("prompt_id", "unknown"),
                prompt=r.get("prompt", ""),
                generated_output=r.get("generated_output", ""),
                expected_output=r.get("expected_output"),
                scores=scores,
                latency_ms=latency,
                timestamp=dt,
                passed=passed,
                success=True
            )
        )
        
    total_cases = len(eval_results)
    avg_latency = total_latency / total_cases if total_cases > 0 else 0.0
    
    # Filter out latency key from summary metrics for overall accuracy
    metrics = {k: v for k, v in run.get("summary_metrics", {}).items() if k != "avg_latency_ms"}
    
    # Created At timestamp
    c_str = run.get("created_at")
    cdt = datetime.now(timezone.utc)
    if c_str:
        try:
            cdt = datetime.fromisoformat(c_str.replace("Z", "+00:00"))
        except ValueError:
            pass

    return Experiment(
        experiment_id=run.get("id", "unknown"),
        timestamp=cdt,
        dataset_name=config.get("dataset_id", "unknown"),
        dataset_version="1.0",
        provider=config.get("provider_name", "unknown"),
        model=config.get("model_name", "unknown"),
        evaluation_configuration={
            "generation_config": config.get("generation_config", {}),
            "system_instruction": config.get("system_instruction")
        },
        evaluation_results=eval_results,
        evaluation_metrics=metrics,
        average_latency=avg_latency,
        total_number_of_test_cases=total_cases,
        passed_test_cases=passed_count,
        failed_test_cases=failed_count
    )


# --- REST API Functions ---
def fetch_experiments() -> List[Experiment]:
    """
    Fetches experiment runs. Checks the FastAPI backend first;
    falls back to generating high-quality mock data if API is offline or returns empty.
    """
    if check_backend_health():
        try:
            response = httpx.get(f"{BACKEND_URL}/api/v1/experiments/", timeout=3.0)
            if response.status_code == 200:
                runs = response.json()
                if runs:
                    return [adapt_run_to_experiment(run) for run in runs]
        except Exception as e:
            logger.error(f"API fetch experiments failed: {str(e)}")
            
    # Return mock data as fallback
    return get_mock_experiments()


def fetch_experiment_by_id(experiment_id: str) -> Optional[Experiment]:
    """Retrieves an experiment by its unique identifier."""
    experiments = fetch_experiments()
    for exp in experiments:
        if exp.experiment_id == experiment_id:
            return exp
    return None


def fetch_providers() -> List[Dict[str, Any]]:
    """Retrieves the list of registered LLM providers."""
    if check_backend_health():
        try:
            response = httpx.get(f"{BACKEND_URL}/api/v1/providers/", timeout=2.0)
            if response.status_code == 200:
                return response.json().get("providers", [])
        except Exception:
            pass
            
    # Fallback default providers
    return [
        {"name": "openai", "models": ["gpt-4o", "gpt-4-turbo"]},
        {"name": "gemini", "models": ["gemini-2.5-flash", "gemini-2.5-pro"]},
        {"name": "anthropic", "models": ["claude-3-5-sonnet"]}
    ]


def fetch_datasets() -> List[Dict[str, Any]]:
    """Retrieves list of benchmark datasets."""
    if check_backend_health():
        try:
            response = httpx.get(f"{BACKEND_URL}/api/v1/datasets/", timeout=2.0)
            if response.status_code == 200:
                return response.json()
        except Exception:
            pass
            
    return [
        {"id": "qa-benchmark-v1", "name": "QA Benchmark v1", "entries_count": 10},
        {"id": "translation-es", "name": "Spanish Translation Benchmark", "entries_count": 5}
    ]


# --- Local Analysis Calculation wrappers ---
def get_experiment_report(experiment: Experiment) -> ExperimentReport:
    """Invokes ReportGenerator to create a detailed report for the experiment."""
    generator = ReportGenerator()
    return generator.generate_report(experiment)


def get_regression_comparison(current: ExperimentReport, previous: ExperimentReport) -> RegressionComparisonReport:
    """Invokes RegressionDetector to compare two experiment reports."""
    detector = RegressionDetector(tolerance=0.01)
    return detector.compare(current, previous)


async def get_failure_analysis(experiment: Experiment) -> FailureAnalysisResult:
    """
    Invokes FailureAnalyzer to evaluate test cases. 
    If Gemini API keys are configured, it calls the model; otherwise it returns a simulated analysis report.
    """
    # Check if Gemini api key is loaded
    from backend.providers.config import provider_settings
    if provider_settings.GEMINI_API_KEY:
        try:
            provider = ProviderFactory.create("gemini", "gemini-2.5-flash")
            analyzer = FailureAnalyzer(provider)
            return await analyzer.analyze_failures(experiment)
        except Exception as e:
            logger.warning(f"Live Failure Analysis execution failed (using simulation instead): {str(e)}")
            
    return get_simulated_failure_analysis(experiment)


# --- Mock Data Generators ---
def get_mock_experiments() -> List[Experiment]:
    """Generates a stable list of high-quality mock experiments for styling & validation."""
    now = datetime.now(timezone.utc)
    
    # 1. Gemini Run (Run 1 - Baseline)
    results_gemini = [
        EvaluationResultEntry(
            test_case_id="tc-1", prompt="What is RAG?", generated_output="RAG stands for Retrieval-Augmented Generation. It retrieves documents to context.", expected_output="Retrieval Augmented Generation is a technique to ground LLMs on external document retrieval.",
            scores={"faithfulness": 0.9, "answer_relevancy": 0.85}, latency_ms=110.0, timestamp=now, passed=True, success=True
        ),
        EvaluationResultEntry(
            test_case_id="tc-2", prompt="Capital of France?", generated_output="The capital of France is Paris.", expected_output="Paris",
            scores={"exact_match": 1.0}, latency_ms=80.0, timestamp=now, passed=True, success=True
        ),
        EvaluationResultEntry(
            test_case_id="tc-3", prompt="Calculate 15 * 6", generated_output="15 * 6 is 90.", expected_output="90",
            scores={"exact_match": 1.0}, latency_ms=90.0, timestamp=now, passed=True, success=True
        ),
        EvaluationResultEntry(
            test_case_id="tc-4", prompt="Who wrote Hamlet?", generated_output="Shakespeare wrote Hamlet.", expected_output="William Shakespeare",
            scores={"exact_match": 0.8, "answer_relevancy": 0.9}, latency_ms=130.0, timestamp=now, passed=True, success=True
        ),
        EvaluationResultEntry(
            test_case_id="tc-5", prompt="What is photosyntesis?", generated_output="Photosynthesis is how plants produce food using sunlight and water. It produces oxygen.", expected_output="Plants capture light to convert carbon dioxide and water into glucose and oxygen.",
            scores={"faithfulness": 0.95, "answer_relevancy": 0.9}, latency_ms=140.0, timestamp=now, passed=True, success=True
        ),
        EvaluationResultEntry(
            test_case_id="tc-6", prompt="Explain gravity briefly.", generated_output="Gravity is the force that pulls things to the ground.", expected_output="Gravity is a natural phenomenon by which all things with mass or energy are brought toward one another.",
            scores={"faithfulness": 0.7, "answer_relevancy": 0.75}, latency_ms=105.0, timestamp=now, passed=False, success=True
        ),
        EvaluationResultEntry(
            test_case_id="tc-7", prompt="Define API.", generated_output="An API is an Application Programming Interface, allowing different apps to talk to each other.", expected_output="Application Programming Interface: protocols and tools for building software applications.",
            scores={"faithfulness": 0.85, "answer_relevancy": 0.85}, latency_ms=115.0, timestamp=now, passed=True, success=True
        ),
        EvaluationResultEntry(
            test_case_id="tc-8", prompt="List primary colors.", generated_output="Red, yellow, and blue.", expected_output="Red, yellow, blue",
            scores={"exact_match": 1.0}, latency_ms=85.0, timestamp=now, passed=True, success=True
        ),
        EvaluationResultEntry(
            test_case_id="tc-9", prompt="Convert 100C to F.", generated_output="100 degrees Celsius is 212 degrees Fahrenheit.", expected_output="212",
            scores={"exact_match": 1.0}, latency_ms=95.0, timestamp=now, passed=True, success=True
        ),
        EvaluationResultEntry(
            test_case_id="tc-10", prompt="Brief summary of WWII.", generated_output="World War II was a global war that lasted from 1939 to 1945.", expected_output="A global military conflict (1939-1945) involving major alliances of Allies and Axis.",
            scores={"faithfulness": 0.6, "answer_relevancy": 0.5}, latency_ms=160.0, timestamp=now, passed=False, success=True
        )
    ]
    exp1 = Experiment(
        experiment_id="exp_gemini_flash_baseline",
        timestamp=datetime(2026, 6, 25, 10, 0, 0, tzinfo=timezone.utc),
        dataset_name="qa-benchmark-v1",
        dataset_version="1.0",
        provider="gemini",
        model="gemini-2.5-flash",
        evaluation_configuration={"temperature": 0.1, "max_tokens": 512},
        evaluation_results=results_gemini,
        evaluation_metrics={"faithfulness": 0.85, "answer_relevancy": 0.81, "exact_match": 0.96},
        average_latency=111.0,
        total_number_of_test_cases=10,
        passed_test_cases=8,
        failed_test_cases=2
    )

    # 2. GPT-4o Run (Run 2 - Improvement)
    results_gpt = [
        EvaluationResultEntry(
            test_case_id="tc-1", prompt="What is RAG?", generated_output="Retrieval-Augmented Generation merges document retrieval with text generation models.", expected_output="Retrieval Augmented Generation is a technique to ground LLMs on external document retrieval.",
            scores={"faithfulness": 0.95, "answer_relevancy": 0.95}, latency_ms=160.0, timestamp=now, passed=True, success=True
        ),
        EvaluationResultEntry(
            test_case_id="tc-2", prompt="Capital of France?", generated_output="Paris.", expected_output="Paris",
            scores={"exact_match": 1.0}, latency_ms=110.0, timestamp=now, passed=True, success=True
        ),
        EvaluationResultEntry(
            test_case_id="tc-3", prompt="Calculate 15 * 6", generated_output="90", expected_output="90",
            scores={"exact_match": 1.0}, latency_ms=120.0, timestamp=now, passed=True, success=True
        ),
        EvaluationResultEntry(
            test_case_id="tc-4", prompt="Who wrote Hamlet?", generated_output="William Shakespeare", expected_output="William Shakespeare",
            scores={"exact_match": 1.0, "answer_relevancy": 1.0}, latency_ms=180.0, timestamp=now, passed=True, success=True
        ),
        EvaluationResultEntry(
            test_case_id="tc-5", prompt="What is photosyntesis?", generated_output="Photosynthesis converts light, CO2, and water into glucose and oxygen.", expected_output="Plants capture light to convert carbon dioxide and water into glucose and oxygen.",
            scores={"faithfulness": 0.98, "answer_relevancy": 0.98}, latency_ms=210.0, timestamp=now, passed=True, success=True
        ),
        EvaluationResultEntry(
            test_case_id="tc-6", prompt="Explain gravity briefly.", generated_output="Gravity is a universal force pulling objects with mass toward each other.", expected_output="Gravity is a natural phenomenon by which all things with mass or energy are brought toward one another.",
            scores={"faithfulness": 0.9, "answer_relevancy": 0.88}, latency_ms=175.0, timestamp=now, passed=True, success=True
        ),
        EvaluationResultEntry(
            test_case_id="tc-7", prompt="Define API.", generated_output="An Application Programming Interface defines protocols and interactions between software.", expected_output="Application Programming Interface: protocols and tools for building software applications.",
            scores={"faithfulness": 0.92, "answer_relevancy": 0.92}, latency_ms=185.0, timestamp=now, passed=True, success=True
        ),
        EvaluationResultEntry(
            test_case_id="tc-8", prompt="List primary colors.", generated_output="Red, Yellow, Blue", expected_output="Red, yellow, blue",
            scores={"exact_match": 1.0}, latency_ms=130.0, timestamp=now, passed=True, success=True
        ),
        EvaluationResultEntry(
            test_case_id="tc-9", prompt="Convert 100C to F.", generated_output="212", expected_output="212",
            scores={"exact_match": 1.0}, latency_ms=140.0, timestamp=now, passed=True, success=True
        ),
        EvaluationResultEntry(
            test_case_id="tc-10", prompt="Brief summary of WWII.", generated_output="World War II (1939-1945) was a global conflict fought between the Allies and the Axis power.", expected_output="A global military conflict (1939-1945) involving major alliances of Allies and Axis.",
            scores={"faithfulness": 0.75, "answer_relevancy": 0.75}, latency_ms=240.0, timestamp=now, passed=False, success=True
        )
    ]
    exp2 = Experiment(
        experiment_id="exp_gpt4o_performance_run",
        timestamp=datetime(2026, 6, 25, 11, 0, 0, tzinfo=timezone.utc),
        dataset_name="qa-benchmark-v1",
        dataset_version="1.0",
        provider="openai",
        model="gpt-4o",
        evaluation_configuration={"temperature": 0.0, "max_tokens": 1024},
        evaluation_results=results_gpt,
        evaluation_metrics={"faithfulness": 0.91, "answer_relevancy": 0.90, "exact_match": 1.0},
        average_latency=165.0,
        total_number_of_test_cases=10,
        passed_test_cases=9,
        failed_test_cases=1
    )

    # 3. Claude-3-5 Run (Run 3 - Regression)
    results_claude = [
        EvaluationResultEntry(
            test_case_id="tc-1", prompt="What is RAG?", generated_output="RAG is a model that creates text from scratch.", expected_output="Retrieval Augmented Generation is a technique to ground LLMs on external document retrieval.",
            scores={"faithfulness": 0.3, "answer_relevancy": 0.4}, latency_ms=190.0, timestamp=now, passed=False, success=True
        ),
        EvaluationResultEntry(
            test_case_id="tc-2", prompt="Capital of France?", generated_output="Paris is nice, but I don't know the capital.", expected_output="Paris",
            scores={"exact_match": 0.0}, latency_ms=130.0, timestamp=now, passed=False, success=True
        ),
        EvaluationResultEntry(
            test_case_id="tc-3", prompt="Calculate 15 * 6", generated_output="80", expected_output="90",
            scores={"exact_match": 0.0}, latency_ms=140.0, timestamp=now, passed=False, success=True
        ),
        EvaluationResultEntry(
            test_case_id="tc-4", prompt="Who wrote Hamlet?", generated_output="William Shakespeare", expected_output="William Shakespeare",
            scores={"exact_match": 1.0, "answer_relevancy": 1.0}, latency_ms=195.0, timestamp=now, passed=True, success=True
        ),
        EvaluationResultEntry(
            test_case_id="tc-5", prompt="What is photosyntesis?", generated_output="Photosynthesis turns water into food.", expected_output="Plants capture light to convert carbon dioxide and water into glucose and oxygen.",
            scores={"faithfulness": 0.5, "answer_relevancy": 0.6}, latency_ms=220.0, timestamp=now, passed=False, success=True
        ),
        EvaluationResultEntry(
            test_case_id="tc-6", prompt="Explain gravity briefly.", generated_output="Gravity is a universal force pulling objects with mass toward each other.", expected_output="Gravity is a natural phenomenon by which all things with mass or energy are brought toward one another.",
            scores={"faithfulness": 0.9, "answer_relevancy": 0.88}, latency_ms=180.0, timestamp=now, passed=True, success=True
        ),
        EvaluationResultEntry(
            test_case_id="tc-7", prompt="Define API.", generated_output="An Application Programming Interface defines protocols and interactions between software.", expected_output="Application Programming Interface: protocols and tools for building software applications.",
            scores={"faithfulness": 0.92, "answer_relevancy": 0.92}, latency_ms=190.0, timestamp=now, passed=True, success=True
        ),
        EvaluationResultEntry(
            test_case_id="tc-8", prompt="List primary colors.", generated_output="Red, Yellow, Blue", expected_output="Red, yellow, blue",
            scores={"exact_match": 1.0}, latency_ms=145.0, timestamp=now, passed=True, success=True
        ),
        EvaluationResultEntry(
            test_case_id="tc-9", prompt="Convert 100C to F.", generated_output="212", expected_output="212",
            scores={"exact_match": 1.0}, latency_ms=150.0, timestamp=now, passed=True, success=True
        ),
        EvaluationResultEntry(
            test_case_id="tc-10", prompt="Brief summary of WWII.", generated_output="World War II (1939-1945) was a global conflict fought between the Allies and the Axis power.", expected_output="A global military conflict (1939-1945) involving major alliances of Allies and Axis.",
            scores={"faithfulness": 0.75, "answer_relevancy": 0.75}, latency_ms=250.0, timestamp=now, passed=False, success=True
        )
    ]
    exp3 = Experiment(
        experiment_id="exp_claude35_regression_run",
        timestamp=datetime(2026, 6, 25, 12, 0, 0, tzinfo=timezone.utc),
        dataset_name="qa-benchmark-v1",
        dataset_version="1.0",
        provider="anthropic",
        model="claude-3-5-sonnet",
        evaluation_configuration={"temperature": 0.7, "max_tokens": 1024},
        evaluation_results=results_claude,
        evaluation_metrics={"faithfulness": 0.67, "answer_relevancy": 0.66, "exact_match": 0.5},
        average_latency=179.0,
        total_number_of_test_cases=10,
        passed_test_cases=5,
        failed_test_cases=5
    )

    return [exp3, exp2, exp1]


def get_simulated_failure_analysis(experiment: Experiment) -> FailureAnalysisResult:
    """Generates structured failure analysis data based on experiment ID."""
    exp_id = experiment.experiment_id
    
    if "gemini" in exp_id:
        categories = ["Hallucination", "Low Relevance"]
        grouped_failures = {
            "Hallucination": [
                GroupedFailure(test_case_ids=["tc-10"], description="Model hallucinated timeline and historical events during WWII summary.")
            ],
            "Low Relevance": [
                GroupedFailure(test_case_ids=["tc-6", "tc-10"], description="Model responses were too brief or diverged from the benchmark evaluation criteria.")
            ]
        }
        category_counts = {"Hallucination": 1, "Low Relevance": 2}
        patterns = ["Factual errors on long-form historical prompts", "Low semantic coverage on physics terms"]
        summary = "The Gemini baseline suffered primarily from low relevance and shallow summaries on broad topics."
        recs = [
            "Increase system prompt detail for historical summarizations.",
            "Use few-shot examples to illustrate standard summary lengths."
        ]
    elif "gpt4o" in exp_id:
        categories = ["Low Relevance"]
        grouped_failures = {
            "Low Relevance": [
                GroupedFailure(test_case_ids=["tc-10"], description="The summary omitted key details about Axis and Allies alliances.")
            ]
        }
        category_counts = {"Low Relevance": 1}
        patterns = ["Mild context omissions in summaries"]
        summary = "GPT-4o performed exceptionally well, showing only minor completeness issues in summaries."
        recs = [
            "Incorporate a list of key points that summaries must touch upon."
        ]
    else: # Claude regression
        categories = ["Hallucination", "Formatting Problems", "Incorrect Reasoning"]
        grouped_failures = {
            "Hallucination": [
                GroupedFailure(test_case_ids=["tc-1", "tc-5"], description="Fabrication of definitions: claimed RAG creates text from scratch and misdefined photosynthesis elements.")
            ],
            "Formatting Problems": [
                GroupedFailure(test_case_ids=["tc-2"], description="Model returned conversational padding instead of a simple direct answer.")
            ],
            "Incorrect Reasoning": [
                GroupedFailure(test_case_ids=["tc-3"], description="Basic math failure: calculated 15 * 6 = 80.")
            ]
        }
        category_counts = {"Hallucination": 2, "Formatting Problems": 1, "Incorrect Reasoning": 1}
        patterns = ["Frequent hallucinations due to high temperature configuration (0.7)", "Formatting failures on factual QA", "Arithmetic error on simple multiplication"]
        summary = "This run saw a significant drop in accuracy across math, formatting, and factual consistency. The higher temperature setting is the likely driver."
        recs = [
            "Reduce generation temperature from 0.7 to 0.0 for factual benchmarks.",
            "Enforce strict formatting schemas or JSON outputs in the prompt.",
            "Provide few-shot logic steps for arithmetic queries."
        ]

    return FailureAnalysisResult(
        experiment_id=exp_id,
        analysis_timestamp=datetime.now(timezone.utc),
        number_of_failed_cases=experiment.failed_test_cases,
        detected_categories=categories,
        grouped_failures=grouped_failures,
        category_counts=category_counts,
        identified_patterns=patterns,
        ai_generated_summary=summary,
        recommendations=recs,
        provider_used_for_analysis="gemini"
    )
