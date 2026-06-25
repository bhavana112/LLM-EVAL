import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone
import json

from backend.experiments.models import Experiment, EvaluationResultEntry
from backend.providers.base import LLMProvider
from backend.providers.response import LLMResponse
from backend.analysis.exceptions import (
    NoFailedCasesError, InvalidExperimentDataError, AnalysisExecutionError
)
from backend.analysis.models import FailureAnalysisResult, GroupedFailure
from backend.analysis.prompt_builder import PromptBuilder
from backend.analysis.failure_analyzer import FailureAnalyzer, DEFAULT_CATEGORIES
from backend.analysis.utils import extract_json

class MockLLMProvider(LLMProvider):
    """Minimal mock provider for testing."""
    def __init__(self, model_name: str = "mock-model", api_key: str = "mock-key"):
        super().__init__(model_name, api_key)
        self.mock_generate = AsyncMock()

    @property
    def provider_name(self) -> str:
        return "mock-provider"

    async def _generate(self, prompt: str, system_prompt: str | None = None, temperature: float = 0.0, max_tokens: int = 1024):
        # This won't be called directly since we will mock the public 'generate' method for simplicity
        return await self.mock_generate(prompt, system_prompt, temperature, max_tokens)


# --- Helper to create sample experiment ---
def make_mock_experiment(results: list) -> Experiment:
    return Experiment(
        experiment_id="exp_test_123",
        timestamp=datetime.now(timezone.utc),
        dataset_name="test-dataset",
        provider="mock-provider",
        model="mock-model",
        evaluation_configuration={"temp": 0.0},
        evaluation_results=results,
        evaluation_metrics={"accuracy": 0.5},
        average_latency=120.0,
        total_number_of_test_cases=len(results),
        passed_test_cases=sum(1 for r in results if r.passed),
        failed_test_cases=sum(1 for r in results if not r.passed)
    )


# --- 1. Test JSON Extraction Utility ---
def test_extract_json_valid():
    text = '{"name": "test", "value": 123}'
    parsed = extract_json(text)
    assert parsed == {"name": "test", "value": 123}

def test_extract_json_markdown():
    text = """
    Some prefix text
    ```json
    {
      "key": "value"
    }
    ```
    Some suffix text
    """
    parsed = extract_json(text)
    assert parsed == {"key": "value"}

def test_extract_json_invalid():
    with pytest.raises(ValueError):
        extract_json("not a json string")


# --- 2. Test Prompt Builder ---
def test_prompt_builder_truncation():
    builder = PromptBuilder(max_content_len=10)
    truncated = builder._truncate("123456789012345")
    assert truncated == "1234567890... [TRUNCATED]"

    none_truncated = builder._truncate(None)
    assert none_truncated == "N/A"

def test_prompt_builder_format_case():
    builder = PromptBuilder()
    case = {
        "test_case_id": "c1",
        "prompt": "Explain RAG",
        "expected_output": "Retrieval Augmented Generation",
        "generated_output": "Random text",
        "scores": {"faithfulness": 0.2},
        "passed": False,
        "success": True,
        "error_message": None
    }
    formatted = builder.format_failed_case(case)
    assert "c1" in formatted
    assert "Explain RAG" in formatted
    assert "Retrieval Augmented Generation" in formatted
    assert "Random text" in formatted
    assert "faithfulness: 0.20" in formatted
    assert "Failed metrics: faithfulness=0.20" in formatted

def test_prompt_builder_format_case_error():
    builder = PromptBuilder()
    case = {
        "test_case_id": "c2",
        "prompt": "Prompt",
        "expected_output": "Expected",
        "generated_output": "",
        "scores": {},
        "passed": False,
        "success": False,
        "error_message": "LLM timed out"
    }
    formatted = builder.format_failed_case(case)
    assert "c2" in formatted
    assert "LLM timed out" in formatted


# --- 3. Test Failure Filtering ---
@pytest.mark.asyncio
async def test_failure_analyzer_no_failures():
    # Setup an experiment where all cases passed and succeeded
    results = [
        EvaluationResultEntry(
            test_case_id="c1",
            prompt="P1",
            generated_output="G1",
            expected_output="E1",
            scores={"s": 1.0},
            latency_ms=10.0,
            timestamp=datetime.now(timezone.utc),
            passed=True,
            success=True
        )
    ]
    exp = make_mock_experiment(results)
    
    provider = MockLLMProvider()
    analyzer = FailureAnalyzer(provider)
    
    with pytest.raises(NoFailedCasesError):
        await analyzer.analyze_failures(exp)

@pytest.mark.asyncio
async def test_failure_analyzer_invalid_data():
    results = [
        EvaluationResultEntry(
            test_case_id="",  # Invalid empty ID
            prompt="P1",
            generated_output="G1",
            scores={},
            latency_ms=10.0,
            timestamp=datetime.now(timezone.utc),
            passed=False,
            success=True
        )
    ]
    exp = make_mock_experiment(results)
    provider = MockLLMProvider()
    analyzer = FailureAnalyzer(provider)
    
    with pytest.raises(InvalidExperimentDataError):
        await analyzer.analyze_failures(exp)

@pytest.mark.asyncio
async def test_failure_analyzer_empty_results():
    exp = make_mock_experiment([])
    provider = MockLLMProvider()
    analyzer = FailureAnalyzer(provider)
    
    with pytest.raises(InvalidExperimentDataError):
        await analyzer.analyze_failures(exp)


# --- 4. Test End-to-End Success Path ---
@pytest.mark.asyncio
async def test_failure_analyzer_success():
    results = [
        EvaluationResultEntry(
            test_case_id="c1",
            prompt="P1",
            generated_output="G1",
            expected_output="E1",
            scores={"s": 0.4},
            latency_ms=100.0,
            timestamp=datetime.now(timezone.utc),
            passed=False,
            success=True
        ),
        EvaluationResultEntry(
            test_case_id="c2",
            prompt="P2",
            generated_output="",
            expected_output="E2",
            scores={},
            latency_ms=50.0,
            timestamp=datetime.now(timezone.utc),
            passed=False,
            success=False,
            error_message="API Rate limit hit"
        )
    ]
    exp = make_mock_experiment(results)
    
    # Mock response JSON
    mock_json_response = {
        "detected_categories": ["Hallucination", "Instruction Following Errors"],
        "grouped_failures": {
            "Hallucination": [
                {
                    "test_case_ids": ["c1"],
                    "description": "Model response lacks factual grounding."
                }
            ],
            "Instruction Following Errors": [
                {
                    "test_case_ids": ["c2"],
                    "description": "API connection failure preventing generation."
                }
            ]
        },
        "category_counts": {
            "Hallucination": 1,
            "Instruction Following Errors": 1
        },
        "identified_patterns": ["Hallucinations on low scores", "API timeout failure patterns"],
        "ai_generated_summary": "We observed hallucinations and API failure patterns.",
        "recommendations": ["Improve prompts", "Increase API timeouts"]
    }
    
    # Set up LLM mock response
    provider = MockLLMProvider()
    provider.generate = AsyncMock(return_value=LLMResponse(
        text=json.dumps(mock_json_response),
        provider_name="mock-provider",
        model_name="mock-model",
        latency_ms=250.0,
        timestamp=datetime.now(timezone.utc),
        request_id="req_123",
        success=True,
        token_usage=None,
        raw_response={}
    ))
    
    analyzer = FailureAnalyzer(provider)
    result = await analyzer.analyze_failures(exp)
    
    assert isinstance(result, FailureAnalysisResult)
    assert result.experiment_id == "exp_test_123"
    assert result.number_of_failed_cases == 2
    assert "Hallucination" in result.detected_categories
    assert "Instruction Following Errors" in result.detected_categories
    assert result.category_counts["Hallucination"] == 1
    assert result.category_counts["Instruction Following Errors"] == 1
    
    # Check that grouped failures are parsed into model objects correctly
    c1_group = result.grouped_failures["Hallucination"][0]
    assert isinstance(c1_group, GroupedFailure)
    assert c1_group.test_case_ids == ["c1"]
    assert c1_group.description == "Model response lacks factual grounding."
    
    assert result.identified_patterns == ["Hallucinations on low scores", "API timeout failure patterns"]
    assert "observed hallucinations" in result.ai_generated_summary
    assert result.recommendations == ["Improve prompts", "Increase API timeouts"]
    assert result.provider_used_for_analysis == "mock-provider"


# --- 5. Test Custom Categories & Resolution ---
@pytest.mark.asyncio
async def test_failure_analyzer_custom_categories():
    results = [
        EvaluationResultEntry(
            test_case_id="c1",
            prompt="P1",
            generated_output="G1",
            expected_output="E1",
            scores={"s": 0.4},
            latency_ms=10.0,
            timestamp=datetime.now(timezone.utc),
            passed=False,
            success=True
        )
    ]
    exp = make_mock_experiment(results)
    
    # Provider returns a category NOT in custom allowed list: "BadCategory"
    mock_json_response = {
        "detected_categories": ["BadCategory"],
        "grouped_failures": {
            "BadCategory": [
                {
                    "test_case_ids": ["c1"],
                    "description": "Unsupported error type."
                }
            ]
        },
        "category_counts": {
            "BadCategory": 1
        },
        "identified_patterns": ["Pattern"],
        "ai_generated_summary": "Summary",
        "recommendations": ["Recommendation"]
    }
    
    provider = MockLLMProvider()
    provider.generate = AsyncMock(return_value=LLMResponse(
        text=json.dumps(mock_json_response),
        provider_name="mock-provider",
        model_name="mock-model",
        latency_ms=10.0,
        timestamp=datetime.now(timezone.utc),
        request_id="req_123",
        success=True,
        token_usage=None,
        raw_response={}
    ))
    
    # Restrict categories to ["MyCustomCategory", "Other"]
    custom_categories = ["MyCustomCategory", "Other"]
    analyzer = FailureAnalyzer(provider, allowed_categories=custom_categories)
    result = await analyzer.analyze_failures(exp)
    
    # "BadCategory" is not in custom categories, so it should map to "Other"
    assert "Other" in result.detected_categories
    assert "BadCategory" not in result.detected_categories
    assert result.category_counts["Other"] == 1
    assert result.grouped_failures["Other"][0].test_case_ids == ["c1"]


# --- 6. Test Error Propagation ---
@pytest.mark.asyncio
async def test_failure_analyzer_provider_fails():
    results = [
        EvaluationResultEntry(
            test_case_id="c1",
            prompt="P1",
            generated_output="G1",
            expected_output="E1",
            scores={"s": 0.4},
            latency_ms=10.0,
            timestamp=datetime.now(timezone.utc),
            passed=False,
            success=True
        )
    ]
    exp = make_mock_experiment(results)
    
    provider = MockLLMProvider()
    # Provider generate raises an exception
    provider.generate = AsyncMock(side_effect=Exception("Provider timed out or API key invalid"))
    
    analyzer = FailureAnalyzer(provider)
    with pytest.raises(AnalysisExecutionError):
        await analyzer.analyze_failures(exp)
