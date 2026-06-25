import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json
import tempfile
import os
from datetime import datetime, timezone

from backend.providers.base import LLMProvider
from backend.providers.response import LLMResponse, TokenUsage
from backend.evaluation import (
    TestCase, EvaluationDataset, MetricScore, TestCaseResult,
    EvaluationSummary, EvaluationReport, DatasetLoader,
    ExactMatchMetric, JaccardSimilarityMetric, TestCaseEvaluator,
    EvaluationEngine, InvalidDatasetError, MetricExecutionError
)

# Mock provider for testing
class MockLLMProvider(LLMProvider):
    def __init__(self, model_name="mock-model", text="Paris"):
        super().__init__(model_name, "mock-key")
        self.text = text
        self.generate_mock = AsyncMock(return_value=LLMResponse(
            text=self.text,
            provider_name="mock",
            model_name=self.model_name,
            latency_ms=10.0,
            timestamp=datetime.now(timezone.utc),
            request_id="req-123",
            success=True
        ))

    @property
    def provider_name(self) -> str:
        return "mock"

    async def _generate(self, prompt: str, system_prompt: str | None = None, temperature: float = 0.0, max_tokens: int = 1024):
        pass

    async def generate(self, prompt: str, system_prompt: str | None = None, temperature: float = 0.0, max_tokens: int = 1024) -> LLMResponse:
        return await self.generate_mock(prompt, system_prompt, temperature, max_tokens)


def test_dataset_loader_valid():
    raw_data = {
        "id": "ds-123",
        "name": "General Q&A",
        "test_cases": [
            {
                "id": "tc-1",
                "prompt": "What color is the sky?",
                "expected_output": "Blue",
                "context": ["The sky is blue during the day."],
                "metadata": {"category": "nature"}
            }
        ]
    }
    dataset = DatasetLoader.load_from_dict(raw_data)
    assert isinstance(dataset, EvaluationDataset)
    assert dataset.id == "ds-123"
    assert len(dataset.test_cases) == 1
    assert dataset.test_cases[0].id == "tc-1"
    assert dataset.test_cases[0].prompt == "What color is the sky?"


def test_dataset_loader_invalid_top_level():
    bad_data = {
        "id": "ds-123",
        "test_cases": []
    }
    with pytest.raises(InvalidDatasetError) as exc:
        DatasetLoader.load_from_dict(bad_data)
    assert "name" in str(exc.value)


def test_dataset_loader_invalid_case():
    bad_data = {
        "id": "ds-123",
        "name": "QA Dataset",
        "test_cases": [
            {
                "id": "tc-1"
            }
        ]
    }
    with pytest.raises(InvalidDatasetError) as exc:
        DatasetLoader.load_from_dict(bad_data)
    assert "missing a prompt string" in str(exc.value)


def test_exact_match_metric():
    metric = ExactMatchMetric(case_sensitive=False)
    
    score = metric.measure(
        MagicMock(actual_output="paris", expected_output="Paris")
    )
    assert score == 1.0
    assert metric.is_successful() is True

    score = metric.measure(
        MagicMock(actual_output="london", expected_output="Paris")
    )
    assert score == 0.0
    assert metric.is_successful() is False


def test_jaccard_similarity_metric():
    metric = JaccardSimilarityMetric(threshold=0.5)
    
    score = metric.measure(
        MagicMock(actual_output="the sky", expected_output="the sky is blue")
    )
    assert score == 0.5
    assert metric.is_successful() is True

    score = metric.measure(
        MagicMock(actual_output="clouds", expected_output="the sky is blue")
    )
    assert score < 0.25
    assert metric.is_successful() is False


@pytest.mark.asyncio
async def test_test_case_evaluator():
    test_case = TestCase(id="tc-1", prompt="What is 2+2?", expected_output="4")
    metrics = [ExactMatchMetric()]
    
    scores = await TestCaseEvaluator.evaluate_case(
        test_case=test_case,
        generated_output="4",
        metrics=metrics
    )
    assert len(scores) == 1
    assert scores[0].name == "Exact Match"
    assert scores[0].score == 1.0
    assert scores[0].passed is True


@pytest.mark.asyncio
async def test_evaluation_engine_success():
    provider = MockLLMProvider(text="Paris")
    metrics = [ExactMatchMetric(threshold=1.0)]
    engine = EvaluationEngine(provider=provider, metrics=metrics)

    dataset = EvaluationDataset(
        id="ds-test",
        name="Test DS",
        test_cases=[
            TestCase(id="tc-1", prompt="Capital of France?", expected_output="Paris"),
            TestCase(id="tc-2", prompt="Capital of UK?", expected_output="London")
        ]
    )

    report = await engine.evaluate(dataset)
    assert isinstance(report, EvaluationReport)
    assert report.dataset_id == "ds-test"
    assert report.summary.total_test_cases == 2
    assert report.summary.completed_test_cases == 2
    assert report.summary.passed_test_cases == 1
    assert report.summary.failed_test_cases == 1
    assert report.summary.error_count == 0
    assert "Exact Match" in report.summary.summary_metrics
    assert report.summary.summary_metrics["Exact Match"] == 0.5

    assert report.results[0].test_case_id == "tc-1"
    assert report.results[0].success is True
    assert report.results[0].generated_output == "Paris"
    assert report.results[0].scores[0].passed is True

    assert report.results[1].test_case_id == "tc-2"
    assert report.results[1].success is True
    assert report.results[1].generated_output == "Paris"
    assert report.results[1].scores[0].passed is False


@pytest.mark.asyncio
async def test_evaluation_engine_graceful_error_handling():
    provider = MockLLMProvider()
    call_count = 0
    async def mock_generate(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return LLMResponse(
                text="Ok", provider_name="mock", model_name="mock", 
                latency_ms=5, timestamp=datetime.now(timezone.utc), request_id="id", success=True
            )
        raise RuntimeError("LLM connection failed")
        
    provider.generate_mock = AsyncMock(side_effect=mock_generate)
    
    metrics = [ExactMatchMetric()]
    engine = EvaluationEngine(provider=provider, metrics=metrics)
    
    dataset = EvaluationDataset(
        id="ds-error",
        name="Error DS",
        test_cases=[
            TestCase(id="tc-1", prompt="Good case", expected_output="Ok"),
            TestCase(id="tc-2", prompt="Failed case", expected_output="Ok")
        ]
    )

    report = await engine.evaluate(dataset)
    assert report.summary.total_test_cases == 2
    assert report.summary.completed_test_cases == 1
    assert report.summary.passed_test_cases == 1
    assert report.summary.error_count == 1

    assert report.results[0].test_case_id == "tc-1"
    assert report.results[0].success is True

    assert report.results[1].test_case_id == "tc-2"
    assert report.results[1].success is False
    assert "LLM connection failed" in report.results[1].error_message
