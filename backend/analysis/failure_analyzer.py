import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from backend.providers.base import LLMProvider
from backend.experiments.models import Experiment
from backend.analysis.models import FailureAnalysisResult, GroupedFailure
from backend.analysis.exceptions import (
    NoFailedCasesError, InvalidExperimentDataError, AnalysisExecutionError
)
from backend.analysis.prompt_builder import PromptBuilder
from backend.analysis.utils import extract_json

logger = logging.getLogger("llm_platform.analysis.failure_analyzer")

DEFAULT_CATEGORIES = [
    "Hallucination",
    "Missing Context",
    "Incorrect Reasoning",
    "Incomplete Answer",
    "Formatting Problems",
    "Instruction Following Errors",
    "Factual Errors",
    "Unsupported Claims",
    "Low Relevance",
    "Other"
]

class FailureAnalyzer:
    """Orchestrates AI-Assisted Failure Analysis on completed experiments."""

    def __init__(
        self,
        provider: LLMProvider,
        allowed_categories: Optional[List[str]] = None,
        prompt_builder: Optional[PromptBuilder] = None
    ):
        """
        provider: The LLMProvider wrapper to use for running the analysis.
        allowed_categories: Custom list of failure categories (defaults to DEFAULT_CATEGORIES).
        prompt_builder: Custom PromptBuilder instance.
        """
        self.provider = provider
        self.allowed_categories = allowed_categories or DEFAULT_CATEGORIES
        self.prompt_builder = prompt_builder or PromptBuilder()

    async def analyze_failures(self, experiment: Experiment) -> FailureAnalysisResult:
        """
        Filters failed cases from the experiment, runs them through the LLM provider,
        and returns a strongly-typed FailureAnalysisResult.
        """
        logger.info(f"Starting failure analysis for experiment: {experiment.experiment_id}")

        if not experiment.evaluation_results:
            logger.warning(f"Experiment {experiment.experiment_id} has no evaluation results.")
            raise InvalidExperimentDataError("Experiment has no evaluation results to analyze.")

        # Identify failed test cases
        # Failures include:
        # 1. passed is False
        # 2. success is False (indicating engine evaluation or generation failure)
        # 3. error_message is populated
        failed_cases: List[Dict[str, Any]] = []
        for case in experiment.evaluation_results:
            # Basic validation
            if not hasattr(case, "test_case_id") or not case.test_case_id:
                raise InvalidExperimentDataError("Evaluation result entry is missing test_case_id.")

            is_failed = not case.passed or not case.success or case.error_message is not None
            if is_failed:
                # Convert Pydantic object to dict for prompt builder
                failed_cases.append(case.model_dump())

        num_failures = len(failed_cases)
        logger.info(f"Found {num_failures} failed cases out of {len(experiment.evaluation_results)} total cases.")

        if num_failures == 0:
            logger.info("No failed cases detected. Raising NoFailedCasesError.")
            raise NoFailedCasesError(f"Experiment '{experiment.experiment_id}' has no failed cases.")

        # Build prompt
        prompt = self.prompt_builder.build_prompt(
            failed_cases=failed_cases,
            total_cases=len(experiment.evaluation_results),
            allowed_categories=self.allowed_categories
        )

        system_instruction = self.prompt_builder.system_instruction

        logger.info("Sending prompt to provider for failure analysis...")
        try:
            response = await self.provider.generate(
                prompt=prompt,
                system_prompt=system_instruction,
                temperature=0.0
            )
        except Exception as e:
            logger.error(f"LLM Provider invocation failed: {str(e)}")
            raise AnalysisExecutionError(f"Failed to execute failure analysis: {str(e)}", details=str(e))

        if not response or not response.success or not response.text:
            logger.error("LLM Provider returned an unsuccessful or empty response.")
            raise AnalysisExecutionError("LLM Provider returned an empty or unsuccessful response.")

        # Extract and parse JSON
        try:
            parsed_data = extract_json(response.text)
        except Exception as e:
            logger.error(f"Failed to parse JSON from provider response: {str(e)}")
            raise AnalysisExecutionError(f"Failed to parse structured JSON from provider output: {str(e)}", details=str(e))

        # Validate and construct FailureAnalysisResult
        try:
            raw_grouped = parsed_data.get("grouped_failures", {})
            grouped_failures: Dict[str, List[GroupedFailure]] = {}
            category_counts: Dict[str, int] = {}

            # Populate categories and compute totals dynamically
            for category, groups in raw_grouped.items():
                # Validate category is in allowed list
                resolved_cat = category if category in self.allowed_categories else "Other"
                if resolved_cat not in grouped_failures:
                    grouped_failures[resolved_cat] = []
                
                cat_count = 0
                for group in groups:
                    ids = group.get("test_case_ids", [])
                    desc = group.get("description", "No description provided")
                    grouped_failures[resolved_cat].append(
                        GroupedFailure(test_case_ids=ids, description=desc)
                    )
                    cat_count += len(ids)
                
                category_counts[resolved_cat] = category_counts.get(resolved_cat, 0) + cat_count

            detected_categories = list(grouped_failures.keys())

            result = FailureAnalysisResult(
                experiment_id=experiment.experiment_id,
                analysis_timestamp=datetime.now(timezone.utc),
                number_of_failed_cases=num_failures,
                detected_categories=detected_categories,
                grouped_failures=grouped_failures,
                category_counts=category_counts,
                identified_patterns=parsed_data.get("identified_patterns", []),
                ai_generated_summary=parsed_data.get("ai_generated_summary", "No summary generated."),
                recommendations=parsed_data.get("recommendations", []),
                provider_used_for_analysis=self.provider.provider_name
            )
            logger.info("Failure analysis report successfully generated.")
            return result

        except Exception as e:
            logger.error(f"Failed to map parsed data to Pydantic models: {str(e)}")
            raise AnalysisExecutionError(f"Schema mapping of provider JSON output failed: {str(e)}", details=str(e))
