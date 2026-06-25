import logging
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from backend.experiments.models import Experiment, EvaluationResultEntry
from backend.experiments.storage import BaseStorage
from backend.experiments.exceptions import ExperimentNotFoundError

logger = logging.getLogger("llm_platform.experiments.manager")

class ExperimentManager:
    """Manager class for creating, saving, loading, listing, and deleting experiments."""

    def __init__(self, storage: BaseStorage):
        self.storage = storage

    def create_experiment(
        self,
        dataset_name: str,
        provider: str,
        model: str,
        evaluation_configuration: Dict[str, Any],
        evaluation_results: List[EvaluationResultEntry],
        evaluation_metrics: Dict[str, float],
        average_latency: float,
        total_number_of_test_cases: int,
        passed_test_cases: int,
        failed_test_cases: int,
        dataset_version: Optional[str] = None,
        experiment_id: Optional[str] = None
    ) -> Experiment:
        """
        Creates and returns a new strongly-typed Experiment model instance with a unique ID.
        """
        if not experiment_id:
            unique_suffix = str(uuid.uuid4())[:8]
            timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            experiment_id = f"exp_{timestamp_str}_{unique_suffix}"

        logger.info(f"Creating experiment structure: {experiment_id} for dataset '{dataset_name}'")

        return Experiment(
            experiment_id=experiment_id,
            timestamp=datetime.now(timezone.utc),
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            provider=provider,
            model=model,
            evaluation_configuration=evaluation_configuration,
            evaluation_results=evaluation_results,
            evaluation_metrics=evaluation_metrics,
            average_latency=average_latency,
            total_number_of_test_cases=total_number_of_test_cases,
            passed_test_cases=passed_test_cases,
            failed_test_cases=failed_test_cases
        )

    async def save_experiment(self, experiment: Experiment) -> None:
        """Saves an experiment run to storage."""
        logger.info(f"Saving experiment run: {experiment.experiment_id}")
        await self.storage.save(experiment)

    async def load_experiment(self, experiment_id: str) -> Experiment:
        """Loads and returns an experiment run from storage."""
        logger.info(f"Loading experiment: {experiment_id}")
        experiment = await self.storage.get(experiment_id)
        if not experiment:
            raise ExperimentNotFoundError(f"Experiment with ID '{experiment_id}' not found.")
        return experiment

    async def list_experiments(self) -> List[Experiment]:
        """Lists all previous experiment runs in storage."""
        logger.info("Listing all experiments in storage")
        return await self.storage.list()

    async def delete_experiment(self, experiment_id: str) -> None:
        """Deletes an experiment run from storage."""
        logger.info(f"Deleting experiment: {experiment_id}")
        # Verify it exists first
        await self.load_experiment(experiment_id)
        await self.storage.delete(experiment_id)
