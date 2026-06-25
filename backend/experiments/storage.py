import abc
import json
import logging
import os
import shutil
from typing import List, Optional
from pydantic import ValidationError

from backend.experiments.models import Experiment
from backend.experiments.exceptions import (
    StorageWriteError, ExperimentNotFoundError, InvalidExperimentError
)

logger = logging.getLogger("llm_platform.experiments.storage")

class BaseStorage(abc.ABC):
    """Abstract Base Class for all experiment storage backends."""

    @abc.abstractmethod
    async def save(self, experiment: Experiment) -> None:
        """Saves an experiment run to the storage backend."""
        pass

    @abc.abstractmethod
    async def get(self, experiment_id: str) -> Optional[Experiment]:
        """Retrieves a single experiment run by ID."""
        pass

    @abc.abstractmethod
    async def list(self) -> List[Experiment]:
        """Lists all stored experiment runs."""
        pass

    @abc.abstractmethod
    async def delete(self, experiment_id: str) -> None:
        """Deletes an experiment run from storage."""
        pass


class JSONStorage(BaseStorage):
    """Local file-based storage backend using JSON files organized in folders."""

    def __init__(self, base_dir: str = "experiments"):
        self.base_dir = base_dir

    def _get_experiment_dir(self, experiment_id: str) -> str:
        return os.path.join(self.base_dir, experiment_id)

    def _get_file_path(self, experiment_id: str) -> str:
        return os.path.join(self._get_experiment_dir(experiment_id), "experiment.json")

    async def save(self, experiment: Experiment) -> None:
        exp_dir = self._get_experiment_dir(experiment.experiment_id)
        file_path = self._get_file_path(experiment.experiment_id)

        try:
            os.makedirs(exp_dir, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(experiment.model_dump_json(indent=2))
            logger.info(f"Experiment JSON written to: {file_path}")
        except Exception as e:
            logger.error(f"Failed to write experiment JSON to {file_path}: {str(e)}")
            raise StorageWriteError(f"Failed to save experiment JSON to filesystem: {str(e)}", details=str(e))

    async def get(self, experiment_id: str) -> Optional[Experiment]:
        file_path = self._get_file_path(experiment_id)
        
        if not os.path.exists(file_path):
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse corrupted experiment JSON: {file_path}")
            raise InvalidExperimentError(f"Experiment JSON is corrupted or invalid: {file_path}", details=str(e))
        except Exception as e:
            logger.error(f"Failed to read experiment file {file_path}: {str(e)}")
            raise InvalidExperimentError(f"Failed to read experiment file: {str(e)}", details=str(e))

        try:
            return Experiment(**data)
        except ValidationError as e:
            logger.error(f"Experiment JSON fails validation schema checks: {file_path}")
            raise InvalidExperimentError(f"Experiment schema validation failed: {str(e)}", details=str(e))

    async def list(self) -> List[Experiment]:
        if not os.path.exists(self.base_dir):
            return []

        experiments: List[Experiment] = []
        
        try:
            for item in os.listdir(self.base_dir):
                item_path = os.path.join(self.base_dir, item)
                if os.path.isdir(item_path):
                    file_path = os.path.join(item_path, "experiment.json")
                    if os.path.exists(file_path):
                        try:
                            exp = await self.get(item)
                            if exp:
                                experiments.append(exp)
                        except InvalidExperimentError as e:
                            logger.warning(f"Skipping corrupted or invalid experiment '{item}': {str(e)}")
        except Exception as e:
            logger.error(f"Failed to list experiments directory: {str(e)}")
            
        experiments.sort(key=lambda x: x.timestamp, reverse=True)
        return experiments

    async def delete(self, experiment_id: str) -> None:
        exp_dir = self._get_experiment_dir(experiment_id)
        if not os.path.exists(exp_dir):
            raise ExperimentNotFoundError(f"Experiment directory '{exp_dir}' not found.")
            
        try:
            shutil.rmtree(exp_dir)
            logger.info(f"Deleted experiment directory: {exp_dir}")
        except Exception as e:
            logger.error(f"Failed to delete directory {exp_dir}: {str(e)}")
            raise StorageWriteError(f"Failed to delete experiment folder: {str(e)}", details=str(e))
