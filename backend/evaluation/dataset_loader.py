import json
import logging
from typing import Dict, Any
from pydantic import ValidationError

from backend.evaluation.models import EvaluationDataset
from backend.evaluation.exceptions import InvalidDatasetError

logger = logging.getLogger("llm_platform.evaluation.dataset_loader")

class DatasetLoader:
    """Loads and validates evaluation benchmark datasets from files or dictionary inputs."""

    @staticmethod
    def load_from_file(filepath: str) -> EvaluationDataset:
        """Loads evaluation dataset from a JSON file."""
        logger.info(f"Loading dataset from file: {filepath}")
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            return DatasetLoader.load_from_dict(data)
        except FileNotFoundError as e:
            raise InvalidDatasetError(f"Dataset file not found: {filepath}", details=str(e))
        except json.JSONDecodeError as e:
            raise InvalidDatasetError(f"Malformed JSON in dataset file {filepath}", details=str(e))
        except Exception as e:
            raise InvalidDatasetError(f"Failed to read dataset file {filepath}", details=str(e))

    @staticmethod
    def load_from_dict(data: Dict[str, Any]) -> EvaluationDataset:
        """Parses and validates evaluation dataset from raw dictionary input."""
        if not isinstance(data, dict):
            raise InvalidDatasetError("Dataset input data must be a dictionary object")

        # Validate presence of required top-level attributes before full schema checks
        if "id" not in data or not data["id"]:
            raise InvalidDatasetError("Dataset 'id' field is required and cannot be empty")
        if "name" not in data or not data["name"]:
            raise InvalidDatasetError("Dataset 'name' field is required and cannot be empty")
        if "test_cases" not in data or not isinstance(data["test_cases"], list):
            raise InvalidDatasetError("Dataset must contain a list of 'test_cases'")

        for i, tc in enumerate(data.get("test_cases", [])):
            if not isinstance(tc, dict):
                raise InvalidDatasetError(f"Test case at index {i} must be a dictionary object")
            if "id" not in tc or not tc["id"]:
                raise InvalidDatasetError(f"Test case at index {i} is missing a unique 'id'")
            if "prompt" not in tc or not tc["prompt"]:
                raise InvalidDatasetError(f"Test case '{tc.get('id', i)}' is missing a prompt string")

        try:
            return EvaluationDataset(**data)
        except ValidationError as e:
            raise InvalidDatasetError(
                "Dataset failed Pydantic schema validation checks",
                details=str(e)
            )
