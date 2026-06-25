from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import os
import json

class BaseStorage(ABC):
    """Abstract interface for all storage backends (SQLite, PostgreSQL, MongoDB, InMemory, JSON)."""

    @abstractmethod
    async def save_experiment(self, experiment_id: str, data: Dict[str, Any]) -> None:
        """Save experiment details and configuration."""
        pass

    @abstractmethod
    async def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve experiment details."""
        pass

    @abstractmethod
    async def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments."""
        pass

    @abstractmethod
    async def save_dataset(self, dataset_id: str, data: Dict[str, Any]) -> None:
        """Save evaluation dataset metadata and entries."""
        pass

    @abstractmethod
    async def get_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve dataset metadata and entries."""
        pass

    @abstractmethod
    async def list_datasets(self) -> List[Dict[str, Any]]:
        """List all available datasets."""
        pass


class InMemoryStorage(BaseStorage):
    """Simple in-memory storage implementation for local dev and testing."""
    
    def __init__(self):
        self._experiments: Dict[str, Dict[str, Any]] = {}
        self._datasets: Dict[str, Dict[str, Any]] = {}

    async def save_experiment(self, experiment_id: str, data: Dict[str, Any]) -> None:
        self._experiments[experiment_id] = data

    async def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        return self._experiments.get(experiment_id)

    async def list_experiments(self) -> List[Dict[str, Any]]:
        return list(self._experiments.values())

    async def save_dataset(self, dataset_id: str, data: Dict[str, Any]) -> None:
        self._datasets[dataset_id] = data

    async def get_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        return self._datasets.get(dataset_id)

    async def list_datasets(self) -> List[Dict[str, Any]]:
        return list(self._datasets.values())


class JSONDatabaseStorage(BaseStorage):
    """
    Adapter class wrapping backend.experiments.storage.JSONStorage 
    to support unified file-based database outputs.
    """
    def __init__(self, base_dir: str = "experiments"):
        from backend.experiments.storage import JSONStorage
        self._storage = JSONStorage(base_dir=base_dir)

    async def save_experiment(self, experiment_id: str, data: Dict[str, Any]) -> None:
        from backend.experiments.models import Experiment
        # Adapter to map old ExperimentRun schemas to new Experiment models
        if "config" in data and "results" in data:
            from datetime import datetime, timezone
            from backend.experiments.models import EvaluationResultEntry
            config = data.get("config", {}) or {}
            results_raw = data.get("results", []) or []
            
            eval_results = []
            passed_count = 0
            failed_count = 0
            total_latency = 0.0
            
            for r in results_raw:
                scores = r.get("metrics", {}) or {}
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
            metrics = {k: v for k, v in data.get("summary_metrics", {}).items() if k != "avg_latency_ms"}
            
            c_str = data.get("created_at")
            cdt = datetime.now(timezone.utc)
            if c_str:
                try:
                    cdt = datetime.fromisoformat(c_str.replace("Z", "+00:00"))
                except ValueError:
                    pass

            experiment = Experiment(
                experiment_id=experiment_id,
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
        else:
            experiment = Experiment(**data)
            
        await self._storage.save(experiment)

    async def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        experiment = await self._storage.get(experiment_id)
        return experiment.model_dump() if experiment else None

    async def list_experiments(self) -> List[Dict[str, Any]]:
        experiments = await self._storage.list()
        return [e.model_dump() for e in experiments]

    async def save_dataset(self, dataset_id: str, data: Dict[str, Any]) -> None:
        os.makedirs("datasets", exist_ok=True)
        file_path = os.path.join("datasets", f"{dataset_id}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    async def get_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        file_path = os.path.join("datasets", f"{dataset_id}.json")
        if not os.path.exists(file_path):
            return None
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    async def list_datasets(self) -> List[Dict[str, Any]]:
        if not os.path.exists("datasets"):
            return []
        datasets = []
        for file in os.listdir("datasets"):
            if file.endswith(".json"):
                file_path = os.path.join("datasets", file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        datasets.append(json.load(f))
                except Exception:
                    pass
        return datasets


# Simple factory/getter for storage backend
_storage_instance: Optional[BaseStorage] = None

def get_storage() -> BaseStorage:
    global _storage_instance
    if _storage_instance is None:
        from backend.core.config import settings
        backend = settings.DATABASE_BACKEND.lower()
        if backend == "json":
            _storage_instance = JSONDatabaseStorage()
        else:
            # Defaults to memory storage
            _storage_instance = InMemoryStorage()
    return _storage_instance
