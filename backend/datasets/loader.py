from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from backend.core.database import get_storage

class BaseDatasetLoader(ABC):
    """Abstract interface for loading and registering evaluation datasets."""

    @abstractmethod
    async def load(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Load dataset by ID."""
        pass

    @abstractmethod
    async def register(self, dataset_id: str, name: str, entries: List[Dict[str, Any]]) -> None:
        """Register a new dataset in the platform."""
        pass


class SimpleDatasetLoader(BaseDatasetLoader):
    """Simple loader backed by storage for datasets."""

    def __init__(self):
        self.storage = get_storage()

    async def load(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        return await self.storage.get_dataset(dataset_id)

    async def register(self, dataset_id: str, name: str, entries: List[Dict[str, Any]]) -> None:
        dataset = {
            "id": dataset_id,
            "name": name,
            "entries": entries
        }
        await self.storage.save_dataset(dataset_id, dataset)


_loader_instance: Optional[BaseDatasetLoader] = None

def get_dataset_loader() -> BaseDatasetLoader:
    global _loader_instance
    if _loader_instance is None:
        _loader_instance = SimpleDatasetLoader()
    return _loader_instance
