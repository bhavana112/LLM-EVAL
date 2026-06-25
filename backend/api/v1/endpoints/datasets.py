from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from backend.datasets.loader import get_dataset_loader
from backend.core.database import get_storage

router = APIRouter()
loader = get_dataset_loader()
storage = get_storage()

class DatasetEntry(BaseModel):
    id: Optional[str] = None
    prompt: str
    expected_output: Optional[str] = None

class DatasetCreate(BaseModel):
    id: str = Field(..., description="Unique dataset identifier")
    name: str = Field(..., description="Human-readable dataset name")
    entries: List[DatasetEntry] = Field(default_factory=list)

@router.post("/", status_code=201)
async def create_dataset(dataset: DatasetCreate):
    """
    Registers a new dataset with prompt entries.
    """
    existing = await storage.get_dataset(dataset.id)
    if existing:
        raise HTTPException(status_code=400, detail=f"Dataset with ID '{dataset.id}' already exists.")
        
    entries_list = [entry.model_dump() for entry in dataset.entries]
    await loader.register(
        dataset_id=dataset.id,
        name=dataset.name,
        entries=entries_list
    )
    return {"status": "success", "dataset_id": dataset.id}

@router.get("/")
async def list_datasets():
    """
    List all registered evaluation datasets.
    """
    return await storage.list_datasets()

@router.get("/{dataset_id}")
async def get_dataset(dataset_id: str):
    """
    Retrieve dataset details by ID.
    """
    dataset = await loader.load(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset
