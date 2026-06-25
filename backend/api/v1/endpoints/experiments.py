from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
import tempfile
import os

from backend.experiments.models import ExperimentConfig, ExperimentRun
from backend.experiments.runner import ExperimentRunner
from backend.core.database import get_storage
from backend.reporting.exporter import JSONExporter, CSVExporter

router = APIRouter()
storage = get_storage()
runner = ExperimentRunner()

@router.post("/", response_model=ExperimentRun, status_code=202)
async def run_experiment(config: ExperimentConfig):
    """
    Triggers an LLM evaluation experiment using specified dataset, provider, and configuration.
    """
    try:
        result = await runner.run(config)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start experiment: {str(e)}")

@router.get("/", response_model=list[dict])
async def list_experiments():
    """
    Retrieve all run experiments and their summary metrics.
    """
    return await storage.list_experiments()

@router.get("/{experiment_id}", response_model=dict)
async def get_experiment(experiment_id: str):
    """
    Get detailed results, inputs, outputs, and metrics of a single experiment.
    """
    result = await storage.get_experiment(experiment_id)
    if not result:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return result

@router.get("/{experiment_id}/export")
async def export_experiment(experiment_id: str, format: str = Query("json", pattern="^(json|csv)$")):
    """
    Export experiment results to CSV or JSON formats.
    """
    result = await storage.get_experiment(experiment_id)
    if not result:
        raise HTTPException(status_code=404, detail="Experiment not found")

    exporter = JSONExporter() if format == "json" else CSVExporter()
    
    # Create temporary file to output
    fd, temp_path = tempfile.mkstemp(suffix=f".{format}")
    os.close(fd)
    
    try:
        exporter.export(result, temp_path)
        filename = f"experiment_{experiment_id}.{format}"
        return FileResponse(
            temp_path, 
            media_type="application/octet-stream", 
            filename=filename
        )
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Failed to export: {str(e)}")
