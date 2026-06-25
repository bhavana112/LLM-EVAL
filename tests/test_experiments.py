import pytest
import os
import shutil
import tempfile
from datetime import datetime, timezone

from backend.experiments import (
    ExperimentManager, JSONStorage, Experiment, EvaluationResultEntry,
    ExperimentNotFoundError, InvalidExperimentError
)

@pytest.fixture
def temp_storage_dir():
    # Use temporary directory for file-based tests
    dirpath = tempfile.mkdtemp()
    yield dirpath
    if os.path.exists(dirpath):
        shutil.rmtree(dirpath)


@pytest.mark.asyncio
async def test_create_and_save_experiment(temp_storage_dir):
    storage = JSONStorage(base_dir=temp_storage_dir)
    manager = ExperimentManager(storage=storage)
    
    results = [
        EvaluationResultEntry(
            test_case_id="tc-1",
            prompt="Hello?",
            generated_output="Hi",
            expected_output="Hello",
            scores={"Exact Match": 0.0},
            latency_ms=12.5,
            timestamp=datetime.now(timezone.utc),
            passed=False
        )
    ]
    
    experiment = manager.create_experiment(
        dataset_name="Test DS",
        provider="openai",
        model="gpt-4o",
        evaluation_configuration={"temperature": 0.0},
        evaluation_results=results,
        evaluation_metrics={"Exact Match": 0.0},
        average_latency=12.5,
        total_number_of_test_cases=1,
        passed_test_cases=0,
        failed_test_cases=1
    )
    
    assert isinstance(experiment, Experiment)
    assert experiment.experiment_id.startswith("exp_")
    
    await manager.save_experiment(experiment)
    
    exp_folder = os.path.join(temp_storage_dir, experiment.experiment_id)
    file_path = os.path.join(exp_folder, "experiment.json")
    assert os.path.exists(exp_folder)
    assert os.path.exists(file_path)
    
    loaded = await manager.load_experiment(experiment.experiment_id)
    assert loaded.experiment_id == experiment.experiment_id
    assert loaded.dataset_name == "Test DS"
    assert len(loaded.evaluation_results) == 1
    assert loaded.evaluation_results[0].test_case_id == "tc-1"


@pytest.mark.asyncio
async def test_list_experiments_ordering(temp_storage_dir):
    storage = JSONStorage(base_dir=temp_storage_dir)
    manager = ExperimentManager(storage=storage)
    
    exp1 = manager.create_experiment(
        dataset_name="DS1", provider="mock", model="mock", evaluation_configuration={},
        evaluation_results=[], evaluation_metrics={}, average_latency=1.0,
        total_number_of_test_cases=0, passed_test_cases=0, failed_test_cases=0,
        experiment_id="exp_first"
    )
    # Set slight delay to ensure different timestamps
    import asyncio
    await asyncio.sleep(0.1)
    
    exp2 = manager.create_experiment(
        dataset_name="DS2", provider="mock", model="mock", evaluation_configuration={},
        evaluation_results=[], evaluation_metrics={}, average_latency=1.0,
        total_number_of_test_cases=0, passed_test_cases=0, failed_test_cases=0,
        experiment_id="exp_second"
    )
    
    await manager.save_experiment(exp1)
    await manager.save_experiment(exp2)
    
    experiments = await manager.list_experiments()
    assert len(experiments) == 2
    assert experiments[0].experiment_id == "exp_second"
    assert experiments[1].experiment_id == "exp_first"


@pytest.mark.asyncio
async def test_list_experiments_handles_corrupt_files(temp_storage_dir):
    storage = JSONStorage(base_dir=temp_storage_dir)
    manager = ExperimentManager(storage=storage)
    
    exp = manager.create_experiment(
        dataset_name="Valid", provider="mock", model="mock", evaluation_configuration={},
        evaluation_results=[], evaluation_metrics={}, average_latency=1.0,
        total_number_of_test_cases=0, passed_test_cases=0, failed_test_cases=0,
        experiment_id="exp_valid"
    )
    await manager.save_experiment(exp)
    
    corrupt_folder = os.path.join(temp_storage_dir, "exp_corrupt")
    os.makedirs(corrupt_folder, exist_ok=True)
    with open(os.path.join(corrupt_folder, "experiment.json"), "w") as f:
        f.write("{invalid json syntax: ...}")
        
    experiments = await manager.list_experiments()
    assert len(experiments) == 1
    assert experiments[0].experiment_id == "exp_valid"


@pytest.mark.asyncio
async def test_delete_experiment(temp_storage_dir):
    storage = JSONStorage(base_dir=temp_storage_dir)
    manager = ExperimentManager(storage=storage)
    
    exp = manager.create_experiment(
        dataset_name="To Delete", provider="mock", model="mock", evaluation_configuration={},
        evaluation_results=[], evaluation_metrics={}, average_latency=1.0,
        total_number_of_test_cases=0, passed_test_cases=0, failed_test_cases=0,
        experiment_id="exp_delete"
    )
    await manager.save_experiment(exp)
    
    exp_folder = os.path.join(temp_storage_dir, "exp_delete")
    assert os.path.exists(exp_folder)
    
    await manager.delete_experiment("exp_delete")
    assert not os.path.exists(exp_folder)
    
    with pytest.raises(ExperimentNotFoundError):
        await manager.load_experiment("exp_delete")


@pytest.mark.asyncio
async def test_load_non_existent_raises_error(temp_storage_dir):
    storage = JSONStorage(base_dir=temp_storage_dir)
    manager = ExperimentManager(storage=storage)
    
    with pytest.raises(ExperimentNotFoundError):
        await manager.load_experiment("exp_non_existent")
