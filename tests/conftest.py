import pytest
from fastapi.testclient import TestClient
import asyncio
import sys
import os

# Ensure the backend module is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.api.main import app
from backend.datasets.loader import get_dataset_loader

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

@pytest.fixture(scope="session", autouse=True)
def setup_test_dataset():
    # Pre-register a test dataset for endpoint testing synchronously using asyncio.run
    loader = get_dataset_loader()
    asyncio.run(loader.register(
        dataset_id="test-dataset",
        name="Test Evaluation Dataset",
        entries=[
            {"id": "q1", "prompt": "What is 2+2?", "expected_output": "4"},
            {"id": "q2", "prompt": "Capital of France?", "expected_output": "Paris"}
        ]
    ))

