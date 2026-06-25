def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_list_providers(client):
    response = client.get("/api/v1/providers/")
    assert response.status_code == 200
    data = response.json()
    assert "providers" in data
    providers = {p["name"] for p in data["providers"]}
    assert "openai" in providers
    assert "anthropic" in providers

def test_get_dataset(client):
    response = client.get("/api/v1/datasets/test-dataset")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "test-dataset"
    assert len(data["entries"]) == 2

def test_create_dataset(client):
    new_dataset = {
        "id": "custom-dataset",
        "name": "Custom User Dataset",
        "entries": [
            {"prompt": "Translate 'hello' to Spanish", "expected_output": "hola"}
        ]
    }
    response = client.post("/api/v1/datasets/", json=new_dataset)
    assert response.status_code == 201
    assert response.json()["status"] == "success"

    # Verify retrieval
    get_response = client.get("/api/v1/datasets/custom-dataset")
    assert get_response.status_code == 200
    assert get_response.json()["name"] == "Custom User Dataset"

def test_run_and_export_experiment(client):
    config = {
        "name": "Test Experiment Run",
        "description": "Scaffolding verification run",
        "provider_name": "openai",
        "model_name": "gpt-4o",
        "dataset_id": "test-dataset",
        "generation_config": {"temperature": 0.0}
    }
    # Run experiment
    response = client.post("/api/v1/experiments/", json=config)
    assert response.status_code == 202
    run_data = response.json()
    assert run_data["status"] == "completed"
    assert run_data["config"]["provider_name"] == "openai"
    assert len(run_data["results"]) == 2
    assert "avg_latency_ms" in run_data["summary_metrics"]
    experiment_id = run_data["id"]

    # Retrieve experiment
    get_response = client.get(f"/api/v1/experiments/{experiment_id}")
    assert get_response.status_code == 200
    assert get_response.json()["status"] == "completed"

    # Export experiment JSON
    export_json = client.get(f"/api/v1/experiments/{experiment_id}/export?format=json")
    assert export_json.status_code == 200
    assert export_json.json()["id"] == experiment_id

    # Export experiment CSV
    export_csv = client.get(f"/api/v1/experiments/{experiment_id}/export?format=csv")
    assert export_csv.status_code == 200
    csv_text = export_csv.text
    assert "prompt_id" in csv_text
    assert "latency_ms" in csv_text
