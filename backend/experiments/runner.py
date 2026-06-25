import uuid
import time
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from backend.experiments.models import ExperimentConfig, ExperimentRun, ExperimentResultEntry
from backend.providers import ProviderFactory
from backend.core.database import get_storage
from backend.datasets.loader import get_dataset_loader
from backend.analysis.metrics import get_metric_evaluator

class ExperimentRunner:
    """Runner class responsible for executing LLM evaluation experiments."""

    def __init__(self):
        self.storage = get_storage()
        self.dataset_loader = get_dataset_loader()
        self.metric_evaluator = get_metric_evaluator()

    async def run(self, config: ExperimentConfig) -> ExperimentRun:
        run_id = str(uuid.uuid4())
        experiment_run = ExperimentRun(
            id=run_id,
            config=config,
            status="running",
            created_at=datetime.now(timezone.utc)
        )
        
        # Save initially as running
        await self.storage.save_experiment(run_id, experiment_run.model_dump())

        try:
            # 1. Fetch dataset entries
            dataset = await self.dataset_loader.load(config.dataset_id)
            if not dataset:
                raise ValueError(f"Dataset {config.dataset_id} not found.")

            # 2. Initialize provider
            provider = ProviderFactory.create(
                provider=config.provider_name,
                model=config.model_name
            )

            results: List[ExperimentResultEntry] = []
            
            # 3. Iterate through dataset inputs
            for entry in dataset.get("entries", []):
                prompt_id = entry.get("id", str(uuid.uuid4()))
                prompt = entry.get("prompt")
                expected = entry.get("expected_output")

                start_time = time.time()
                try:
                    response = await provider.generate(
                        prompt=prompt,
                        system_instruction=config.system_instruction,
                        generation_config=config.generation_config
                    )
                    latency = (time.time() - start_time) * 1000
                    generated = response.get("text", "")
                    usage = response.get("usage", {})
                except Exception as e:
                    generated = f"Error during generation: {str(e)}"
                    latency = (time.time() - start_time) * 1000
                    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

                # 4. Compute metrics using metric evaluator
                metrics = await self.metric_evaluator.evaluate(generated, expected)

                results.append(
                    ExperimentResultEntry(
                        prompt_id=prompt_id,
                        prompt=prompt,
                        expected_output=expected,
                        generated_output=generated,
                        metrics=metrics,
                        latency_ms=latency,
                        usage=usage,
                        timestamp=datetime.now(timezone.utc)
                    )
                )

            # 5. Compute summary metrics across all runs
            summary = self._compute_summary(results)

            # 6. Update experiment state to completed
            experiment_run.results = results
            experiment_run.summary_metrics = summary
            experiment_run.status = "completed"
            
        except Exception as e:
            experiment_run.status = "failed"
            experiment_run.summary_metrics = {"error_rate": 1.0}
        
        await self.storage.save_experiment(run_id, experiment_run.model_dump())
        return experiment_run

    def _compute_summary(self, results: List[ExperimentResultEntry]) -> Dict[str, float]:
        if not results:
            return {}
        
        sums: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        total_latency = 0.0

        for r in results:
            total_latency += r.latency_ms
            for k, v in r.metrics.items():
                sums[k] = sums.get(k, 0.0) + v
                counts[k] = counts.get(k, 0) + 1

        summary = {k: (sums[k] / counts[k]) for k in sums}
        summary["avg_latency_ms"] = total_latency / len(results)
        return summary
