import json
import csv
from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseExporter(ABC):
    """Abstract interface for exporting experiment results."""

    @abstractmethod
    def export(self, run_data: Dict[str, Any], filepath: str) -> None:
        """Export experiment run data to a destination file."""
        pass


class JSONExporter(BaseExporter):
    """Exports experiment runs as JSON files."""

    def export(self, run_data: Dict[str, Any], filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(run_data, f, indent=2, default=str)


class CSVExporter(BaseExporter):
    """Exports experiment run entries as a flat CSV file."""

    def export(self, run_data: Dict[str, Any], filepath: str) -> None:
        results = run_data.get("results", [])
        if not results:
            return

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Define header: ID, Prompt, Expected, Generated, Latency, plus dynamic metric headers
            metric_keys = sorted(list(results[0].get("metrics", {}).keys()))
            header = ["prompt_id", "prompt", "expected_output", "generated_output", "latency_ms"] + [f"metric_{k}" for k in metric_keys]
            writer.writerow(header)

            for entry in results:
                row = [
                    entry.get("prompt_id"),
                    entry.get("prompt"),
                    entry.get("expected_output"),
                    entry.get("generated_output"),
                    entry.get("latency_ms")
                ]
                metrics = entry.get("metrics", {})
                for k in metric_keys:
                    row.append(metrics.get(k, 0.0))
                writer.writerow(row)
