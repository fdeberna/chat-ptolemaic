from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def _to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_serializable(v) for v in value]
    if isinstance(value, tuple):
        return [_to_serializable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def sanitize_experiment_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "run"


def make_run_dir(*, runs_root: Path, experiment_name: Optional[str]) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp = sanitize_experiment_name(experiment_name) if experiment_name else "run"
    base_name = f"{timestamp}_{exp}"
    run_dir = runs_root / base_name
    suffix = 1
    while run_dir.exists():
        run_dir = runs_root / f"{base_name}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


class RunLogger:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.config_path = run_dir / "config.json"
        self.metrics_csv_path = run_dir / "metrics.csv"
        self.jsonl_path = run_dir / "training_log.jsonl"
        self.samples_path = run_dir / "generation_samples.txt"
        self.checkpoint_path = run_dir / "model_checkpoint.pt"
        self._fieldnames = [
            "step",
            "epoch",
            "train_loss",
            "val_loss",
            "learning_rate",
            "tokens_processed",
            "time_elapsed",
        ]

    def save_config(self, config: Dict[str, Any]) -> None:
        payload = _to_serializable(config)
        self.config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def log_metrics(self, row: Dict[str, Any]) -> None:
        serializable_row = _to_serializable(row)
        exists = self.metrics_csv_path.exists()
        with self.metrics_csv_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._fieldnames)
            if not exists:
                writer.writeheader()
            writer.writerow(serializable_row)
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(serializable_row) + "\n")

    def append_sample(self, *, step: int, prompt: str, generated_text: str) -> None:
        with self.samples_path.open("a", encoding="utf-8") as handle:
            handle.write(f"step: {step}\n")
            handle.write(f"prompt: {prompt}\n")
            handle.write("generated:\n")
            handle.write(generated_text)
            handle.write("\n")
            handle.write("=" * 80)
            handle.write("\n")
