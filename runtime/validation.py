import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
)

from .lib.io import load_json, save_json, load_pickle, save_pickle

logger = logging.getLogger(__name__)


class ModelValidator:
    def __init__(self, config: dict):
        self.cfg = config["validation"]
        self.model_dir = Path(self.cfg["model_store_dir"])
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.model_dir / "registry.json"
        self.promotion_metric = self.cfg.get("promotion_metric", "rmse")
        self.min_improvement = self.cfg.get("promotion_min_improvement", 0.01)
        self.drift_threshold = self.cfg.get("model_drift_threshold", 0.1)

    def evaluate(self, model, X, y, label: str = "") -> dict:
        y_pred = model.predict(X)
        rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
        mae = float(mean_absolute_error(y, y_pred))
        r2 = float(r2_score(y, y_pred))
        metrics = {"label": label, "rmse": rmse, "mae": mae,
                   "r2": r2, "n_samples": int(len(y))}
        logger.info("[%s] RMSE=%.4f  MAE=%.4f  R²=%.4f", label, rmse, mae, r2)
        return metrics

    def _load_registry(self) -> list:
        return load_json(self.registry_file, default=[])

    def _save_registry(self, registry: list):
        save_json(self.registry_file, registry)

    def save_model(self, model, metrics: dict,
                   model_name: str, batch_idx: int) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.model_dir / f"{model_name}_batch{batch_idx:03d}_{ts}.pkl"
        save_pickle(path, model)
        entry = {
            "path": str(path),
            "model_name": model_name,
            "batch_index": batch_idx,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
        }
        registry = self._load_registry()
        registry.append(entry)
        self._save_registry(registry)
        logger.info("Saved model to %s (RMSE=%.4f)", path, metrics.get("rmse", 0))
        return path

    def get_best_model_entry(self) -> dict | None:
        registry = self._load_registry()
        if not registry:
            return None
        return min(registry, key=lambda e: e["metrics"].get(
            self.promotion_metric, float("inf")))

    def load_model(self, path: str):
        return load_pickle(path)

    def should_promote(self, new_metrics: dict, model_name: str) -> bool:
        best = self.get_best_model_entry()
        if best is None:
            return True
        best_score = best["metrics"].get(self.promotion_metric, float("inf"))
        new_score = new_metrics.get(self.promotion_metric, float("inf"))
        improvement = best_score - new_score
        if improvement >= self.min_improvement:
            logger.info(
                "Promoting %s: RMSE %.4f → %.4f (↓%.4f)",
                model_name, best_score, new_score, improvement,
            )
            return True
        logger.info(
            "Not promoting %s: improvement %.4f < threshold %.4f",
            model_name, improvement, self.min_improvement,
        )
        return False

    def detect_model_drift(self) -> dict:
        registry = self._load_registry()
        if len(registry) < 2:
            return {"drift_detected": False, "message": "Not enough history."}
        scores = [e["metrics"].get(self.promotion_metric, 0) for e in registry]
        recent = scores[-1]
        baseline = min(scores[:-1])
        increase = recent - baseline
        if increase > self.drift_threshold:
            logger.warning("Model drift: RMSE increased by %.4f.", increase)
        return {
            "drift_detected": increase > self.drift_threshold,
            "baseline_best": float(baseline),
            "recent_score": float(recent),
            "increase": float(increase),
        }

    def get_metrics_history(self) -> list:
        return [
            {"model_name": e["model_name"], "batch_index": e["batch_index"],
             "timestamp": e["timestamp"], **e["metrics"]}
            for e in self._load_registry()
        ]
