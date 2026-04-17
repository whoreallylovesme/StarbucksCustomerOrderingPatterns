import logging
import time
import tracemalloc
from datetime import datetime
from pathlib import Path

import pandas as pd

from .lib.io import append_json_list, load_json, load_pickle, save_json, save_pickle

logger = logging.getLogger(__name__)


class ModelServer:
    def __init__(self, config: dict):
        self.cfg = config["serving"]
        self.prod_dir = Path(self.cfg["production_dir"])
        self.log_dir = Path(self.cfg["log_dir"])
        self.monitoring_file = Path(self.cfg["monitoring_file"])
        self.predictions_dir = Path(
            self.cfg.get("predictions_dir", "artifacts/data/predictions")
        )
        self.prod_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_dir.mkdir(parents=True, exist_ok=True)

    def serialize_model(self, model, preparer,
                        model_name: str, metrics: dict) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        bundle = {
            "model": model, "preparer": preparer,
            "model_name": model_name, "metrics": metrics,
            "serialized_at": datetime.now().isoformat(),
        }
        path = self.prod_dir / f"production_{model_name}_{ts}.pkl"
        save_pickle(path, bundle)
        save_pickle(self.prod_dir / "latest.pkl", bundle)
        save_json(self.prod_dir / "latest_meta.json", {
            "model_name": model_name, "path": str(path),
            "metrics": metrics, "serialized_at": bundle["serialized_at"],
        })
        logger.info("Serialized production model to %s", path)
        return path

    def load_production_model(self) -> tuple:
        latest = self.prod_dir / "latest.pkl"
        if not latest.exists():
            raise FileNotFoundError(
                "No production model found. Run 'update' first."
            )
        bundle = load_pickle(latest)
        logger.info("Loaded production model: %s", bundle.get("model_name"))
        return bundle["model"], bundle["preparer"], bundle

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        model, preparer, meta = self.load_production_model()
        tracemalloc.start()
        t0 = time.perf_counter()
        X = preparer.transform(df)
        y_pred = preparer.inverse_transform_target(model.predict(X))
        elapsed_ms = (time.perf_counter() - t0) * 1000
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        result = df.copy()
        result["predict"] = y_pred
        perf = {
            "timestamp": datetime.now().isoformat(),
            "model_name": meta.get("model_name"),
            "n_samples": len(df),
            "inference_time_ms": round(elapsed_ms, 3),
            "peak_memory_kb": round(peak_mem / 1024, 2),
            "ms_per_sample": round(elapsed_ms / max(len(df), 1), 4),
        }
        append_json_list(self.monitoring_file, perf)
        logger.info(
            "Inference: %d samples in %.1f ms (%.3f ms/sample)",
            perf["n_samples"], perf["inference_time_ms"], perf["ms_per_sample"],
        )
        return result

    def save_predictions(self, df: pd.DataFrame) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.predictions_dir / f"predictions_{ts}.csv"
        df.to_csv(path, index=False)
        logger.info("Saved predictions to %s", path)
        return path

    def get_monitoring_history(self) -> list:
        return load_json(self.monitoring_file, default=[])

    def has_production_model(self) -> bool:
        return (self.prod_dir / "latest.pkl").exists()
