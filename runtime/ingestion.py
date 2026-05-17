import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .lib.io import load_json, save_json

logger = logging.getLogger(__name__)


class DataIngester:
    def __init__(self, config: dict):
        self.cfg = config
        self.data_cfg = config["data"]
        self.ing_cfg = config["ingestion"]
        self.source_file = self.data_cfg["source_file"]
        self.target_col = self.data_cfg["target_column"]
        self.time_col = self.data_cfg["time_column"]
        self.id_cols = self.data_cfg["id_columns"]
        self.batch_size = self.data_cfg["batch_size"]
        self.raw_dir = Path(self.ing_cfg["raw_store_dir"])
        self.meta_dir = Path(self.ing_cfg["meta_dir"])
        self.state_file = Path(self.ing_cfg["state_file"])
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_source_file(self):
        source = Path(self.source_file)
        if source.exists():
            return
        kaggle_cfg = self.data_cfg.get("kaggle", {})
        if not kaggle_cfg.get("enabled", False):
            raise FileNotFoundError(
                f"Source file not found: {source}. "
                "Set data.kaggle.enabled=true in config to download."
            )
        import kaggle
        dataset = kaggle_cfg["dataset"]
        filename = kaggle_cfg["filename"]
        source.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading dataset '%s' from Kaggle...", dataset)
        kaggle.api.authenticate()
        kaggle.api.dataset_download_file(
            dataset, file_name=filename,
            path=str(source.parent), force=False, quiet=False,
        )
        zip_path = source.parent / (filename + ".zip")
        if zip_path.exists():
            import zipfile
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(source.parent)
            zip_path.unlink()
        if not source.exists():
            raise FileNotFoundError(
                f"Download succeeded but file not found at {source}"
            )
        logger.info("Dataset saved to %s", source)

    def _load_state(self) -> dict:
        return load_json(
            self.state_file,
            default={"current_batch": 0, "total_batches": 0,
                     "initialized": False},
        )

    def _save_state(self, state: dict):
        save_json(self.state_file, state)

    def _inject_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        mi_cfg = self.data_cfg.get("missing_injection", {})
        if not mi_cfg.get("enabled", False):
            return df
        rate = mi_cfg.get("rate", 0.05)
        cols = mi_cfg.get("columns", [])
        seed = mi_cfg.get("seed", 42)
        rng = np.random.default_rng(seed)
        df = df.copy()
        for col in cols:
            if col not in df.columns:
                continue
            n = int(len(df) * rate)
            idx = rng.choice(df.index, size=n, replace=False)
            df.loc[idx, col] = np.nan
        logger.info(
            "Injected ~%.0f%% missing into: %s", rate * 100, cols
        )
        return df

    def initialize(self) -> dict:
        state = self._load_state()
        if state.get("initialized"):
            logger.info(
                "Already initialized: %d batches.", state["total_batches"]
            )
            return state
        self._ensure_source_file()
        logger.info("Initializing raw store from %s", self.source_file)
        df = pd.read_csv(self.source_file)
        df[self.time_col] = pd.to_datetime(df[self.time_col])
        df = df.sort_values(self.time_col).reset_index(drop=True)
        df = self._inject_missing(df)
        total_batches = max(1, len(df) // self.batch_size)
        batches = [
            df.iloc[i * self.batch_size:(i + 1) * self.batch_size]
            for i in range(total_batches)
        ]
        for i, batch in enumerate(batches):
            batch.to_csv(self.raw_dir / f"batch_{i:03d}.csv", index=False)
            save_json(
                self.meta_dir / f"meta_batch_{i:03d}.json",
                self._compute_metaparameters(batch, i),
            )
        state = {
            "current_batch": 0,
            "total_batches": total_batches,
            "batch_size": self.batch_size,
            "initialized": True,
            "initialized_at": datetime.now().isoformat(),
        }
        self._save_state(state)
        logger.info("Initialized %d batches from %d rows.", total_batches, len(df))
        return state

    def ingest_next_batch(self) -> tuple:
        state = self._load_state()
        if not state.get("initialized"):
            state = self.initialize()
        idx = state["current_batch"]
        if idx >= state["total_batches"]:
            logger.warning("All %d batches consumed.", state["total_batches"])
            return None, state
        df = pd.read_csv(self.raw_dir / f"batch_{idx:03d}.csv")
        meta = load_json(self.meta_dir / f"meta_batch_{idx:03d}.json")
        logger.info(
            "Ingested batch %d/%d (%d rows)",
            idx + 1, state["total_batches"], len(df),
        )
        state["current_batch"] = idx + 1
        state["last_ingested_at"] = datetime.now().isoformat()
        self._save_state(state)
        return df, meta

    def load_accumulated_data(self) -> pd.DataFrame:
        state = self._load_state()
        n = state.get("current_batch", 0)
        if n == 0:
            raise RuntimeError("No batches ingested yet.")
        frames = [
            pd.read_csv(self.raw_dir / f"batch_{i:03d}.csv")
            for i in range(n)
            if (self.raw_dir / f"batch_{i:03d}.csv").exists()
        ]
        return pd.concat(frames, ignore_index=True)

    def _compute_metaparameters(self, df: pd.DataFrame, batch_idx: int) -> dict:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include="object").columns.tolist()
        return {
            "batch_index": batch_idx,
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "n_numeric": len(num_cols),
            "n_categorical": len(cat_cols),
            "missing_total_pct": float(df.isnull().mean().mean()),
            "missing_per_column": df.isnull().mean().to_dict(),
            "date_min": str(df[self.time_col].min())
            if self.time_col in df.columns else None,
            "date_max": str(df[self.time_col].max())
            if self.time_col in df.columns else None,
            "target_distribution": df[self.target_col]
            .value_counts(normalize=True).to_dict()
            if self.target_col in df.columns else {},
            "computed_at": datetime.now().isoformat(),
        }

    def get_state(self) -> dict:
        return self._load_state()
