import copy
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def load_config() -> dict:
    cfg_dir = ROOT / "configs"
    parts = ["data.yaml", "analysis.yaml", "training.yaml", "serving.yaml"]
    merged = {}
    for part in parts:
        p = cfg_dir / part
        if p.exists():
            merged.update(yaml.safe_load(p.read_text()) or {})
    return merged


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def isolated_cfg(tmp_path, cfg):
    c = copy.deepcopy(cfg)
    c["training"]["model_path"] = str(tmp_path / "catboost_incremental.pkl")
    c["training"]["catboost"]["train_dir"] = str(tmp_path / "catboost_info")
    c["validation"]["model_store_dir"] = str(tmp_path / "models")
    c["serving"]["production_dir"] = str(tmp_path / "production")
    c["serving"]["log_dir"] = str(tmp_path / "logs")
    c["serving"]["monitoring_file"] = str(tmp_path / "logs" / "monitoring.json")
    c["ingestion"]["raw_store_dir"] = str(tmp_path / "raw")
    c["ingestion"]["meta_dir"] = str(tmp_path / "meta")
    c["ingestion"]["state_file"] = str(tmp_path / "meta" / "state.json")
    c["reports"]["output_dir"] = str(tmp_path / "reports")
    return c


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame({
        "customer_id": [f"C{i}" for i in range(n)],
        "order_id": [f"O{i}" for i in range(n)],
        "order_date": pd.date_range("2023-01-01", periods=n, freq="D").astype(str),
        "order_time": [f"{rng.integers(8, 22):02d}:{rng.integers(0, 59):02d}" for _ in range(n)],
        "drink_category": rng.choice(["Coffee", "Tea", "Juice"], n),
        "order_channel": rng.choice(["App", "In-store", "Drive-thru"], n),
        "cart_size": rng.integers(1, 5, n).astype(float),
        "num_customizations": rng.integers(0, 4, n).astype(float),
        "fulfillment_time_min": rng.uniform(2, 15, n),
        "total_spend": rng.uniform(5, 30, n),
    })
