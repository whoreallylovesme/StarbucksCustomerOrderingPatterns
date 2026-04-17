import logging
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from .lib.io import load_pickle, save_pickle

logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, config: dict):
        self.cfg = config["training"]["catboost"]
        self.model_path = Path(config["training"]["model_path"])
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self._model = None

    def _make_model(self) -> CatBoostRegressor:
        return CatBoostRegressor(
            iterations=self.cfg.get("iterations_per_batch", 100),
            depth=self.cfg.get("depth", 6),
            learning_rate=self.cfg.get("learning_rate", 0.1),
            loss_function=self.cfg.get("loss_function", "RMSE"),
            eval_metric=self.cfg.get("eval_metric", "RMSE"),
            thread_count=self.cfg.get("thread_count", 4),
            bootstrap_type=self.cfg.get("bootstrap_type", "Bernoulli"),
            subsample=self.cfg.get("subsample", 0.8),
            random_seed=self.cfg.get("random_seed", 42),
            verbose=self.cfg.get("verbose", 0),
            train_dir=self.cfg.get("train_dir", "artifacts/logs/catboost_info"),
        )

    def _load_prev_model(self):
        if self.model_path.exists():
            m = load_pickle(self.model_path)
            logger.info("Loaded previous model (%d trees)", m.tree_count_)
            return m
        return None

    def train(self, X: pd.DataFrame, y: np.ndarray,
              cat_features: list) -> CatBoostRegressor:
        prev = self._load_prev_model()
        pool = Pool(X, y, cat_features=cat_features)
        model = self._make_model()
        if prev is not None:
            model.fit(pool, init_model=prev)
            logger.info("Incremental fit: %d total trees", model.tree_count_)
        else:
            model.fit(pool)
            logger.info("Initial fit: %d trees", model.tree_count_)
        save_pickle(self.model_path, model)
        self._model = model
        return model

    def train_all(self, X_new: pd.DataFrame, y_new: np.ndarray,
                  cat_features: list) -> dict:
        return {"catboost": self.train(X_new, y_new, cat_features)}

    def get_model(self):
        return self._model
