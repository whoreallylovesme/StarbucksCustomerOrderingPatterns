import logging
import numpy as np
import pandas as pd
logger = logging.getLogger(__name__)


class DataPreparer:
    def __init__(self, config: dict):
        self.cfg = config["preparation"]
        self.data_cfg = config["data"]
        self.target_col = self.data_cfg["target_column"]
        self.time_col = self.data_cfg["time_column"]
        self.id_cols = self.data_cfg["id_columns"]
        self.num_strategy = self.cfg.get("numeric_impute_strategy", "median")
        self.num_constant = self.cfg.get("numeric_impute_constant", 0.0)
        self.cat_fill = self.cfg.get("categorical_fill_value", "Unknown")
        self._num_fill: dict = {}
        self._cat_features: list = []
        self._feature_cols: list = []
        self._fitted = False

    def _compute_fill_value(self, series: pd.Series) -> float:
        if self.num_strategy == "mean":
            return float(series.mean())
        if self.num_strategy == "mode":
            mode = series.mode()
            return float(mode.iloc[0]) if not mode.empty else 0.0
        if self.num_strategy == "constant":
            return float(self.num_constant)
        return float(series.median())

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "order_time" in df.columns:
            df["hour"] = pd.to_datetime(
                df["order_time"],
                format="%H:%M",
                errors="coerce").dt.hour
        if self.time_col in df.columns:
            dt = pd.to_datetime(df[self.time_col], errors="coerce")
            df["month"] = dt.dt.month
            df["year"] = dt.dt.year
        return df

    def _drop_unused(self, df: pd.DataFrame) -> pd.DataFrame:
        drop = self.id_cols + [self.time_col, "order_time"]
        return df.drop(
            columns=[
                c for c in drop if c in df.columns],
            errors="ignore")

    def fit_transform(
            self,
            df: pd.DataFrame,
            features_to_drop: list = None) -> tuple:
        df = self._engineer_features(df)
        df = self._drop_unused(df)
        y = df[self.target_col].astype(float).values
        X = df.drop(columns=[self.target_col]).copy()
        if features_to_drop:
            X = X.drop(
                columns=[
                    c for c in features_to_drop if c in X.columns],
                errors="ignore")
            logger.info("Dropped insignificant features: %s", features_to_drop)
        for c in X.select_dtypes(include="bool").columns:
            X[c] = X[c].astype(int)
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        for c in num_cols:
            self._num_fill[c] = self._compute_fill_value(X[c].dropna())
            X[c] = X[c].fillna(self._num_fill[c])
        logger.info(
            "Numeric imputation strategy: %s", self.num_strategy
        )
        cat_cols = X.select_dtypes(include="object").columns.tolist()
        for c in cat_cols:
            X[c] = X[c].fillna(self.cat_fill).astype(str)
        self._cat_features = cat_cols
        self._feature_cols = X.columns.tolist()
        self._fitted = True
        logger.info("Fitted preparer: X=%s, cat_features=%s", X.shape, cat_cols)
        return X, y

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Preparer not fitted yet.")
        df = self._engineer_features(df)
        df = self._drop_unused(df)
        if self.target_col in df.columns:
            df = df.drop(columns=[self.target_col])
        df = df.copy()
        for c in df.select_dtypes(include="bool").columns:
            df[c] = df[c].astype(int)
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for c in num_cols:
            df[c] = df[c].fillna(self._num_fill.get(c, 0.0))
        cat_cols = df.select_dtypes(include="object").columns.tolist()
        for c in cat_cols:
            df[c] = df[c].fillna(self.cat_fill).astype(str)
        for c in self._feature_cols:
            if c not in df.columns:
                df[c] = 0
        return df[self._feature_cols]

    def transform_target(self, y: np.ndarray) -> np.ndarray:
        return y.astype(float)

    def inverse_transform_target(self, y: np.ndarray) -> np.ndarray:
        return y.astype(float)

    def get_cat_features(self) -> list:
        return self._cat_features

    def get_feature_names(self) -> list:
        return self._feature_cols
