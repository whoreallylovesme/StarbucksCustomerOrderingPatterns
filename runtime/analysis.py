import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .lib.io import load_json, save_json

logger = logging.getLogger(__name__)


class DataAnalyzer:
    def __init__(self, config: dict):
        self.cfg = config["analysis"]
        self.data_cfg = config["data"]
        self.quality_dir = Path(self.cfg["quality_dir"])
        self.quality_dir.mkdir(parents=True, exist_ok=True)
        self.missing_threshold = self.cfg["missing_threshold"]
        self.dup_threshold = self.cfg["duplicate_threshold"]
        self.z_threshold = self.cfg["outlier_z_threshold"]
        self.ks_pvalue = self.cfg["drift_ks_pvalue"]
        self.target_col = self.data_cfg["target_column"]
        self.id_cols = self.data_cfg["id_columns"]
        self.time_col = self.data_cfg["time_column"]

    def compute_data_quality(self, df: pd.DataFrame, batch_idx: int) -> dict:
        n = len(df)
        missing_pct = df.isnull().mean().to_dict()
        dup_pct = float(df.duplicated().sum() / n) if n > 0 else 0.0
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        outlier_pct = {}
        for col in num_cols:
            if col == self.target_col:
                continue
            clean = df[col].dropna()
            if len(clean) < 2:
                continue
            z = np.abs(stats.zscore(clean))
            outlier_pct[col] = float((z > self.z_threshold).mean())
        cols_above = [c for c, v in missing_pct.items()
                      if v > self.missing_threshold]
        return {
            "batch_index": batch_idx,
            "n_rows": n,
            "missing_pct_per_col": missing_pct,
            "avg_missing_pct": float(df.isnull().mean().mean()),
            "duplicate_pct": dup_pct,
            "outlier_pct_per_col": outlier_pct,
            "cols_exceeding_missing_threshold": cols_above,
            "passes_missing_check": len(cols_above) == 0,
            "passes_duplicate_check": dup_pct <= self.dup_threshold,
            "computed_at": datetime.now().isoformat(),
        }

    def check_quality_thresholds(self, dq: dict) -> tuple:
        issues = []
        if not dq["passes_missing_check"]:
            issues.append(
                "Columns exceed missing threshold: "
                f"{dq['cols_exceeding_missing_threshold']}"
            )
        if not dq["passes_duplicate_check"]:
            issues.append(f"Duplicate rows: {dq['duplicate_pct']:.1%}")
        passed = len(issues) == 0
        if passed:
            logger.info("Data quality checks passed.")
        else:
            logger.warning("Data quality issues: %s", issues)
        return passed, issues

    def save_dq_report(self, metrics: dict, batch_idx: int) -> Path:
        path = self.quality_dir / f"dq_batch_{batch_idx:03d}.json"
        save_json(path, metrics)
        logger.info("Saved DQ report to %s", path)
        return path

    def clean_data(self, df: pd.DataFrame, dq: dict) -> pd.DataFrame:
        df = df.copy()
        drop_cols = dq.get("cols_exceeding_missing_threshold", [])
        if drop_cols:
            logger.info("Dropping high-missing columns: %s", drop_cols)
            df = df.drop(
                columns=[c for c in drop_cols if c in df.columns],
                errors="ignore",
            )
        before = len(df)
        df = df.drop_duplicates()
        if len(df) < before:
            logger.info("Removed %d duplicate rows.", before - len(df))
        return df

    def auto_eda(self, df: pd.DataFrame, batch_idx: int) -> dict:
        drop = self.id_cols + [self.time_col]
        num_cols = [c for c in df.select_dtypes(
            include=[np.number]).columns if c not in drop]
        cat_cols = [c for c in df.select_dtypes(
            include="object").columns if c not in drop]
        num_stats = {
            col: {
                "mean": float(s.mean()), "median": float(s.median()),
                "std": float(s.std()), "min": float(s.min()),
                "max": float(s.max()), "q25": float(s.quantile(0.25)),
                "q75": float(s.quantile(0.75)), "skew": float(s.skew()),
            }
            for col in num_cols
            if len(s := df[col].dropna()) > 0
        }
        cat_stats = {
            col: {
                "n_unique": int(df[col].nunique()),
                "top_5": df[col].value_counts(normalize=True).head(5).to_dict(),
                "missing_pct": float(df[col].isnull().mean()),
            }
            for col in cat_cols
        }
        corr_cols = [c for c in num_cols if c != self.target_col]
        corr = {
            col: float(df[[col, self.target_col]].dropna()[col].corr(
                df[[col, self.target_col]].dropna()[self.target_col]
            ))
            for col in corr_cols
            if self.target_col in df.columns
            and len(df[[col, self.target_col]].dropna()) > 1
        }
        eda = {
            "batch_index": batch_idx,
            "numeric_stats": num_stats,
            "categorical_stats": cat_stats,
            "target_distribution": df[self.target_col]
            .value_counts(normalize=True).to_dict()
            if self.target_col in df.columns else {},
            "feature_target_correlations": corr,
            "computed_at": datetime.now().isoformat(),
        }
        path = self.quality_dir / f"eda_batch_{batch_idx:03d}.json"
        save_json(path, eda)
        logger.info("Saved EDA report to %s", path)
        return eda

    def statsmodels_eda(self, df: pd.DataFrame, batch_idx: int) -> dict:
        import statsmodels.api as sm
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        drop = set(self.id_cols + [self.time_col, self.target_col])
        num_cols = [c for c in df.select_dtypes(
            include=[np.number]).columns if c not in drop]
        if not num_cols or self.target_col not in df.columns:
            return {"batch_index": batch_idx,
                    "insignificant_features": [], "high_vif_features": []}
        sub = df[num_cols + [self.target_col]].dropna()
        X = sub[num_cols]
        y = sub[self.target_col].astype(float)
        ols = sm.OLS(y, sm.add_constant(X)).fit()
        pvalues = {c: float(ols.pvalues[c])
                   for c in num_cols if c in ols.pvalues}
        vif = {}
        if len(num_cols) > 1:
            for i, col in enumerate(num_cols):
                try:
                    vif[col] = float(variance_inflation_factor(X.values, i))
                except Exception:
                    vif[col] = float("nan")
        jb = {
            col: {"stat": round(float(stats.jarque_bera(
                sub[col].dropna())[0]), 4),
                "normal": stats.jarque_bera(sub[col].dropna())[1] > 0.05}
            for col in num_cols
        }
        result = {
            "batch_index": batch_idx,
            "ols_r2": round(float(ols.rsquared), 4),
            "ols_pvalues": {c: round(p, 4) for c, p in pvalues.items()},
            "significant_features": [c for c, p in pvalues.items() if p <= 0.05],
            "insignificant_features": [c for c, p in pvalues.items() if p > 0.05],
            "vif": {c: round(v, 2) for c, v in vif.items()},
            "high_vif_features": [c for c, v in vif.items() if v > 10],
            "jarque_bera": jb,
            "computed_at": datetime.now().isoformat(),
        }
        path = self.quality_dir / f"statsmodels_eda_batch_{batch_idx:03d}.json"
        save_json(path, result)
        logger.info(
            "Statsmodels EDA: OLS R²=%.4f | insignificant=%s | high_vif=%s",
            ols.rsquared, result["insignificant_features"],
            result["high_vif_features"],
        )
        return result

    def get_features_to_drop(self, stats_eda: dict) -> list:
        return stats_eda.get("insignificant_features", [])

    def detect_data_drift(
            self, df_ref: pd.DataFrame, df_new: pd.DataFrame) -> dict:
        drop = [self.target_col] + self.id_cols
        num_cols = [
            c for c in df_ref.select_dtypes(include=[np.number]).columns
            if c not in drop and c in df_new.columns
        ]
        results = {}
        drifted = []
        for col in num_cols:
            a = df_ref[col].dropna().values
            b = df_new[col].dropna().values
            if len(a) < 2 or len(b) < 2:
                continue
            stat, pvalue = stats.ks_2samp(a, b)
            drifted_flag = pvalue < self.ks_pvalue
            results[col] = {"ks_stat": float(stat),
                            "p_value": float(pvalue), "drifted": drifted_flag}
            if drifted_flag:
                drifted.append(col)
        if drifted:
            logger.warning("Data drift detected in: %s", drifted)
        else:
            logger.info("No data drift detected.")
        return {
            "drift_detected": len(drifted) > 0,
            "drifted_columns": drifted,
            "column_results": results,
            "computed_at": datetime.now().isoformat(),
        }

    def load_all_dq_reports(self) -> list:
        return [
            load_json(p)
            for p in sorted(self.quality_dir.glob("dq_batch_*.json"))
        ]
