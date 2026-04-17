import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from runtime.ingestion import DataIngester
from runtime.analysis import DataAnalyzer
from runtime.preparation import DataPreparer
from runtime.training import ModelTrainer
from runtime.validation import ModelValidator
from runtime.serving import ModelServer


def setup_logging(log_path: str = "artifacts/logs/pipeline.log"):
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )


def load_config(path: str = "configs/config.yaml") -> dict:
    cfg_dir = Path(path).parent
    parts = ["data.yaml", "analysis.yaml", "training.yaml", "serving.yaml"]
    merged = {}
    for part in parts:
        p = cfg_dir / part
        if p.exists():
            with open(p) as f:
                merged.update(yaml.safe_load(f) or {})
    if not merged:
        with open(path) as f:
            merged = yaml.safe_load(f)
    return merged


def run_update(config: dict) -> bool:
    logger = logging.getLogger("update")
    logger.info("=== UPDATE mode started ===")
    try:
        ingester = DataIngester(config)
        analyzer = DataAnalyzer(config)
        preparer = DataPreparer(config)
        trainer = ModelTrainer(config)
        validator = ModelValidator(config)
        server = ModelServer(config)
        test_size = config["validation"].get("test_size", 0.2)
        batch_df, batch_meta = ingester.ingest_next_batch()
        if batch_df is None:
            logger.warning("No new batches available.")
            return False
        batch_idx = batch_meta["batch_index"]
        logger.info("Processing batch %d (%d rows).", batch_idx, len(batch_df))
        dq = analyzer.compute_data_quality(batch_df, batch_idx)
        analyzer.save_dq_report(dq, batch_idx)
        passed, issues = analyzer.check_quality_thresholds(dq)
        if not passed:
            logger.warning(
                "DQ issues on batch %d: %s. Continuing with cleaning.",
                batch_idx,
                issues)
        batch_df = analyzer.clean_data(batch_df, dq)
        analyzer.auto_eda(batch_df, batch_idx)
        prev_path = ingester.raw_dir / f"batch_{batch_idx - 1:03d}.csv"
        if batch_idx > 0 and prev_path.exists():
            df_prev = pd.read_csv(prev_path)
            drift_result = analyzer.detect_data_drift(df_prev, batch_df)
            logger.info("Drift check: %s", drift_result)
        accum_df = ingester.load_accumulated_data()
        accum_df = analyzer.clean_data(accum_df, dq)
        features_config_path = Path(
            config["ingestion"]["meta_dir"]) / "features_config.json"
        if not features_config_path.exists():
            stats_eda = analyzer.statsmodels_eda(accum_df, batch_idx)
            features_to_drop = analyzer.get_features_to_drop(stats_eda)
            with open(features_config_path, "w") as _f:
                json.dump({"features_to_drop": features_to_drop}, _f, indent=2)
            logger.info(
                "Feature selection fixed: dropping %s",
                features_to_drop)
        else:
            with open(features_config_path) as _f:
                features_to_drop = json.load(_f)["features_to_drop"]
            logger.info(
                "Using fixed feature selection: dropping %s",
                features_to_drop)
        X_accum, y_accum = preparer.fit_transform(
            accum_df, features_to_drop=features_to_drop)
        cat_features = preparer.get_cat_features()
        X_new = preparer.transform(batch_df)
        y_new = preparer.transform_target(
            batch_df[config["data"]["target_column"]].values
        )
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_accum, y_accum,
            test_size=test_size,
            random_state=42,
        )
        logger.info("Incremental training on %d new rows.", len(batch_df))
        models = trainer.train_all(X_new, y_new, cat_features)
        results = {}
        for name, model in models.items():
            m = validator.evaluate(model, X_te, y_te, label=name)
            results[name] = m
        best_name = min(results, key=lambda k: results[k]["rmse"])
        best_model = models[best_name]
        best_metrics = results[best_name]
        logger.info("Best model: %s  RMSE=%.4f  R²=%.4f",
                    best_name, best_metrics["rmse"], best_metrics["r2"])
        promote = validator.should_promote(best_metrics, best_name)
        for name, model in models.items():
            validator.save_model(model, results[name], name, batch_idx)
        drift = validator.detect_model_drift()
        if drift.get("drift_detected"):
            logger.warning("Model performance drift: %s", drift)
        if promote:
            server.serialize_model(
                best_model, preparer, best_name, best_metrics)
        logger.info("=== UPDATE mode completed successfully ===")
        return True
    except Exception as e:
        logging.getLogger("update").exception("Update failed: %s", e)
        return False


def run_inference(config: dict, file_path: str) -> str:
    logger = logging.getLogger("inference")
    logger.info("=== INFERENCE mode: %s ===", file_path)
    if not Path(file_path).exists():
        logger.error("Input file not found: %s", file_path)
        sys.exit(1)
    server = ModelServer(config)
    if not server.has_production_model():
        logger.error("No production model. Run 'update' first.")
        sys.exit(1)
    df = pd.read_csv(file_path) if file_path.endswith(
        ".csv") else pd.read_excel(file_path)
    result_df = server.predict(df)
    out_path = server.save_predictions(result_df)
    logger.info("Predictions saved to %s", out_path)
    return str(out_path)


def run_summary(config: dict) -> str:
    logger = logging.getLogger("summary")
    logger.info("=== SUMMARY mode ===")
    analyzer = DataAnalyzer(config)
    validator = ModelValidator(config)
    server = ModelServer(config)
    ingester = DataIngester(config)
    state = ingester.get_state()
    dq_reports = analyzer.load_all_dq_reports()
    metrics_history = validator.get_metrics_history()
    monitoring = server.get_monitoring_history()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Starbucks ML Pipeline — Monitoring Summary",
        "",
        f"> Generated: {ts}  ",
        f"> Task: Regression — predict `total_spend` (CatBoost, incremental)",
        "",
        "## Pipeline State",
        "",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| Total batches | {state.get('total_batches', 'N/A')} |",
        f"| Processed batches | {state.get('current_batch', 0)} |",
        f"| Last update | {state.get('last_ingested_at', 'N/A')} |",
        "",
        "## Data Quality",
        "",
        "| Batch | Rows | Missing avg | Duplicates | Status |",
        "|-------|------|-------------|------------|--------|",
    ]
    for dq in dq_reports:
        status = "✅ OK" if dq["passes_missing_check"] else "⚠️ WARN"
        lines.append(
            f"| {dq['batch_index']} | {dq['n_rows']:,} | "
            f"{dq['avg_missing_pct']:.1%} | "
            f"{dq['duplicate_pct']:.1%} | {status} |"
        )
    lines += [
        "",
        "## Model Metrics History",
        "",
        "| Timestamp | Batch | Model | RMSE | MAE | R² |",
        "|-----------|-------|-------|------|-----|-----|",
    ]
    for m in metrics_history:
        lines.append(
            f"| {m['timestamp'][:16]} | {m['batch_index']} | {m['model_name']} | "
            f"{m.get('rmse', 0):.4f} | {m.get('mae', 0):.4f} | {m.get('r2', 0):.4f} |")
    lines += ["", "## Best Model", ""]
    best = validator.get_best_model_entry()
    if best:
        lines += [
            f"**{best['model_name']}** — batch {best['batch_index']}",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| RMSE | {best['metrics'].get('rmse', 0):.4f} |",
            f"| MAE  | {best['metrics'].get('mae', 0):.4f} |",
            f"| R²   | {best['metrics'].get('r2', 0):.4f} |",
            f"| Samples | {best['metrics'].get('n_samples', 0):,} |",
        ]
    else:
        lines.append("_No models trained yet._")
    lines += ["", "## Inference Performance", ""]
    if monitoring:
        lines += [
            "| Timestamp | Model | Samples | Latency (ms) | Memory (KB) |",
            "|-----------|-------|---------|-------------|-------------|",
        ]
        for rec in monitoring[-10:]:
            lines.append(
                f"| {rec['timestamp'][:16]} | {rec.get('model_name','?')} | "
                f"{rec.get('n_samples',0):,} | "
                f"{rec.get('inference_time_ms',0):.1f} | "
                f"{rec.get('peak_memory_kb',0):.0f} |"
            )
    else:
        lines.append("_No inference calls recorded yet._")
    report_dir = Path(config["reports"]["output_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    file_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = report_dir / f"summary_{file_ts}.md"
    content = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write(content)
    print(content)
    logger.info("Summary saved to %s", out_path)
    return str(out_path)


def main():
    parser = argparse.ArgumentParser(description="Starbucks MLOps Pipeline")
    parser.add_argument(
        "-mode",
        required=True,
        choices=[
            "update",
            "inference",
            "summary"])
    parser.add_argument(
        "-file",
        default=None,
        help="Input file path (required for inference mode)")
    parser.add_argument(
        "-config",
        default="configs/config.yaml",
        help="Config file path")
    args = parser.parse_args()
    setup_logging()
    config = load_config(args.config)
    if args.mode == "update":
        success = run_update(config)
        print(success)
        sys.exit(0 if success else 1)
    elif args.mode == "inference":
        if not args.file:
            print(
                "Error: -file is required for inference mode.",
                file=sys.stderr)
            sys.exit(1)
        path = run_inference(config, args.file)
        print(path)
    elif args.mode == "summary":
        path = run_summary(config)
        print(path)


if __name__ == "__main__":
    main()
