import numpy as np


def _train(sample_df, isolated_cfg):
    from runtime.preparation import DataPreparer
    from runtime.training import ModelTrainer
    preparer = DataPreparer(isolated_cfg)
    X, y = preparer.fit_transform(sample_df)
    models = ModelTrainer(isolated_cfg).train_all(X, y, preparer.get_cat_features())
    return models, X, y


def test_trainer_returns_models(sample_df, isolated_cfg):
    models, _, _ = _train(sample_df, isolated_cfg)
    assert isinstance(models, dict) and len(models) >= 1


def test_trainer_predictions_finite(sample_df, isolated_cfg):
    models, X, y = _train(sample_df, isolated_cfg)
    preds = next(iter(models.values())).predict(X)
    assert len(preds) == len(y)
    assert np.isfinite(preds).all()


def test_validator_metrics_keys(sample_df, isolated_cfg):
    from runtime.validation import ModelValidator
    models, X, y = _train(sample_df, isolated_cfg)
    model = next(iter(models.values()))
    metrics = ModelValidator(isolated_cfg).evaluate(model, X, y, label="test")
    assert "rmse" in metrics and "r2" in metrics
    assert metrics["rmse"] >= 0
