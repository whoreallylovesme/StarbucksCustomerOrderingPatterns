def test_config_valid(cfg):
    for key in ("data", "training", "validation", "serving"):
        assert key in cfg, f"Missing config section: '{key}'"
    assert cfg["data"].get("target_column"), "target_column must be set"
    assert cfg["data"]["batch_size"] > 0
    assert cfg["training"]["catboost"]["iterations_per_batch"] > 0
    assert 0 < cfg["training"]["catboost"]["learning_rate"] <= 1
    assert 0 < cfg["validation"]["test_size"] < 1
