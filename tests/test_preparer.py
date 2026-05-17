def test_fit_transform_returns_xy(sample_df, isolated_cfg):
    from runtime.preparation import DataPreparer
    X, y = DataPreparer(isolated_cfg).fit_transform(sample_df)
    assert X is not None and y is not None
    assert len(X) == len(y)


def test_target_not_in_features(sample_df, isolated_cfg):
    from runtime.preparation import DataPreparer
    X, _ = DataPreparer(isolated_cfg).fit_transform(sample_df)
    target = isolated_cfg["data"]["target_column"]
    cols = X.columns.tolist() if hasattr(X, "columns") else []
    assert target not in cols


def test_transform_consistent_shape(sample_df, isolated_cfg):
    from runtime.preparation import DataPreparer
    preparer = DataPreparer(isolated_cfg)
    X_train, _ = preparer.fit_transform(sample_df)
    X_new = preparer.transform(sample_df.head(10))
    assert X_new.shape[1] == X_train.shape[1]
