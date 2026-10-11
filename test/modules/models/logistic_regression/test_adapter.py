import pandas as pd
import pytest

from securebank.modules.models.logistic_regression.adapter import (
    LogisticRegressionAdapter,
)


@pytest.fixture
def sample_training_df():
    return pd.DataFrame({
        "trans_num": ["abc123", "def456", "ghi789"],
        "index": [1, 2, 3],
        "index_customer": [10, 20, 10],
        "cc_num": [123, 456, 123],
        "first": ["Jane", "John", "Jane"],
        "last": ["Doe", "Smith", "Doe"],
        "street": ["Main St", "Second St", "Main St"],
        "city": ["Casper", "Denver", "Casper"],
        "zip": [12345, 67890, 12345],
        "minute": [23, 12, 28],
        "seconds": [10, 15, 30],
        "unix_time": [1355603000, 1355604000, 1355603300],
        "merchant": ["Fraud_StoreA", "Fraud_StoreA", "Fraud_StoreB"],
        "category": ["GROCERY_POS", "PERSONAL_CARE", "GROCERY_POS"],
        "amt": [66.01, 91.73, 25.50],
        "merch_lat": [26.88, 42.53, 27.10],
        "merch_long": [-79.27, -86.61, -79.50],
        "is_fraud": [0, 1, 0],
        "sex": ["F", "M", "F"],
        "state": ["WY", "CO", "WY"],
        "lat": [43.0, 40.0, 43.0],
        "long": [-108.8, -105.2, -108.8],
        "city_pop": [1000, 5000, 1000],
        "job": ["Manager", "Engineer", "Manager"],
        "dob": ["1986-02-17", "1990-01-01", "1986-02-17"],
        "day_of_week": ["Sunday", "Monday", "Sunday"],
        "hour": [20, 11, 20],
        "day_date": [15, 16, 15],
        "month_date": ["December", "January", "December"],
        "year_date": [2019, 2019, 2019],
    })


def test_prepare_training_data_returns_X_and_y(sample_training_df):
    adapter = LogisticRegressionAdapter()

    X, y = adapter.prepare_training_data(sample_training_df)

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == 3
    assert len(y) == 3


def test_prepare_training_data_drops_unused_and_target_columns(sample_training_df):
    adapter = LogisticRegressionAdapter()

    X, y = adapter.prepare_training_data(sample_training_df)

    for column in adapter.DROP_COLUMNS:
        assert column not in X.columns

    assert adapter.TARGET_COLUMN not in X.columns
    assert y.tolist() == [0, 1, 0]


def test_prepare_training_data_creates_derived_features(sample_training_df):
    adapter = LogisticRegressionAdapter()

    X, _ = adapter.prepare_training_data(sample_training_df)

    assert "age" in X.columns
    assert "dob" not in X.columns

    assert "minutes_since_previous_transaction" in X.columns
    assert "repeat_within_5_minutes" in X.columns
    assert "repeat_within_30_minutes" in X.columns
    assert "repeat_within_1_hour" in X.columns

    assert "minute_of_day" in X.columns
    assert "is_night" in X.columns
    assert "is_evening" in X.columns
    assert "is_weekend" in X.columns


def test_prepare_training_data_does_not_create_removed_features(sample_training_df):
    adapter = LogisticRegressionAdapter()

    X, _ = adapter.prepare_training_data(sample_training_df)

    assert "distance" not in X.columns
    assert "city_pop" not in X.columns
    assert "merchant" not in X.columns
    assert "merchant_frequency" not in X.columns


def test_prepare_training_data_encodes_categorical_features(sample_training_df):
    adapter = LogisticRegressionAdapter()

    X, _ = adapter.prepare_training_data(sample_training_df)

    for column in adapter.CATEGORICAL_COLUMNS:
        assert column not in X.columns

    assert "job_frequency" in X.columns

    assert any(column.startswith("category_") for column in X.columns)
    assert any(column.startswith("sex_") for column in X.columns)
    assert any(column.startswith("state_") for column in X.columns)
    assert any(column.startswith("day_of_week_") for column in X.columns)
    assert any(column.startswith("month_date_") for column in X.columns)


def test_prepare_training_data_saves_adapter_state(sample_training_df):
    adapter = LogisticRegressionAdapter()

    X, _ = adapter.prepare_training_data(sample_training_df)

    assert adapter.feature_columns == X.columns.tolist()
    assert "job" in adapter.frequency_maps
    assert "merchant" not in adapter.frequency_maps
    assert adapter.median_minutes_since_previous_transaction >= 0


def test_time_delta_features_are_created_before_cc_num_is_dropped(sample_training_df):
    adapter = LogisticRegressionAdapter()

    X, _ = adapter.prepare_training_data(sample_training_df)

    assert "cc_num" not in X.columns
    assert "minutes_since_previous_transaction" in X.columns

    repeat_columns = [
        "repeat_within_5_minutes",
        "repeat_within_30_minutes",
        "repeat_within_1_hour",
    ]

    for column in repeat_columns:
        assert set(X[column].dropna().unique()).issubset({0, 1})


def test_time_behavior_features_are_created_before_minute_is_dropped(sample_training_df):
    adapter = LogisticRegressionAdapter()

    X, _ = adapter.prepare_training_data(sample_training_df)

    assert "minute" not in X.columns
    assert "seconds" not in X.columns
    assert "minute_of_day" in X.columns


def test_prepare_training_data_requires_target(sample_training_df):
    adapter = LogisticRegressionAdapter()
    df_without_target = sample_training_df.drop(columns=["is_fraud"])

    with pytest.raises(ValueError, match="is_fraud"):
        adapter.prepare_training_data(df_without_target)


def test_prepare_prediction_data_aligns_to_training_columns(sample_training_df):
    adapter = LogisticRegressionAdapter()

    X_train, _ = adapter.prepare_training_data(sample_training_df)

    prediction_df = sample_training_df.drop(columns=["is_fraud"]).iloc[[0]].copy()
    X_pred = adapter.prepare_prediction_data(prediction_df)

    assert isinstance(X_pred, pd.DataFrame)
    assert X_pred.columns.tolist() == X_train.columns.tolist()
    assert len(X_pred) == 1


def test_prepare_prediction_data_requires_training_first(sample_training_df):
    adapter = LogisticRegressionAdapter()
    prediction_df = sample_training_df.drop(columns=["is_fraud"]).iloc[[0]].copy()

    with pytest.raises(ValueError, match="fitted"):
        adapter.prepare_prediction_data(prediction_df)