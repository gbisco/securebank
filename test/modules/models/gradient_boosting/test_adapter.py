import pandas as pd
import pytest

from securebank.modules.models.gradient_boosting.adapter import (
    HistGradientBoostingAdapter,
)


@pytest.fixture
def cleaned_training_df():
    return pd.DataFrame({
        "trans_num": ["t1", "t2", "t3", "t4"],
        "index": [1, 2, 3, 4],
        "index_customer": [10, 20, 30, 40],
        "cc_num": [111, 222, 111, 444],
        "unix_time": [1000, 2000, 1300, 4000],
        "merchant": ["store_a", "store_b", "store_a", "store_c"],
        "category": ["grocery", "gas", "grocery", "shopping"],
        "amt": [10.0, 500.0, 15.0, 800.0],
        "merch_lat": [40.0, 41.0, 40.5, 42.0],
        "merch_long": [-75.0, -76.0, -75.5, -77.0],
        "first": ["A", "B", "C", "D"],
        "last": ["One", "Two", "Three", "Four"],
        "sex": ["F", "M", "F", "M"],
        "street": ["s1", "s2", "s3", "s4"],
        "city": ["c1", "c2", "c1", "c4"],
        "state": ["NY", "CA", "NY", "TX"],
        "zip": [10001, 90001, 10002, 73301],
        "lat": [39.0, 40.0, 39.5, 41.0],
        "long": [-74.0, -75.0, -74.5, -76.0],
        "city_pop": [1000, 2000, 1500, 3000],
        "job": ["engineer", "teacher", "engineer", "doctor"],
        "dob": ["1990-01-01", "1980-01-01", "1995-01-01", "1975-01-01"],
        "day_of_week": ["Monday", "Tuesday", "Monday", "Friday"],
        "hour": [9, 23, 10, 1],
        "minute": [10, 20, 30, 40],
        "seconds": [1, 2, 3, 4],
        "day_date": [1, 2, 3, 4],
        "month_date": ["January", "February", "January", "March"],
        "year_date": [2025, 2025, 2025, 2025],
        "is_fraud": [0, 1, 0, 1],
    })


@pytest.fixture
def cleaned_prediction_df():
    return pd.DataFrame({
        "trans_num": ["t5", "t6"],
        "index": [5, 6],
        "index_customer": [50, 60],
        "cc_num": [555, 666],
        "unix_time": [5000, 6000],
        "merchant": ["store_a", "new_store"],
        "category": ["grocery", "gas"],
        "amt": [20.0, 900.0],
        "merch_lat": [40.2, 42.5],
        "merch_long": [-75.2, -77.5],
        "first": ["E", "F"],
        "last": ["Five", "Six"],
        "sex": ["F", "M"],
        "street": ["s5", "s6"],
        "city": ["c5", "c6"],
        "state": ["NY", "CA"],
        "zip": [10003, 90002],
        "lat": [39.2, 41.5],
        "long": [-74.2, -76.5],
        "city_pop": [1200, 3200],
        "job": ["engineer", "unknown_job"],
        "dob": ["1992-01-01", "1982-01-01"],
        "day_of_week": ["Monday", "Tuesday"],
        "hour": [11, 2],
        "minute": [15, 25],
        "seconds": [5, 6],
        "day_date": [5, 6],
        "month_date": ["January", "February"],
        "year_date": [2025, 2025],
    })


def test_constructor_sets_default_state():
    adapter = HistGradientBoostingAdapter()

    assert adapter.feature_columns is None
    assert adapter.frequency_maps == {}
    assert adapter.median_minutes_since_previous_transaction == 0


def test_frequency_encode_column_fits_frequency_map(cleaned_training_df):
    adapter = HistGradientBoostingAdapter()

    result = adapter._frequency_encode_column(
        df=cleaned_training_df,
        column="job",
        fit=True,
    )

    assert "job" not in result.columns
    assert "job_frequency" in result.columns

    assert adapter.frequency_maps["job"]["engineer"] == 0.5
    assert adapter.frequency_maps["job"]["teacher"] == 0.25
    assert adapter.frequency_maps["job"]["doctor"] == 0.25


def test_frequency_encode_column_uses_existing_map_for_prediction(
    cleaned_training_df,
    cleaned_prediction_df,
):
    adapter = HistGradientBoostingAdapter()

    adapter._frequency_encode_column(
        df=cleaned_training_df,
        column="job",
        fit=True,
    )

    result = adapter._frequency_encode_column(
        df=cleaned_prediction_df,
        column="job",
        fit=False,
    )

    assert "job" not in result.columns
    assert "job_frequency" in result.columns

    assert result["job_frequency"].iloc[0] == 0.5
    assert result["job_frequency"].iloc[1] == 0.0


def test_frequency_encode_column_returns_original_when_column_missing(
    cleaned_training_df,
):
    adapter = HistGradientBoostingAdapter()

    result = adapter._frequency_encode_column(
        df=cleaned_training_df,
        column="missing_column",
        fit=True,
    )

    pd.testing.assert_frame_equal(result, cleaned_training_df)


def test_encode_categorical_columns_creates_numeric_features(cleaned_training_df):
    adapter = HistGradientBoostingAdapter()

    result = adapter._encode_categorical_columns(
        df=cleaned_training_df,
        fit=True,
    )

    assert "job" not in result.columns
    assert "category" not in result.columns
    assert "sex" not in result.columns
    assert "state" not in result.columns
    assert "day_of_week" not in result.columns
    assert "month_date" not in result.columns

    assert "job_frequency" in result.columns

    assert any(column.startswith("category_") for column in result.columns)
    assert any(column.startswith("sex_") for column in result.columns)
    assert any(column.startswith("state_") for column in result.columns)
    assert any(column.startswith("day_of_week_") for column in result.columns)
    assert any(column.startswith("month_date_") for column in result.columns)


def test_encode_categorical_columns_does_not_encode_merchant(cleaned_training_df):
    adapter = HistGradientBoostingAdapter()

    result = adapter._encode_categorical_columns(
        df=cleaned_training_df,
        fit=True,
    )

    assert "merchant" in result.columns
    assert "merchant_frequency" not in result.columns
    assert "merchant" not in adapter.frequency_maps


def test_create_age_feature_creates_age_and_drops_dob(cleaned_training_df):
    adapter = HistGradientBoostingAdapter()

    result = adapter._create_age_feature(cleaned_training_df)

    assert "dob" not in result.columns
    assert "age" in result.columns
    assert result["age"].tolist() == [35, 45, 30, 50]


def test_create_age_feature_fills_invalid_dates_with_median(cleaned_training_df):
    adapter = HistGradientBoostingAdapter()

    df = cleaned_training_df.copy()
    df.loc[0, "dob"] = "bad-date"

    result = adapter._create_age_feature(df)

    assert "dob" not in result.columns
    assert "age" in result.columns
    assert result["age"].isna().sum() == 0


def test_create_age_feature_returns_original_when_dob_missing(cleaned_training_df):
    adapter = HistGradientBoostingAdapter()

    df = cleaned_training_df.drop(columns=["dob"])

    result = adapter._create_age_feature(df)

    pd.testing.assert_frame_equal(result, df)


def test_create_time_delta_features_creates_expected_dt_values(cleaned_training_df):
    adapter = HistGradientBoostingAdapter()

    feature_df = cleaned_training_df.drop(columns=["is_fraud"]).copy()
    result = adapter._create_time_delta_features(feature_df, fit=True)

    second_transaction_for_card = result[result["trans_num"] == "t3"].iloc[0]

    assert "minutes_since_previous_transaction" in result.columns
    assert "repeat_within_5_minutes" in result.columns
    assert "repeat_within_30_minutes" in result.columns
    assert "repeat_within_1_hour" in result.columns

    assert second_transaction_for_card["minutes_since_previous_transaction"] == 5
    assert second_transaction_for_card["repeat_within_5_minutes"] == 1
    assert second_transaction_for_card["repeat_within_30_minutes"] == 1
    assert second_transaction_for_card["repeat_within_1_hour"] == 1


def test_create_time_delta_features_returns_original_when_required_columns_missing(
    cleaned_training_df,
):
    adapter = HistGradientBoostingAdapter()

    df = cleaned_training_df.drop(columns=["cc_num"])

    result = adapter._create_time_delta_features(df, fit=True)

    pd.testing.assert_frame_equal(result, df)


def test_create_time_behavior_features_creates_expected_columns(cleaned_training_df):
    adapter = HistGradientBoostingAdapter()

    result = adapter._create_time_behavior_features(cleaned_training_df)

    assert "minute_of_day" in result.columns
    assert "is_night" in result.columns
    assert "is_evening" in result.columns
    assert "is_weekend" in result.columns

    assert result.loc[0, "minute_of_day"] == 550
    assert result.loc[1, "is_evening"] == 1
    assert result.loc[3, "is_night"] == 1


def test_align_prediction_columns_rejects_unfitted_adapter(cleaned_prediction_df):
    adapter = HistGradientBoostingAdapter()

    with pytest.raises(ValueError, match="Adapter must be fitted"):
        adapter._align_prediction_columns(cleaned_prediction_df)


def test_align_prediction_columns_matches_training_columns(
    cleaned_training_df,
    cleaned_prediction_df,
):
    adapter = HistGradientBoostingAdapter()

    X_train, _ = adapter.prepare_training_data(cleaned_training_df)
    X_prediction = adapter.prepare_prediction_data(cleaned_prediction_df)

    assert X_prediction.columns.tolist() == X_train.columns.tolist()


def test_prepare_training_data_rejects_missing_target(cleaned_training_df):
    adapter = HistGradientBoostingAdapter()

    df = cleaned_training_df.drop(columns=["is_fraud"])

    with pytest.raises(ValueError, match="Training data must contain is_fraud"):
        adapter.prepare_training_data(df)


def test_prepare_training_data_returns_X_and_y(cleaned_training_df):
    adapter = HistGradientBoostingAdapter()

    X, y = adapter.prepare_training_data(cleaned_training_df)

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)

    assert len(X) == len(cleaned_training_df)
    assert len(y) == len(cleaned_training_df)

    assert "is_fraud" not in X.columns
    assert "trans_num" not in X.columns
    assert "cc_num" not in X.columns
    assert "dob" not in X.columns
    assert "age" in X.columns

    assert adapter.feature_columns == X.columns.tolist()
    assert y.tolist() == [0, 1, 0, 1]


def test_prepare_training_data_drops_removed_features(cleaned_training_df):
    adapter = HistGradientBoostingAdapter()

    X, _ = adapter.prepare_training_data(cleaned_training_df)

    assert "distance" not in X.columns
    assert "city_pop" not in X.columns
    assert "merchant" not in X.columns
    assert "merchant_frequency" not in X.columns


def test_prepare_training_data_creates_time_features(cleaned_training_df):
    adapter = HistGradientBoostingAdapter()

    X, _ = adapter.prepare_training_data(cleaned_training_df)

    assert "minutes_since_previous_transaction" in X.columns
    assert "repeat_within_5_minutes" in X.columns
    assert "repeat_within_30_minutes" in X.columns
    assert "repeat_within_1_hour" in X.columns
    assert "minute_of_day" in X.columns
    assert "is_night" in X.columns
    assert "is_evening" in X.columns
    assert "is_weekend" in X.columns


def test_prepare_training_data_drops_time_source_columns_after_feature_creation(
    cleaned_training_df,
):
    adapter = HistGradientBoostingAdapter()

    X, _ = adapter.prepare_training_data(cleaned_training_df)

    assert "minute" not in X.columns
    assert "seconds" not in X.columns
    assert "minute_of_day" in X.columns


def test_prepare_training_data_does_not_scale_numeric_values(cleaned_training_df):
    adapter = HistGradientBoostingAdapter()

    X, _ = adapter.prepare_training_data(cleaned_training_df)

    assert X["amt"].tolist() == [10.0, 500.0, 15.0, 800.0]
    assert X["hour"].tolist() == [9, 23, 10, 1]


def test_prepare_training_data_has_no_missing_values(cleaned_training_df):
    adapter = HistGradientBoostingAdapter()

    X, _ = adapter.prepare_training_data(cleaned_training_df)

    assert X.isna().sum().sum() == 0


def test_prepare_training_data_saves_adapter_state(cleaned_training_df):
    adapter = HistGradientBoostingAdapter()

    X, _ = adapter.prepare_training_data(cleaned_training_df)

    assert adapter.feature_columns == X.columns.tolist()
    assert "job" in adapter.frequency_maps
    assert "merchant" not in adapter.frequency_maps
    assert adapter.median_minutes_since_previous_transaction >= 0


def test_prepare_prediction_data_drops_target_if_present(
    cleaned_training_df,
    cleaned_prediction_df,
):
    adapter = HistGradientBoostingAdapter()

    adapter.prepare_training_data(cleaned_training_df)

    prediction_df = cleaned_prediction_df.copy()
    prediction_df["is_fraud"] = [0, 1]

    X_prediction = adapter.prepare_prediction_data(prediction_df)

    assert "is_fraud" not in X_prediction.columns


def test_prepare_prediction_data_returns_aligned_X(
    cleaned_training_df,
    cleaned_prediction_df,
):
    adapter = HistGradientBoostingAdapter()

    X_train, _ = adapter.prepare_training_data(cleaned_training_df)
    X_prediction = adapter.prepare_prediction_data(cleaned_prediction_df)

    assert isinstance(X_prediction, pd.DataFrame)
    assert len(X_prediction) == len(cleaned_prediction_df)
    assert X_prediction.columns.tolist() == X_train.columns.tolist()


def test_prepare_prediction_data_handles_unseen_categories(
    cleaned_training_df,
    cleaned_prediction_df,
):
    adapter = HistGradientBoostingAdapter()

    adapter.prepare_training_data(cleaned_training_df)

    X_prediction = adapter.prepare_prediction_data(cleaned_prediction_df)

    assert "job_frequency" in X_prediction.columns
    assert X_prediction["job_frequency"].iloc[1] == 0.0
    assert "merchant_frequency" not in X_prediction.columns