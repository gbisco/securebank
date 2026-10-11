import json
import pandas as pd
import pytest

from securebank.modules.data.raw_data_handler import RawDataHandlerV2


@pytest.fixture
def handler(tmp_path):
    return RawDataHandlerV2(storage_path=str(tmp_path))


@pytest.fixture
def customer_df():
    return pd.DataFrame({
        "cc_num": [123, 456],
        "sex": ["F", "M"],
        "state": ["WY", "OK"],
        "lat": [43.0048, 36.9412],
        "long": [-108.8964, -98.2458],
        "city_pop": [1645, 83],
        "job": ["Manager", "Clerk"],
        "dob": ["Feb 17, 1986", "Jul 28, 1958"],
    })


@pytest.fixture
def transaction_df():
    df = pd.DataFrame({
        "cc_num": [123, 456],
        "trans_date_trans_time": ["2019-12-15 20:23:10", "2019-01-05 11:12:15"],
        "unix_time": [1355602990, None],
        "merchant": ["fraud_Yost-Rogahn", None],
        "category": ["personal_care", None],
        "amt": [66.01, 91.73],
        "merch_lat": [26.880712, 42.533807],
        "merch_long": [-79.277659, -86.610405],
    })

    df.index = ["abc123", "def456"]
    df.index.name = "trans_num"

    return df


@pytest.fixture
def fraud_df():
    return pd.DataFrame({
        "trans_id": ["abc123", "def456"],
        "is_fraud": [0, 1],
    })


def test_ensure_dataframe_from_dict(handler):
    data = {
        "cc_num": 123,
        "amt": 50.75,
        "trans_date_trans_time": "2024-09-15 12:34:56",
    }

    df = handler.ensure_dataframe(data)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert "cc_num" in df.columns


def test_normalize_transaction_index(handler, transaction_df):
    normalized_df = handler.normalize_transaction_index(transaction_df)

    assert "trans_num" in normalized_df.columns
    assert normalized_df.loc[0, "trans_num"] == "abc123"


def test_attach_fraud_labels(handler, transaction_df, fraud_df):
    transaction_df = handler.normalize_transaction_index(transaction_df)

    merged_df = handler.attach_fraud_labels(transaction_df, fraud_df)

    assert "is_fraud" in merged_df.columns
    assert merged_df["is_fraud"].tolist() == [0, 1]


def test_merge_customer_data(handler, transaction_df, customer_df):
    transaction_df = handler.normalize_transaction_index(transaction_df)

    merged_df = handler.merge_customer_data(transaction_df, customer_df)

    assert "state" in merged_df.columns
    assert "job" in merged_df.columns
    assert len(merged_df) == 2


def test_build_training_frame(handler, customer_df, transaction_df, fraud_df):
    training_df = handler.build_training_frame(
        customer_df=customer_df,
        transaction_df=transaction_df,
        fraud_df=fraud_df,
    )

    assert "trans_num" in training_df.columns
    assert "is_fraud" in training_df.columns
    assert "state" in training_df.columns
    assert len(training_df) == 2


def test_build_prediction_frame(handler):
    input_data = {
        "trans_date_trans_time": "2024-09-15 12:34:56",
        "cc_num": 38859492057661,
        "unix_time": 1694775296,
        "merchant": "fraud_Lind-Buckridge",
        "category": "entertainment",
        "amt": 50.75,
        "merch_lat": 43.150704,
        "merch_long": -112.154481,
    }

    prediction_df = handler.build_prediction_frame(input_data)

    assert isinstance(prediction_df, pd.DataFrame)
    assert len(prediction_df) == 1
    assert "cc_num" in prediction_df.columns
    assert "amt" in prediction_df.columns


def test_convert_dates(handler, transaction_df):
    transaction_df = handler.normalize_transaction_index(transaction_df)

    converted_df = handler.convert_dates(transaction_df)

    assert "trans_date_trans_time" not in converted_df.columns
    assert "day_of_week" in converted_df.columns
    assert "hour" in converted_df.columns
    assert "month_date" in converted_df.columns


def test_drop_missing_required(handler):
    df = pd.DataFrame({
        "cc_num": [123, None],
        "amt": [50.0, 20.0],
        "trans_date_trans_time": ["2024-01-01 12:00:00", "2024-01-01 13:00:00"],
    })

    cleaned_df = handler.drop_missing_required(
        df,
        required_columns=["cc_num", "amt", "trans_date_trans_time"],
    )

    assert len(cleaned_df) == 1