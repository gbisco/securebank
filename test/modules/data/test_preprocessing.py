import json
import pandas as pd
import pytest

from securebank.modules.data.preprocessing import TransactionPreprocessingPipeline
from securebank.modules.log.logger import JsonLogger


@pytest.fixture
def sample_customer_df():
    return pd.DataFrame({
        "cc_num": [123],
        "sex": ["F"],
        "state": ["WY"],
        "lat": [43.0],
        "long": [-108.8],
        "city_pop": [1000],
        "job": ["Manager"],
        "dob": ["1986-02-17"],
    })


@pytest.fixture
def sample_transaction_df(tmp_path):
    df = pd.DataFrame({
        "cc_num": [123],
        "trans_date_trans_time": ["2019-12-15 20:23:10"],
        "unix_time": [None],
        "merchant": [None],
        "category": [None],
        "amt": [66.01],
        "merch_lat": [26.88],
        "merch_long": [-79.27],
    })

    df.index = ["abc123"]
    df.index.name = "trans_num"

    file_path = tmp_path / "transactions.parquet"
    df.to_parquet(file_path)

    return file_path


@pytest.fixture
def sample_customer_file(tmp_path, sample_customer_df):
    file_path = tmp_path / "customer.csv"
    sample_customer_df.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def sample_fraud_file(tmp_path):
    fraud_dict = {
        "abc123": 1
    }

    file_path = tmp_path / "fraud.json"
    with open(file_path, "w") as f:
        json.dump(fraud_dict, f)

    return file_path


def test_run_training_preprocessing(
    tmp_path,
    sample_customer_file,
    sample_transaction_df,
    sample_fraud_file,
):
    pipeline = TransactionPreprocessingPipeline(
        storage_path=str(sample_customer_file.parent)
    )
    pipeline.logger = JsonLogger(log_dir=tmp_path / "logs")

    df = pipeline.run_training_preprocessing(
        customer_filename=sample_customer_file.name,
        transaction_filename=sample_transaction_df.name,
        fraud_filename=sample_fraud_file.name,
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1

    assert "cc_num" in df.columns
    assert "amt" in df.columns
    assert "is_fraud" in df.columns

    assert "hour" in df.columns
    assert "day_of_week" in df.columns


def test_run_prediction_preprocessing(tmp_path):
    pipeline = TransactionPreprocessingPipeline()
    pipeline.logger = JsonLogger(log_dir=tmp_path / "logs")

    input_data = {
        "trans_date_trans_time": "2024-09-15 12:34:56",
        "cc_num": 38859492057661,
        "unix_time": None,
        "merchant": None,
        "category": None,
        "amt": 50.75,
        "merch_lat": 43.15,
        "merch_long": -112.15,
    }

    df = pipeline.run_prediction_preprocessing(input_data)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1

    assert "cc_num" in df.columns
    assert "amt" in df.columns

    # fill applied AFTER preprocessing
    assert df["merchant"].iloc[0] == "unknown"

    assert "hour" in df.columns
    assert "day_of_week" in df.columns


def test_run_prediction_preprocessing_with_customer_enrichment(
    tmp_path,
    sample_customer_file,
):
    pipeline = TransactionPreprocessingPipeline(
        storage_path=str(sample_customer_file.parent)
    )
    pipeline.logger = JsonLogger(log_dir=tmp_path / "logs")

    input_data = {
        "trans_date_trans_time": "2024-09-15 12:34:56",
        "cc_num": 123,
        "unix_time": None,
        "merchant": None,
        "category": None,
        "amt": 50.75,
        "merch_lat": 43.15,
        "merch_long": -112.15,
    }

    df = pipeline.run_prediction_preprocessing(
        input_data=input_data,
        customer_filename=sample_customer_file.name,
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1

    # transaction fields
    assert "cc_num" in df.columns
    assert "amt" in df.columns

    # enrichment fields
    assert "sex" in df.columns
    assert "state" in df.columns
    assert "lat" in df.columns
    assert "long" in df.columns
    assert "city_pop" in df.columns
    assert "job" in df.columns
    assert "dob" in df.columns

    # check enrichment values
    assert df["sex"].iloc[0] == "F"
    assert df["state"].iloc[0] == "WY"
    assert df["lat"].iloc[0] == 43.0
    assert df["long"].iloc[0] == -108.8

    # fill still works after merge
    assert df["merchant"].iloc[0] == "unknown"

    # date features
    assert "hour" in df.columns
    assert "day_of_week" in df.columns