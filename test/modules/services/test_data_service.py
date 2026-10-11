from unittest.mock import MagicMock

import pandas as pd
import pytest

from securebank.modules.services.dataset_service import DatasetService


@pytest.fixture
def sample_dataset_df():
    return pd.DataFrame({
        "trans_num": ["t1", "t2", "t3"],
        "cc_num": [111, 222, 111],
        "amt": [10.0, 500.0, 25.0],
        "is_fraud": [0, 1, 0],
        "category": ["grocery", "gas", "grocery"],
    })


@pytest.fixture
def dataset_service(tmp_path):
    service = DatasetService(
        data_sources_path=str(tmp_path / "data_sources"),
        dataset_storage_path=str(tmp_path / "storage" / "datasets"),
    )

    service.logger = MagicMock()

    return service


def test_constructor_sets_paths_and_creates_dataset_directory(tmp_path):
    dataset_storage_path = tmp_path / "storage" / "datasets"

    service = DatasetService(
        data_sources_path=str(tmp_path / "data_sources"),
        dataset_storage_path=str(dataset_storage_path),
    )

    assert service.data_sources_path == tmp_path / "data_sources"
    assert service.dataset_storage_path == dataset_storage_path
    assert dataset_storage_path.exists()


def test_build_dataset_path_uses_required_format(dataset_service):
    result = dataset_service._build_dataset_path(
        partition_type="Full Dataset",
        timestamp="20260503_120000",
    )

    expected = (
        dataset_service.dataset_storage_path
        / "dataset_full_dataset_20260503_120000.csv"
    )

    assert result == expected


def test_timestamp_returns_filesystem_safe_value(dataset_service):
    timestamp = dataset_service._timestamp()

    assert isinstance(timestamp, str)
    assert "_" in timestamp
    assert ":" not in timestamp
    assert "/" not in timestamp


def test_summarize_dataset_returns_quality_summary(
    dataset_service,
    sample_dataset_df,
):
    summary = dataset_service._summarize_dataset(sample_dataset_df)

    assert summary["row_count"] == 3
    assert summary["column_count"] == 5
    assert summary["total_missing_values"] == 0
    assert summary["fraud_rate"] == pytest.approx(1 / 3)
    assert summary["fraud_label_counts"] == {"0": 2, "1": 1}
    assert summary["unique_transactions"] == 3
    assert summary["duplicate_transactions"] == 0
    assert summary["unique_customers"] == 2
    assert summary["columns"] == sample_dataset_df.columns.tolist()


def test_summarize_dataset_handles_missing_optional_columns(dataset_service):
    df = pd.DataFrame({
        "amt": [10.0, 20.0],
        "category": ["gas", "grocery"],
    })

    summary = dataset_service._summarize_dataset(df)

    assert summary["row_count"] == 2
    assert summary["column_count"] == 2
    assert summary["total_missing_values"] == 0
    assert "fraud_rate" not in summary
    assert "fraud_label_counts" not in summary
    assert "unique_transactions" not in summary
    assert "unique_customers" not in summary


def test_run_preprocessing_pipeline_calls_existing_pipeline(
    dataset_service,
    sample_dataset_df,
):
    mock_pipeline = MagicMock()
    mock_pipeline.run_training_preprocessing.return_value = sample_dataset_df

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            "securebank.modules.services.dataset_service.TransactionPreprocessingPipeline",
            MagicMock(return_value=mock_pipeline),
        )

        result = dataset_service._run_preprocessing_pipeline(
            customer_filename="customer_release.csv",
            transaction_filename="transactions_release.parquet",
            fraud_filename="fraud_release.json",
        )

    pd.testing.assert_frame_equal(result, sample_dataset_df)

    mock_pipeline.run_training_preprocessing.assert_called_once_with(
        customer_filename="customer_release.csv",
        transaction_filename="transactions_release.parquet",
        fraud_filename="fraud_release.json",
    )


def test_create_dataset_saves_csv_and_returns_response(
    dataset_service,
    sample_dataset_df,
):
    dataset_service._timestamp = MagicMock(return_value="20260503_120000")
    dataset_service._run_preprocessing_pipeline = MagicMock(
        return_value=sample_dataset_df
    )

    result = dataset_service.create_dataset(
        partition_type="full",
    )

    dataset_path = dataset_service.dataset_storage_path / "dataset_full_20260503_120000.csv"

    assert result["status"] == "success"
    assert result["dataset_path"] == str(dataset_path)
    assert result["partition_type"] == "full"
    assert result["created_at"] == "20260503_120000"
    assert result["dataset_summary"]["row_count"] == 3

    assert dataset_path.exists()

    saved_df = pd.read_csv(dataset_path)
    pd.testing.assert_frame_equal(saved_df, sample_dataset_df)

    dataset_service._run_preprocessing_pipeline.assert_called_once_with(
        customer_filename="customer_release.csv",
        transaction_filename="transactions_release.parquet",
        fraud_filename="fraud_release.json",
    )


def test_create_dataset_logs_success(
    dataset_service,
    sample_dataset_df,
):
    dataset_service._timestamp = MagicMock(return_value="20260503_120000")
    dataset_service._run_preprocessing_pipeline = MagicMock(
        return_value=sample_dataset_df
    )

    dataset_service.create_dataset()

    dataset_service.logger.write_log.assert_called_once()

    _, kwargs = dataset_service.logger.write_log.call_args

    assert kwargs["endpoint"] == "create_dataset"
    assert kwargs["status"] == "success"
    assert kwargs["request_data"]["partition_type"] == "full"
    assert kwargs["response_data"]["status"] == "success"


def test_create_dataset_logs_error_and_reraises(dataset_service):
    dataset_service._run_preprocessing_pipeline = MagicMock(
        side_effect=ValueError("preprocessing failed")
    )

    with pytest.raises(ValueError, match="preprocessing failed"):
        dataset_service.create_dataset()

    dataset_service.logger.write_log.assert_called_once()

    _, kwargs = dataset_service.logger.write_log.call_args

    assert kwargs["endpoint"] == "create_dataset"
    assert kwargs["status"] == "error"
    assert "preprocessing failed" in kwargs["error"]