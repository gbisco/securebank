from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from securebank.modules.services.training_service import TrainingService


@pytest.fixture
def sample_validation_df():
    return pd.DataFrame({
        "is_fraud": [0, 1, 0, 1],
        "amt": [10.0, 500.0, 20.0, 800.0],
    })


@pytest.fixture
def sample_test_df():
    return pd.DataFrame({
        "is_fraud": [0, 1, 1, 0],
        "amt": [15.0, 600.0, 700.0, 25.0],
    })


@pytest.fixture
def training_result(sample_validation_df, sample_test_df):
    return {
        "model_type": "gradient_boosting",
        "model_path": "fake_model.joblib",
        "splits": {
            "train_df": pd.DataFrame({
                "is_fraud": [0, 1],
                "amt": [10.0, 500.0],
            }),
            "validation_df": sample_validation_df,
            "test_df": sample_test_df,
        },
    }


@pytest.fixture
def training_service(tmp_path):
    service = TrainingService(
        data_sources_path=str(tmp_path / "data_sources"),
        model_storage_path=str(tmp_path / "storage" / "models"),
        default_model_metadata_path=str(
            tmp_path / "storage" / "models" / "default_model.json"
        ),
        model_type="gradient_boosting",
        model_config={"max_iter": 10},
    )

    service.logger = MagicMock()

    return service


def test_constructor_creates_expected_directories(tmp_path):
    model_storage_path = tmp_path / "storage" / "models"
    default_metadata_path = model_storage_path / "default_model.json"

    service = TrainingService(
        data_sources_path=str(tmp_path / "data_sources"),
        model_storage_path=str(model_storage_path),
        default_model_metadata_path=str(default_metadata_path),
    )

    assert service.data_sources_path == tmp_path / "data_sources"
    assert service.model_storage_path == model_storage_path
    assert service.default_model_metadata_path == default_metadata_path
    assert model_storage_path.exists()
    assert default_metadata_path.parent.exists()


def test_build_model_path_uses_model_type_and_timestamp(training_service):
    result = training_service._build_model_path("20260503_120000")

    expected = (
        training_service.model_storage_path
        / "model_gradient_boosting_20260503_120000.joblib"
    )

    assert result == expected


def test_timestamp_returns_filesystem_safe_value(training_service):
    timestamp = training_service._timestamp()

    assert isinstance(timestamp, str)
    assert "_" in timestamp
    assert ":" not in timestamp
    assert "/" not in timestamp


def test_write_json_creates_file(training_service, tmp_path):
    output_path = tmp_path / "nested" / "result.json"

    training_service._write_json(
        path=output_path,
        data={"status": "ok", "value": 1},
    )

    assert output_path.exists()

    saved_text = output_path.read_text()

    assert '"status": "ok"' in saved_text
    assert '"value": 1' in saved_text


def test_save_model_metadata_writes_versioned_metadata(training_service):
    metadata = {
        "model_path": "fake_model.joblib",
        "model_type": "gradient_boosting",
        "selected_threshold": 0.975,
    }

    metadata_path = training_service._save_model_metadata(
        metadata=metadata,
        timestamp="20260503_120000",
    )

    assert metadata_path.exists()
    assert metadata_path.name == "model_gradient_boosting_20260503_120000_metadata.json"

    saved_text = metadata_path.read_text()

    assert "fake_model.joblib" in saved_text
    assert "0.975" in saved_text


def test_save_default_model_metadata_writes_compact_default_file(training_service):
    metadata = {
        "model_path": "fake_model.joblib",
        "model_type": "gradient_boosting",
        "model_config": {"max_iter": 10},
        "selected_threshold": 0.975,
        "created_at": "20260503_120000",
        "selection_rule": "highest_f1_with_min_precision_recall",
        "selected_validation_result": {"f1": 0.8},
        "test_result": {"f1": 0.79},
    }

    training_service._save_default_model_metadata(metadata)

    assert training_service.default_model_metadata_path.exists()

    saved_text = training_service.default_model_metadata_path.read_text()

    assert "fake_model.joblib" in saved_text
    assert "gradient_boosting" in saved_text
    assert "0.975" in saved_text


def test_compact_training_summary_removes_split_dataframes(
    training_service,
    training_result,
):
    summary = training_service._compact_training_summary(training_result)

    assert "splits" not in summary
    assert "split_shapes" in summary
    assert summary["split_shapes"]["train_df"]["rows"] == 2
    assert summary["split_shapes"]["validation_df"]["rows"] == 4
    assert summary["split_shapes"]["test_df"]["rows"] == 4


def test_select_best_threshold_uses_eligible_highest_f1(training_service):
    threshold_results = [
        {
            "threshold": 0.90,
            "precision": 0.65,
            "recall": 0.90,
            "f1": 0.75,
            "balanced_accuracy": 0.88,
        },
        {
            "threshold": 0.95,
            "precision": 0.80,
            "recall": 0.75,
            "f1": 0.77,
            "balanced_accuracy": 0.86,
        },
        {
            "threshold": 0.975,
            "precision": 0.85,
            "recall": 0.76,
            "f1": 0.80,
            "balanced_accuracy": 0.88,
        },
    ]

    selected = training_service._select_best_threshold(
        threshold_results=threshold_results,
        min_precision=0.70,
        min_recall=0.70,
    )

    assert selected["threshold"] == 0.975
    assert selected["selection_rule"] == "highest_f1_with_min_precision_recall"
    assert selected["met_minimum_precision_recall"] is True


def test_select_best_threshold_falls_back_to_highest_f1(training_service):
    threshold_results = [
        {
            "threshold": 0.50,
            "precision": 0.20,
            "recall": 0.80,
            "f1": 0.32,
            "balanced_accuracy": 0.70,
        },
        {
            "threshold": 0.90,
            "precision": 0.40,
            "recall": 0.40,
            "f1": 0.40,
            "balanced_accuracy": 0.65,
        },
    ]

    selected = training_service._select_best_threshold(
        threshold_results=threshold_results,
        min_precision=0.70,
        min_recall=0.70,
    )

    assert selected["threshold"] == 0.90
    assert selected["selection_rule"] == "fallback_highest_f1"
    assert selected["met_minimum_precision_recall"] is False


def test_select_best_threshold_rejects_empty_results(training_service):
    with pytest.raises(ValueError, match="threshold_results cannot be empty"):
        training_service._select_best_threshold(
            threshold_results=[],
            min_precision=0.70,
            min_recall=0.70,
        )


def test_evaluate_thresholds_calls_single_threshold_for_each_threshold(
    training_service,
    sample_validation_df,
):
    training_service._evaluate_single_threshold = MagicMock(
        side_effect=[
            {"threshold": 0.90, "f1": 0.70},
            {"threshold": 0.95, "f1": 0.80},
        ]
    )

    results = training_service._evaluate_thresholds(
        model_path="fake_model.joblib",
        evaluation_df=sample_validation_df,
        thresholds=[0.90, 0.95],
    )

    assert results == [
        {"threshold": 0.90, "f1": 0.70},
        {"threshold": 0.95, "f1": 0.80},
    ]

    assert training_service._evaluate_single_threshold.call_count == 2


def test_run_training_orchestrator_calls_existing_orchestrator(
    training_service,
    training_result,
):
    mock_orchestrator = MagicMock()
    mock_orchestrator.run.return_value = training_result

    with patch(
        "securebank.modules.services.training_service.TrainingOrchestrator",
        return_value=mock_orchestrator,
    ) as mock_training_orchestrator:
        result = training_service._run_training_orchestrator(
            model_path=Path("fake_model.joblib"),
            customer_filename="customer_release.csv",
            transaction_filename="transactions_release.parquet",
            fraud_filename="fraud_release.json",
            split_strategy="holdout",
        )

    assert result == training_result

    mock_training_orchestrator.assert_called_once()
    mock_orchestrator.run.assert_called_once_with(
        customer_filename="customer_release.csv",
        transaction_filename="transactions_release.parquet",
        fraud_filename="fraud_release.json",
        model_save_path="fake_model.joblib",
    )


def test_evaluate_single_threshold_uses_prediction_orchestrator(
    training_service,
    sample_validation_df,
):
    mock_prediction_orchestrator = MagicMock()
    mock_prediction_orchestrator.run.return_value = {
        "predictions": [0, 1, 0, 1],
        "probabilities": [0.01, 0.99, 0.05, 0.95],
    }

    with patch(
        "securebank.modules.services.training_service.PredictionOrchestrator",
        return_value=mock_prediction_orchestrator,
    ):
        result = training_service._evaluate_single_threshold(
            model_path="fake_model.joblib",
            evaluation_df=sample_validation_df,
            threshold=0.95,
        )

    assert result["threshold"] == 0.95
    assert result["accuracy"] == 1.0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0
    assert result["true_positive"] == 2
    assert result["false_positive"] == 0
    assert result["false_negative"] == 0
    assert result["true_negative"] == 2

    mock_prediction_orchestrator.run.assert_called_once_with(
        input_data=sample_validation_df,
        threshold=0.95,
        preprocessed=True,
    )


def test_train_and_select_model_success_flow(training_service, training_result):
    training_service._timestamp = MagicMock(return_value="20260503_120000")
    training_service._run_training_orchestrator = MagicMock(return_value=training_result)
    training_service._evaluate_thresholds = MagicMock(return_value=[
        {
            "threshold": 0.95,
            "precision": 0.80,
            "recall": 0.75,
            "f1": 0.77,
            "balanced_accuracy": 0.85,
        },
        {
            "threshold": 0.975,
            "precision": 0.86,
            "recall": 0.76,
            "f1": 0.81,
            "balanced_accuracy": 0.88,
        },
    ])
    training_service._evaluate_single_threshold = MagicMock(return_value={
        "threshold": 0.975,
        "precision": 0.85,
        "recall": 0.76,
        "f1": 0.80,
        "balanced_accuracy": 0.87,
    })

    result = training_service.train_and_select_model(
        register_as_default=True,
    )

    assert result["status"] == "success"
    assert result["selected_threshold"] == 0.975
    assert result["default_model_updated"] is True
    assert Path(result["metadata_path"]).exists()
    assert training_service.default_model_metadata_path.exists()


def test_train_and_select_model_raises_when_splits_missing(
    training_service,
):
    training_service._run_training_orchestrator = MagicMock(return_value={})

    with pytest.raises(ValueError, match="dataset splits"):
        training_service.train_and_select_model()