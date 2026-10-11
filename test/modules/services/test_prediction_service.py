import json
from unittest.mock import MagicMock, patch

import pytest

from securebank.modules.services.prediction_service import PredictionService


@pytest.fixture
def sample_transaction():
    return {
        "trans_date_trans_time": "2024-09-15 12:34:56",
        "cc_num": 38859492057661,
        "unix_time": 1694775296,
        "merchant": "fraud_Lind-Buckridge",
        "category": "entertainment",
        "amt": 50.75,
        "merch_lat": 43.150704,
        "merch_long": -112.154481,
    }


@pytest.fixture
def default_model_metadata():
    return {
        "model_path": "securebank/storage/models/model_gradient_boosting_20260503_120000.joblib",
        "model_type": "gradient_boosting",
        "model_config": {"max_iter": 100},
        "selected_threshold": 0.975,
    }


@pytest.fixture
def metadata_path(tmp_path, default_model_metadata):
    path = tmp_path / "storage" / "models" / "default_model.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as file:
        json.dump(default_model_metadata, file)

    return path


@pytest.fixture
def prediction_service(tmp_path, metadata_path):
    service = PredictionService(
        data_sources_path=str(tmp_path / "data_sources"),
        default_model_metadata_path=str(metadata_path),
        customer_filename="customer_release.csv",
    )

    service.logger = MagicMock()

    return service


def test_constructor_sets_paths_and_defaults(tmp_path, metadata_path):
    service = PredictionService(
        data_sources_path=str(tmp_path / "data_sources"),
        default_model_metadata_path=str(metadata_path),
    )

    assert service.data_sources_path == tmp_path / "data_sources"
    assert service.default_model_metadata_path == metadata_path
    assert service.customer_filename == "customer_release.csv"


def test_load_default_model_metadata_returns_metadata(
    prediction_service,
    default_model_metadata,
):
    metadata = prediction_service._load_default_model_metadata()

    assert metadata == default_model_metadata


def test_load_default_model_metadata_raises_when_file_missing(tmp_path):
    service = PredictionService(
        data_sources_path=str(tmp_path / "data_sources"),
        default_model_metadata_path=str(
            tmp_path / "storage" / "models" / "missing_default_model.json"
        ),
    )

    with pytest.raises(FileNotFoundError, match="Default model metadata not found"):
        service._load_default_model_metadata()


def test_load_default_model_metadata_raises_when_required_key_missing(
    tmp_path,
    default_model_metadata,
):
    metadata = default_model_metadata.copy()
    metadata.pop("selected_threshold")

    path = tmp_path / "storage" / "models" / "default_model.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as file:
        json.dump(metadata, file)

    service = PredictionService(
        data_sources_path=str(tmp_path / "data_sources"),
        default_model_metadata_path=str(path),
    )

    with pytest.raises(ValueError, match="missing required keys"):
        service._load_default_model_metadata()


def test_build_prediction_orchestrator_uses_metadata(prediction_service):
    metadata = {
        "model_path": "fake_model.joblib",
        "model_type": "gradient_boosting",
        "model_config": {"max_iter": 100},
        "selected_threshold": 0.975,
    }

    with patch(
        "securebank.modules.services.prediction_service.PredictionOrchestrator"
    ) as mock_prediction_orchestrator:
        prediction_service._build_prediction_orchestrator(metadata)

    mock_prediction_orchestrator.assert_called_once_with(
        model_path="fake_model.joblib",
        storage_path=str(prediction_service.data_sources_path),
        model_type="gradient_boosting",
        model_config={"max_iter": 100},
    )


def test_convert_prediction_label_returns_fraudulent(prediction_service):
    assert prediction_service._convert_prediction_label(1) == "fraudulent"


def test_convert_prediction_label_returns_legitimate(prediction_service):
    assert prediction_service._convert_prediction_label(0) == "legitimate"


def test_predict_transaction_returns_assignment_ready_response(
    prediction_service,
    sample_transaction,
):
    mock_prediction_orchestrator = MagicMock()
    mock_prediction_orchestrator.run.return_value = {
        "model_path": "fake_model.joblib",
        "model_type": "gradient_boosting",
        "threshold": 0.975,
        "row_count": 1,
        "preprocessed": False,
        "predictions": [1],
        "probabilities": [0.991],
    }

    with patch(
        "securebank.modules.services.prediction_service.PredictionOrchestrator",
        return_value=mock_prediction_orchestrator,
    ):
        result = prediction_service.predict_transaction(sample_transaction)

    assert result["predict"] == "fraudulent"
    assert result["prediction"] == 1
    assert result["fraud_probability"] == 0.991
    assert result["threshold"] == 0.975
    assert result["model_type"] == "gradient_boosting"
    assert result["model_path"] == "fake_model.joblib"

    mock_prediction_orchestrator.run.assert_called_once_with(
        input_data=sample_transaction,
        customer_filename="customer_release.csv",
        threshold=0.975,
        preprocessed=False,
    )


def test_predict_transaction_returns_legitimate_response(
    prediction_service,
    sample_transaction,
):
    mock_prediction_orchestrator = MagicMock()
    mock_prediction_orchestrator.run.return_value = {
        "model_path": "fake_model.joblib",
        "model_type": "gradient_boosting",
        "threshold": 0.975,
        "row_count": 1,
        "preprocessed": False,
        "predictions": [0],
        "probabilities": [0.12],
    }

    with patch(
        "securebank.modules.services.prediction_service.PredictionOrchestrator",
        return_value=mock_prediction_orchestrator,
    ):
        result = prediction_service.predict_transaction(sample_transaction)

    assert result["predict"] == "legitimate"
    assert result["prediction"] == 0
    assert result["fraud_probability"] == 0.12


def test_predict_transaction_logs_success(
    prediction_service,
    sample_transaction,
):
    mock_prediction_orchestrator = MagicMock()
    mock_prediction_orchestrator.run.return_value = {
        "model_path": "fake_model.joblib",
        "model_type": "gradient_boosting",
        "threshold": 0.975,
        "row_count": 1,
        "preprocessed": False,
        "predictions": [1],
        "probabilities": [0.991],
    }

    with patch(
        "securebank.modules.services.prediction_service.PredictionOrchestrator",
        return_value=mock_prediction_orchestrator,
    ):
        prediction_service.predict_transaction(sample_transaction)

    prediction_service.logger.write_log.assert_called_once()

    _, kwargs = prediction_service.logger.write_log.call_args

    assert kwargs["endpoint"] == "predict"
    assert kwargs["status"] == "success"
    assert kwargs["request_data"]["input_data"] == sample_transaction
    assert kwargs["response_data"]["predict"] == "fraudulent"


def test_predict_transaction_logs_error_and_reraises(
    prediction_service,
    sample_transaction,
):
    mock_prediction_orchestrator = MagicMock()
    mock_prediction_orchestrator.run.side_effect = ValueError("prediction failed")

    with patch(
        "securebank.modules.services.prediction_service.PredictionOrchestrator",
        return_value=mock_prediction_orchestrator,
    ):
        with pytest.raises(ValueError, match="prediction failed"):
            prediction_service.predict_transaction(sample_transaction)

    prediction_service.logger.write_log.assert_called_once()

    _, kwargs = prediction_service.logger.write_log.call_args

    assert kwargs["endpoint"] == "predict"
    assert kwargs["status"] == "error"
    assert "prediction failed" in kwargs["error"]


def test_predict_transaction_raises_when_orchestrator_returns_empty_predictions(
    prediction_service,
    sample_transaction,
):
    mock_prediction_orchestrator = MagicMock()
    mock_prediction_orchestrator.run.return_value = {
        "model_path": "fake_model.joblib",
        "model_type": "gradient_boosting",
        "threshold": 0.975,
        "row_count": 1,
        "preprocessed": False,
        "predictions": [],
        "probabilities": [],
    }

    with patch(
        "securebank.modules.services.prediction_service.PredictionOrchestrator",
        return_value=mock_prediction_orchestrator,
    ):
        with pytest.raises(ValueError, match="returned no predictions"):
            prediction_service.predict_transaction(sample_transaction)