import pandas as pd
import pytest
from unittest.mock import MagicMock

from securebank.modules.predict.prediction_orchestrator import PredictionOrchestrator
from securebank.modules.models.model_wrapper import ModelWrapper


@pytest.fixture
def model_path(tmp_path):
    path = tmp_path / "trained_model.joblib"
    path.touch()
    return str(path)


@pytest.fixture
def orchestrator(model_path, monkeypatch):
    monkeypatch.setattr(ModelWrapper, "load", lambda self, path: None)

    return PredictionOrchestrator(
        model_path=model_path,
        storage_path=None,
        model_type="logistic_regression",
        model_config={"activation_function": "sigmoid"},
    )


@pytest.fixture
def input_data():
    return {
        "trans_date_trans_time": "2025-01-01 12:00:00",
        "cc_num": 123,
        "unix_time": 1000,
        "merchant": "store_a",
        "category": "grocery",
        "amt": 25.0,
        "merch_lat": 40.0,
        "merch_long": -75.0,
    }


@pytest.fixture
def prediction_df():
    return pd.DataFrame({
        "cc_num": [123],
        "unix_time": [1000],
        "merchant": ["store_a"],
        "category": ["grocery"],
        "amt": [25.0],
        "merch_lat": [40.0],
        "merch_long": [-75.0],
        "day_of_week": ["Wednesday"],
        "hour": [12],
        "minute": [0],
        "seconds": [0],
        "day_date": [1],
        "month_date": ["January"],
        "year_date": [2025],
    })


@pytest.fixture
def X_prediction():
    return pd.DataFrame({
        "amt": [25.0],
        "hour": [12],
        "merchant_frequency": [0.5],
    })


def test_constructor_loads_model(model_path, monkeypatch):
    load_mock = MagicMock()
    monkeypatch.setattr(ModelWrapper, "load", load_mock)

    orchestrator = PredictionOrchestrator(
        model_path=model_path,
        storage_path=None,
        model_type="logistic_regression",
        model_config={"activation_function": "sigmoid"},
    )

    assert orchestrator.model_path == model_path
    assert orchestrator.storage_path is None
    assert orchestrator.model_type == "logistic_regression"
    assert orchestrator.model_config == {"activation_function": "sigmoid"}
    assert orchestrator.preprocessing_pipeline is not None
    assert orchestrator.model_wrapper is not None

    load_mock.assert_called_once_with(model_path)


def test_constructor_rejects_empty_model_path():
    with pytest.raises(ValueError, match="model_path is required"):
        PredictionOrchestrator(
            model_path="",
            storage_path=None,
        )


def test_constructor_rejects_missing_model_file(tmp_path):
    missing_path = tmp_path / "missing_model.joblib"

    with pytest.raises(FileNotFoundError, match="Model file not found"):
        PredictionOrchestrator(
            model_path=str(missing_path),
            storage_path=None,
        )


def test_load_model_routes_to_model_wrapper(orchestrator, model_path):
    orchestrator.model_wrapper.load = MagicMock()

    orchestrator._load_model(model_path)

    orchestrator.model_wrapper.load.assert_called_once_with(model_path)


def test_preprocess_prediction_data_routes_to_preprocessing_pipeline(
    orchestrator,
    input_data,
    prediction_df,
):
    orchestrator.preprocessing_pipeline.run_prediction_preprocessing = MagicMock(
        return_value=prediction_df
    )

    result = orchestrator._preprocess_prediction_data(
        input_data=input_data,
        customer_filename="customer_release.csv",
    )

    orchestrator.preprocessing_pipeline.run_prediction_preprocessing.assert_called_once_with(
        input_data=input_data,
        customer_filename="customer_release.csv",
    )

    pd.testing.assert_frame_equal(result, prediction_df)


def test_adapt_prediction_data_rejects_empty_dataframe(orchestrator):
    empty_df = pd.DataFrame()

    with pytest.raises(ValueError, match="Prediction data cannot be empty"):
        orchestrator._adapt_prediction_data(empty_df)


def test_adapt_prediction_data_routes_to_model_wrapper(
    orchestrator,
    prediction_df,
    X_prediction,
):
    orchestrator.model_wrapper.prepare_prediction_data = MagicMock(
        return_value=X_prediction
    )

    result = orchestrator._adapt_prediction_data(prediction_df)

    orchestrator.model_wrapper.prepare_prediction_data.assert_called_once_with(
        prediction_df
    )

    pd.testing.assert_frame_equal(result, X_prediction)


def test_predict_rejects_empty_X(orchestrator):
    empty_X = pd.DataFrame()

    with pytest.raises(ValueError, match="Prediction feature data cannot be empty"):
        orchestrator._predict(empty_X, threshold=0.5)


def test_predict_rejects_threshold_below_zero(orchestrator, X_prediction):
    with pytest.raises(ValueError, match="Threshold must be between 0 and 1"):
        orchestrator._predict(X_prediction, threshold=-0.1)


def test_predict_rejects_threshold_above_one(orchestrator, X_prediction):
    with pytest.raises(ValueError, match="Threshold must be between 0 and 1"):
        orchestrator._predict(X_prediction, threshold=1.1)


def test_predict_routes_to_model_wrapper(orchestrator, X_prediction):
    orchestrator.model_wrapper.predict = MagicMock(
        return_value=([1], [0.82], 0.6)
    )

    predictions, probabilities, threshold = orchestrator._predict(
        X_prediction=X_prediction,
        threshold=0.6,
    )

    orchestrator.model_wrapper.predict.assert_called_once_with(
        X=X_prediction,
        threshold=0.6,
    )

    assert predictions == [1]
    assert probabilities == [0.82]
    assert threshold == 0.6


def test_run_executes_full_prediction_workflow_from_raw_input(
    orchestrator,
    input_data,
    prediction_df,
    X_prediction,
):
    orchestrator._preprocess_prediction_data = MagicMock(return_value=prediction_df)
    orchestrator._adapt_prediction_data = MagicMock(return_value=X_prediction)
    orchestrator._predict = MagicMock(return_value=([1], [0.82], 0.6))

    result = orchestrator.run(
        input_data=input_data,
        customer_filename="customer_release.csv",
        threshold=0.6,
        preprocessed=False,
    )

    orchestrator._preprocess_prediction_data.assert_called_once_with(
        input_data=input_data,
        customer_filename="customer_release.csv",
    )

    orchestrator._adapt_prediction_data.assert_called_once_with(prediction_df)

    orchestrator._predict.assert_called_once_with(
        X_prediction=X_prediction,
        threshold=0.6,
    )

    assert result["model_path"] == orchestrator.model_path
    assert result["model_type"] == "logistic_regression"
    assert result["threshold"] == 0.6
    assert result["row_count"] == 1
    assert result["preprocessed"] is False
    assert result["predictions"] == [1]
    assert result["probabilities"] == [0.82]


def test_run_skips_preprocessing_when_input_is_preprocessed_dataframe(
    orchestrator,
    prediction_df,
    X_prediction,
):
    orchestrator._preprocess_prediction_data = MagicMock()
    orchestrator._adapt_prediction_data = MagicMock(return_value=X_prediction)
    orchestrator._predict = MagicMock(return_value=([0], [0.25], 0.5))

    result = orchestrator.run(
        input_data=prediction_df,
        threshold=0.5,
        preprocessed=True,
    )

    orchestrator._preprocess_prediction_data.assert_not_called()

    adapted_df = orchestrator._adapt_prediction_data.call_args.args[0]
    pd.testing.assert_frame_equal(adapted_df, prediction_df)

    orchestrator._predict.assert_called_once_with(
        X_prediction=X_prediction,
        threshold=0.5,
    )

    assert result["model_path"] == orchestrator.model_path
    assert result["model_type"] == "logistic_regression"
    assert result["threshold"] == 0.5
    assert result["row_count"] == 1
    assert result["preprocessed"] is True
    assert result["predictions"] == [0]
    assert result["probabilities"] == [0.25]


def test_run_accepts_preprocessed_dict_input(
    orchestrator,
    input_data,
    X_prediction,
):
    orchestrator._preprocess_prediction_data = MagicMock()
    orchestrator._adapt_prediction_data = MagicMock(return_value=X_prediction)
    orchestrator._predict = MagicMock(return_value=([1], [0.91], 0.4))

    result = orchestrator.run(
        input_data=input_data,
        threshold=0.4,
        preprocessed=True,
    )

    orchestrator._preprocess_prediction_data.assert_not_called()

    adapted_df = orchestrator._adapt_prediction_data.call_args.args[0]

    assert isinstance(adapted_df, pd.DataFrame)
    assert len(adapted_df) == 1
    assert adapted_df.iloc[0]["cc_num"] == input_data["cc_num"]

    orchestrator._predict.assert_called_once_with(
        X_prediction=X_prediction,
        threshold=0.4,
    )

    assert result["preprocessed"] is True
    assert result["row_count"] == 1
    assert result["predictions"] == [1]
    assert result["probabilities"] == [0.91]
    assert result["threshold"] == 0.4

def test_constructor_supports_gradient_boosting(model_path, monkeypatch):
    load_mock = MagicMock()
    monkeypatch.setattr(ModelWrapper, "load", load_mock)

    orchestrator = PredictionOrchestrator(
        model_path=model_path,
        storage_path=None,
        model_type="gradient_boosting",
        model_config={
            "max_iter": 5,
            "learning_rate": 0.1,
            "max_leaf_nodes": 7,
            "min_samples_leaf": 1,
            "random_state": 42,
        },
    )

    assert orchestrator.model_path == model_path
    assert orchestrator.storage_path is None
    assert orchestrator.model_type == "gradient_boosting"
    assert orchestrator.model_config == {
        "max_iter": 5,
        "learning_rate": 0.1,
        "max_leaf_nodes": 7,
        "min_samples_leaf": 1,
        "random_state": 42,
    }
    assert orchestrator.preprocessing_pipeline is not None
    assert orchestrator.model_wrapper is not None
    assert orchestrator.model_wrapper.model_type == "gradient_boosting"

    load_mock.assert_called_once_with(model_path)