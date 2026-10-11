import pandas as pd
import pytest
from unittest.mock import MagicMock

from securebank.modules.train.training_orchestrator import TrainingOrchestrator


@pytest.fixture
def orchestrator(tmp_path):
    return TrainingOrchestrator(
        storage_path=str(tmp_path),
        model_type="logistic_regression",
        model_config={"activation_function": "sigmoid"},
        split_config={
            "strategy": "holdout",
            "validation_size": 0.1,
            "test_size": 0.2,
            "random_state": 42,
            "stratify_column": "is_fraud",
        },
    )


@pytest.fixture
def orchestrator_no_validation(tmp_path):
    return TrainingOrchestrator(
        storage_path=str(tmp_path),
        model_type="logistic_regression",
        model_config={"activation_function": "sigmoid"},
        split_config={
            "strategy": "holdout",
            "validation_size": 0,
            "test_size": 0.2,
            "random_state": 42,
            "stratify_column": "is_fraud",
        },
    )


@pytest.fixture
def cleaned_df():
    return pd.DataFrame({
        "amt": [10.0, 20.0, 30.0, 40.0, 50.0],
        "is_fraud": [0, 1, 0, 1, 0],
    })


@pytest.fixture
def train_df():
    return pd.DataFrame({
        "amt": [10.0, 20.0, 30.0],
        "is_fraud": [0, 1, 0],
    })


@pytest.fixture
def validation_df():
    return pd.DataFrame({
        "amt": [40.0],
        "is_fraud": [1],
    })


@pytest.fixture
def test_df():
    return pd.DataFrame({
        "amt": [50.0],
        "is_fraud": [0],
    })


@pytest.fixture
def X_train():
    return pd.DataFrame({
        "amt": [10.0, 20.0, 30.0],
    })


@pytest.fixture
def y_train():
    return pd.Series([0, 1, 0])


def test_constructor_sets_components(tmp_path):
    orchestrator = TrainingOrchestrator(
        storage_path=str(tmp_path),
        model_type="logistic_regression",
        model_config={"activation_function": "sigmoid"},
        split_config={"strategy": "holdout"},
    )

    assert orchestrator.storage_path == str(tmp_path)
    assert orchestrator.model_type == "logistic_regression"
    assert orchestrator.model_config == {"activation_function": "sigmoid"}
    assert orchestrator.split_config == {"strategy": "holdout"}
    assert orchestrator.preprocessing_pipeline is not None
    assert orchestrator.model_wrapper is not None
    assert orchestrator.splitter is not None


def test_preprocess_training_data_routes_to_preprocessing_pipeline(orchestrator, cleaned_df):
    orchestrator.preprocessing_pipeline.run_training_preprocessing = MagicMock(
        return_value=cleaned_df
    )

    result = orchestrator._preprocess_training_data(
        customer_filename="customers.csv",
        transaction_filename="transactions.parquet",
        fraud_filename="fraud.json",
    )

    orchestrator.preprocessing_pipeline.run_training_preprocessing.assert_called_once_with(
        customer_filename="customers.csv",
        transaction_filename="transactions.parquet",
        fraud_filename="fraud.json",
    )

    pd.testing.assert_frame_equal(result, cleaned_df)


def test_split_data_routes_to_splitter(
    orchestrator,
    cleaned_df,
    train_df,
    validation_df,
    test_df,
):
    orchestrator.splitter.split = MagicMock(
        return_value=(train_df, validation_df, test_df)
    )

    result = orchestrator._split_data(cleaned_df)

    orchestrator.splitter.split.assert_called_once_with(cleaned_df)

    pd.testing.assert_frame_equal(result[0], train_df)
    pd.testing.assert_frame_equal(result[1], validation_df)
    pd.testing.assert_frame_equal(result[2], test_df)


def test_adapt_training_data_routes_to_model_wrapper(
    orchestrator,
    train_df,
    X_train,
    y_train,
):
    orchestrator.model_wrapper.prepare_training_data = MagicMock(
        return_value=(X_train, y_train)
    )

    X_result, y_result = orchestrator._adapt_training_data(train_df)

    orchestrator.model_wrapper.prepare_training_data.assert_called_once_with(train_df)

    pd.testing.assert_frame_equal(X_result, X_train)
    pd.testing.assert_series_equal(y_result, y_train)


def test_train_model_routes_to_model_wrapper(orchestrator, X_train, y_train):
    orchestrator.model_wrapper.train = MagicMock()

    orchestrator._train_model(X_train, y_train)

    orchestrator.model_wrapper.train.assert_called_once_with(X_train, y_train)


def test_save_model_does_nothing_when_path_is_none(orchestrator):
    orchestrator.model_wrapper.save = MagicMock()

    orchestrator._save_model(None)

    orchestrator.model_wrapper.save.assert_not_called()


def test_save_model_routes_to_model_wrapper_when_path_provided(orchestrator):
    orchestrator.model_wrapper.save = MagicMock()

    orchestrator._save_model("model.joblib")

    orchestrator.model_wrapper.save.assert_called_once_with("model.joblib")


def test_run_executes_training_workflow(
    orchestrator,
    cleaned_df,
    train_df,
    validation_df,
    test_df,
    X_train,
    y_train,
):
    orchestrator._preprocess_training_data = MagicMock(return_value=cleaned_df)
    orchestrator._split_data = MagicMock(return_value=(train_df, validation_df, test_df))
    orchestrator._adapt_training_data = MagicMock(return_value=(X_train, y_train))
    orchestrator._train_model = MagicMock()
    orchestrator._save_model = MagicMock()

    result = orchestrator.run(
        customer_filename="customers.csv",
        transaction_filename="transactions.parquet",
        fraud_filename="fraud.json",
        model_save_path="model.joblib",
    )

    orchestrator._preprocess_training_data.assert_called_once_with(
        customer_filename="customers.csv",
        transaction_filename="transactions.parquet",
        fraud_filename="fraud.json",
    )
    orchestrator._split_data.assert_called_once_with(cleaned_df)
    orchestrator._adapt_training_data.assert_called_once_with(train_df)
    orchestrator._train_model.assert_called_once_with(
        X_train=X_train,
        y_train=y_train,
    )
    orchestrator._save_model.assert_called_once_with("model.joblib")

    assert result["trained"] is True
    assert result["model_type"] == "logistic_regression"
    assert result["model_path"] == "model.joblib"

    assert result["row_counts"] == {
        "cleaned": 5,
        "train": 3,
        "validation": 1,
        "test": 1,
    }

    pd.testing.assert_frame_equal(result["splits"]["train_df"], train_df)
    pd.testing.assert_frame_equal(result["splits"]["validation_df"], validation_df)
    pd.testing.assert_frame_equal(result["splits"]["test_df"], test_df)


def test_run_handles_empty_validation_split(
    orchestrator,
    cleaned_df,
    train_df,
    test_df,
    X_train,
    y_train,
):
    empty_validation_df = pd.DataFrame(columns=train_df.columns)

    orchestrator._preprocess_training_data = MagicMock(return_value=cleaned_df)
    orchestrator._split_data = MagicMock(
        return_value=(train_df, empty_validation_df, test_df)
    )
    orchestrator._adapt_training_data = MagicMock(return_value=(X_train, y_train))
    orchestrator._train_model = MagicMock()
    orchestrator._save_model = MagicMock()

    result = orchestrator.run(
        customer_filename="customers.csv",
        transaction_filename="transactions.parquet",
        fraud_filename="fraud.json",
        model_save_path=None,
    )

    orchestrator._adapt_training_data.assert_called_once_with(train_df)
    orchestrator._train_model.assert_called_once_with(
        X_train=X_train,
        y_train=y_train,
    )
    orchestrator._save_model.assert_called_once_with(None)

    assert result["trained"] is True
    assert result["model_path"] is None

    assert result["row_counts"] == {
        "cleaned": 5,
        "train": 3,
        "validation": 0,
        "test": 1,
    }

    pd.testing.assert_frame_equal(result["splits"]["train_df"], train_df)
    pd.testing.assert_frame_equal(result["splits"]["validation_df"], empty_validation_df)
    pd.testing.assert_frame_equal(result["splits"]["test_df"], test_df)


def test_run_supports_validation_size_zero(
    orchestrator_no_validation,
    cleaned_df,
    train_df,
    test_df,
    X_train,
    y_train,
):
    empty_validation_df = pd.DataFrame(columns=train_df.columns)

    orchestrator_no_validation._preprocess_training_data = MagicMock(
        return_value=cleaned_df
    )
    orchestrator_no_validation._split_data = MagicMock(
        return_value=(train_df, empty_validation_df, test_df)
    )
    orchestrator_no_validation._adapt_training_data = MagicMock(
        return_value=(X_train, y_train)
    )
    orchestrator_no_validation._train_model = MagicMock()
    orchestrator_no_validation._save_model = MagicMock()

    result = orchestrator_no_validation.run(
        customer_filename="customers.csv",
        transaction_filename="transactions.parquet",
        fraud_filename="fraud.json",
        model_save_path=None,
    )

    assert orchestrator_no_validation.split_config["validation_size"] == 0

    orchestrator_no_validation._adapt_training_data.assert_called_once_with(train_df)
    orchestrator_no_validation._train_model.assert_called_once_with(
        X_train=X_train,
        y_train=y_train,
    )
    orchestrator_no_validation._save_model.assert_called_once_with(None)

    assert result["trained"] is True
    assert result["model_path"] is None

    assert result["row_counts"]["validation"] == 0
    assert result["row_counts"]["test"] == 1

    pd.testing.assert_frame_equal(result["splits"]["train_df"], train_df)
    pd.testing.assert_frame_equal(result["splits"]["validation_df"], empty_validation_df)
    pd.testing.assert_frame_equal(result["splits"]["test_df"], test_df)

def test_constructor_supports_gradient_boosting(tmp_path):
    orchestrator = TrainingOrchestrator(
        storage_path=str(tmp_path),
        model_type="gradient_boosting",
        model_config={
            "max_iter": 5,
            "learning_rate": 0.1,
            "max_leaf_nodes": 7,
            "min_samples_leaf": 1,
            "random_state": 42,
        },
        split_config={"strategy": "holdout"},
    )

    assert orchestrator.storage_path == str(tmp_path)
    assert orchestrator.model_type == "gradient_boosting"
    assert orchestrator.model_config == {
        "max_iter": 5,
        "learning_rate": 0.1,
        "max_leaf_nodes": 7,
        "min_samples_leaf": 1,
        "random_state": 42,
    }
    assert orchestrator.split_config == {"strategy": "holdout"}
    assert orchestrator.preprocessing_pipeline is not None
    assert orchestrator.model_wrapper is not None
    assert orchestrator.splitter is not None
    assert orchestrator.model_wrapper.model_type == "gradient_boosting"