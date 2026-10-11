import pytest
import pandas as pd

from securebank.modules.models.model_wrapper import ModelWrapper
from securebank.modules.models.logistic_regression.adapter import LogisticRegressionAdapter
from securebank.modules.models.logistic_regression.model import LogisticRegressionModel
from securebank.modules.models.gradient_boosting.adapter import (
    HistGradientBoostingAdapter,
)
from securebank.modules.models.gradient_boosting.model import (
    HistGradientBoostingModel,
)


@pytest.fixture
def cleaned_training_df():
    return pd.DataFrame({
        "trans_num": ["t1", "t2", "t3", "t4"],
        "cc_num": [111, 222, 333, 444],
        "unix_time": [1000, 2000, 3000, 4000],
        "merchant": ["store_a", "store_b", "store_a", "store_c"],
        "category": ["grocery", "gas", "grocery", "shopping"],
        "amt": [10.0, 500.0, 15.0, 800.0],
        "merch_lat": [40.0, 41.0, 40.5, 42.0],
        "merch_long": [-75.0, -76.0, -75.5, -77.0],
        "first": ["A", "B", "C", "D"],
        "last": ["One", "Two", "Three", "Four"],
        "sex": ["F", "M", "F", "M"],
        "street": ["s1", "s2", "s3", "s4"],
        "city": ["c1", "c2", "c3", "c4"],
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


def test_constructor_defaults_to_logistic_regression():
    wrapper = ModelWrapper()

    assert wrapper.model_type == "logistic_regression"
    assert wrapper.model_config == {}
    assert isinstance(wrapper.adapter, LogisticRegressionAdapter)
    assert isinstance(wrapper.model, LogisticRegressionModel)
    assert wrapper.model.activation_function == "sigmoid"
    assert wrapper.model.class_weight is None


def test_constructor_accepts_model_config():
    wrapper = ModelWrapper(
        model_type="logistic_regression",
        model_config={"activation_function": "probit"},
    )

    assert wrapper.model_type == "logistic_regression"
    assert wrapper.model_config == {"activation_function": "probit"}
    assert wrapper.model.activation_function == "probit"
    assert wrapper.model.class_weight is None


def test_constructor_accepts_class_weight_model_config():
    wrapper = ModelWrapper(
        model_type="logistic_regression",
        model_config={
            "activation_function": "sigmoid",
            "class_weight": "balanced",
        },
    )

    assert wrapper.model_type == "logistic_regression"
    assert wrapper.model_config == {
        "activation_function": "sigmoid",
        "class_weight": "balanced",
    }
    assert wrapper.model.activation_function == "sigmoid"
    assert wrapper.model.class_weight == "balanced"


def test_constructor_accepts_gradient_boosting_model_type():
    wrapper = ModelWrapper(model_type="gradient_boosting")

    assert wrapper.model_type == "gradient_boosting"
    assert wrapper.model_config == {}
    assert isinstance(wrapper.adapter, HistGradientBoostingAdapter)
    assert isinstance(wrapper.model, HistGradientBoostingModel)

    assert wrapper.model.max_iter == 100
    assert wrapper.model.learning_rate == 0.1
    assert wrapper.model.max_leaf_nodes == 31
    assert wrapper.model.max_depth is None
    assert wrapper.model.min_samples_leaf == 20
    assert wrapper.model.l2_regularization == 0.0
    assert wrapper.model.class_weight is None
    assert wrapper.model.random_state == 42


def test_constructor_accepts_gradient_boosting_model_config():
    wrapper = ModelWrapper(
        model_type="gradient_boosting",
        model_config={
            "max_iter": 50,
            "learning_rate": 0.05,
            "max_leaf_nodes": 15,
            "max_depth": 3,
            "min_samples_leaf": 5,
            "l2_regularization": 0.1,
            "class_weight": "balanced",
            "random_state": 7,
        },
    )

    assert wrapper.model_type == "gradient_boosting"
    assert wrapper.model_config == {
        "max_iter": 50,
        "learning_rate": 0.05,
        "max_leaf_nodes": 15,
        "max_depth": 3,
        "min_samples_leaf": 5,
        "l2_regularization": 0.1,
        "class_weight": "balanced",
        "random_state": 7,
    }
    assert isinstance(wrapper.adapter, HistGradientBoostingAdapter)
    assert isinstance(wrapper.model, HistGradientBoostingModel)

    assert wrapper.model.max_iter == 50
    assert wrapper.model.learning_rate == 0.05
    assert wrapper.model.max_leaf_nodes == 15
    assert wrapper.model.max_depth == 3
    assert wrapper.model.min_samples_leaf == 5
    assert wrapper.model.l2_regularization == 0.1
    assert wrapper.model.class_weight == "balanced"
    assert wrapper.model.random_state == 7


def test_constructor_rejects_invalid_model_type():
    with pytest.raises(ValueError, match="Model type not supported"):
        ModelWrapper(model_type="invalid_model")


def test_constructor_rejects_invalid_class_weight():
    with pytest.raises(ValueError, match="Class weight not supported"):
        ModelWrapper(
            model_type="logistic_regression",
            model_config={"class_weight": "invalid"},
        )


def test_constructor_rejects_invalid_gradient_boosting_class_weight():
    with pytest.raises(ValueError, match="Class weight not supported"):
        ModelWrapper(
            model_type="gradient_boosting",
            model_config={"class_weight": "invalid"},
        )


def test_prepare_training_data_rejects_empty_dataframe():
    wrapper = ModelWrapper()
    empty_df = pd.DataFrame()

    with pytest.raises(ValueError, match="Training data cannot be empty"):
        wrapper.prepare_training_data(empty_df)


def test_prepare_training_data_returns_X_and_y(cleaned_training_df):
    wrapper = ModelWrapper()

    X, y = wrapper.prepare_training_data(cleaned_training_df)

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)

    assert len(X) == len(cleaned_training_df)
    assert len(y) == len(cleaned_training_df)

    assert "is_fraud" not in X.columns
    assert y.tolist() == [0, 1, 0, 1]


def test_prepare_prediction_data_rejects_empty_dataframe():
    wrapper = ModelWrapper()
    empty_df = pd.DataFrame()

    with pytest.raises(ValueError, match="Prediction data cannot be empty"):
        wrapper.prepare_prediction_data(empty_df)


def test_prepare_prediction_data_returns_model_ready_X(
    cleaned_training_df,
    cleaned_prediction_df,
):
    wrapper = ModelWrapper()

    X_train, y_train = wrapper.prepare_training_data(cleaned_training_df)
    X_prediction = wrapper.prepare_prediction_data(cleaned_prediction_df)

    assert isinstance(X_prediction, pd.DataFrame)
    assert len(X_prediction) == len(cleaned_prediction_df)

    assert X_prediction.columns.tolist() == X_train.columns.tolist()


def test_train_rejects_empty_X(cleaned_training_df):
    wrapper = ModelWrapper()
    X_train, y_train = wrapper.prepare_training_data(cleaned_training_df)

    empty_X = pd.DataFrame()

    with pytest.raises(ValueError, match="Training feature data cannot be empty"):
        wrapper.train(empty_X, y_train)


def test_train_rejects_empty_y(cleaned_training_df):
    wrapper = ModelWrapper()
    X_train, y_train = wrapper.prepare_training_data(cleaned_training_df)

    empty_y = pd.Series(dtype=int)

    with pytest.raises(ValueError, match="Training target data cannot be empty"):
        wrapper.train(X_train, empty_y)


def test_train_rejects_mismatched_X_and_y(cleaned_training_df):
    wrapper = ModelWrapper()
    X_train, y_train = wrapper.prepare_training_data(cleaned_training_df)

    mismatched_y = pd.Series([0, 1])

    with pytest.raises(ValueError, match="same number of rows"):
        wrapper.train(X_train, mismatched_y)


def test_train_routes_to_selected_model(cleaned_training_df):
    wrapper = ModelWrapper()

    X_train, y_train = wrapper.prepare_training_data(cleaned_training_df)
    wrapper.train(X_train, y_train)

    assert wrapper.model.is_trained is True
    assert wrapper.model.model is not None


def test_train_routes_class_weight_to_selected_model(cleaned_training_df):
    wrapper = ModelWrapper(
        model_type="logistic_regression",
        model_config={"class_weight": "balanced"},
    )

    X_train, y_train = wrapper.prepare_training_data(cleaned_training_df)
    wrapper.train(X_train, y_train)

    assert wrapper.model.is_trained is True
    assert wrapper.model.model is not None
    assert wrapper.model.class_weight == "balanced"
    assert wrapper.model.model.class_weight == "balanced"


def test_train_routes_to_gradient_boosting_model(cleaned_training_df):
    wrapper = ModelWrapper(
        model_type="gradient_boosting",
        model_config={
            "max_iter": 5,
            "min_samples_leaf": 1,
            "random_state": 42,
        },
    )

    X_train, y_train = wrapper.prepare_training_data(cleaned_training_df)
    wrapper.train(X_train, y_train)

    assert wrapper.model.is_trained is True
    assert wrapper.model.model is not None
    assert isinstance(wrapper.adapter, HistGradientBoostingAdapter)
    assert isinstance(wrapper.model, HistGradientBoostingModel)


def test_predict_rejects_empty_X(cleaned_training_df):
    wrapper = ModelWrapper()

    X_train, y_train = wrapper.prepare_training_data(cleaned_training_df)
    wrapper.train(X_train, y_train)

    empty_X = pd.DataFrame()

    with pytest.raises(ValueError, match="Prediction feature data cannot be empty"):
        wrapper.predict(empty_X)


def test_predict_rejects_threshold_below_zero(cleaned_training_df):
    wrapper = ModelWrapper()

    X_train, y_train = wrapper.prepare_training_data(cleaned_training_df)
    wrapper.train(X_train, y_train)

    with pytest.raises(ValueError, match="Threshold must be between 0 and 1"):
        wrapper.predict(X_train, threshold=-0.1)


def test_predict_rejects_threshold_above_one(cleaned_training_df):
    wrapper = ModelWrapper()

    X_train, y_train = wrapper.prepare_training_data(cleaned_training_df)
    wrapper.train(X_train, y_train)

    with pytest.raises(ValueError, match="Threshold must be between 0 and 1"):
        wrapper.predict(X_train, threshold=1.1)


def test_predict_returns_predictions_probabilities_and_threshold(cleaned_training_df):
    wrapper = ModelWrapper()

    X_train, y_train = wrapper.prepare_training_data(cleaned_training_df)
    wrapper.train(X_train, y_train)

    predictions, probabilities, threshold = wrapper.predict(X_train, threshold=0.5)

    assert isinstance(predictions, list)
    assert isinstance(probabilities, list)
    assert threshold == 0.5

    assert len(predictions) == len(X_train)
    assert len(probabilities) == len(X_train)

    assert all(prediction in [0, 1] for prediction in predictions)
    assert all(0 <= probability <= 1 for probability in probabilities)


def test_save_creates_file(cleaned_training_df, tmp_path):
    wrapper = ModelWrapper()

    X_train, y_train = wrapper.prepare_training_data(cleaned_training_df)
    wrapper.train(X_train, y_train)

    save_path = tmp_path / "model_wrapper.joblib"
    wrapper.save(str(save_path))

    assert save_path.exists()


def test_load_restores_wrapper_state(cleaned_training_df, cleaned_prediction_df, tmp_path):
    wrapper = ModelWrapper(
        model_type="logistic_regression",
        model_config={"activation_function": "sigmoid"},
    )

    X_train, y_train = wrapper.prepare_training_data(cleaned_training_df)
    wrapper.train(X_train, y_train)

    save_path = tmp_path / "model_wrapper.joblib"
    wrapper.save(str(save_path))

    loaded_wrapper = ModelWrapper()
    loaded_wrapper.load(str(save_path))

    assert loaded_wrapper.model_type == "logistic_regression"
    assert loaded_wrapper.model_config == {"activation_function": "sigmoid"}
    assert isinstance(loaded_wrapper.adapter, LogisticRegressionAdapter)
    assert isinstance(loaded_wrapper.model, LogisticRegressionModel)
    assert loaded_wrapper.model.is_trained is True

    X_prediction = loaded_wrapper.prepare_prediction_data(cleaned_prediction_df)
    predictions, probabilities, threshold = loaded_wrapper.predict(X_prediction)

    assert len(predictions) == len(cleaned_prediction_df)
    assert len(probabilities) == len(cleaned_prediction_df)
    assert threshold == 0.5


def test_load_restores_wrapper_state_with_class_weight(
    cleaned_training_df,
    cleaned_prediction_df,
    tmp_path,
):
    wrapper = ModelWrapper(
        model_type="logistic_regression",
        model_config={
            "activation_function": "sigmoid",
            "class_weight": "balanced",
        },
    )

    X_train, y_train = wrapper.prepare_training_data(cleaned_training_df)
    wrapper.train(X_train, y_train)

    save_path = tmp_path / "model_wrapper_balanced.joblib"
    wrapper.save(str(save_path))

    loaded_wrapper = ModelWrapper()
    loaded_wrapper.load(str(save_path))

    assert loaded_wrapper.model_type == "logistic_regression"
    assert loaded_wrapper.model_config == {
        "activation_function": "sigmoid",
        "class_weight": "balanced",
    }
    assert isinstance(loaded_wrapper.adapter, LogisticRegressionAdapter)
    assert isinstance(loaded_wrapper.model, LogisticRegressionModel)
    assert loaded_wrapper.model.class_weight == "balanced"
    assert loaded_wrapper.model.model.class_weight == "balanced"
    assert loaded_wrapper.model.is_trained is True

    X_prediction = loaded_wrapper.prepare_prediction_data(cleaned_prediction_df)
    predictions, probabilities, threshold = loaded_wrapper.predict(X_prediction)

    assert len(predictions) == len(cleaned_prediction_df)
    assert len(probabilities) == len(cleaned_prediction_df)
    assert threshold == 0.5


def test_load_restores_gradient_boosting_wrapper_state(
    cleaned_training_df,
    cleaned_prediction_df,
    tmp_path,
):
    wrapper = ModelWrapper(
        model_type="gradient_boosting",
        model_config={
            "max_iter": 5,
            "learning_rate": 0.05,
            "max_leaf_nodes": 7,
            "max_depth": 3,
            "min_samples_leaf": 1,
            "l2_regularization": 0.1,
            "class_weight": "balanced",
            "random_state": 7,
        },
    )

    X_train, y_train = wrapper.prepare_training_data(cleaned_training_df)
    wrapper.train(X_train, y_train)

    save_path = tmp_path / "gradient_boosting_wrapper.joblib"
    wrapper.save(str(save_path))

    loaded_wrapper = ModelWrapper()
    loaded_wrapper.load(str(save_path))

    assert loaded_wrapper.model_type == "gradient_boosting"
    assert loaded_wrapper.model_config == {
        "max_iter": 5,
        "learning_rate": 0.05,
        "max_leaf_nodes": 7,
        "max_depth": 3,
        "min_samples_leaf": 1,
        "l2_regularization": 0.1,
        "class_weight": "balanced",
        "random_state": 7,
    }
    assert isinstance(loaded_wrapper.adapter, HistGradientBoostingAdapter)
    assert isinstance(loaded_wrapper.model, HistGradientBoostingModel)

    assert loaded_wrapper.model.max_iter == 5
    assert loaded_wrapper.model.learning_rate == 0.05
    assert loaded_wrapper.model.max_leaf_nodes == 7
    assert loaded_wrapper.model.max_depth == 3
    assert loaded_wrapper.model.min_samples_leaf == 1
    assert loaded_wrapper.model.l2_regularization == 0.1
    assert loaded_wrapper.model.class_weight == "balanced"
    assert loaded_wrapper.model.random_state == 7
    assert loaded_wrapper.model.is_trained is True

    X_prediction = loaded_wrapper.prepare_prediction_data(cleaned_prediction_df)
    predictions, probabilities, threshold = loaded_wrapper.predict(X_prediction)

    assert len(predictions) == len(cleaned_prediction_df)
    assert len(probabilities) == len(cleaned_prediction_df)
    assert threshold == 0.5