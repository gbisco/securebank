import pytest
import pandas as pd

from securebank.modules.models.logistic_regression.model import LogisticRegressionModel


@pytest.fixture
def sample_X():
    return pd.DataFrame({
        "amt": [10.0, 500.0, 15.0, 800.0],
        "hour": [9, 23, 10, 1],
        "distance": [1.0, 50.0, 2.0, 80.0],
    })


@pytest.fixture
def sample_y():
    return pd.Series([0, 1, 0, 1])


def test_constructor_defaults_to_sigmoid():
    model = LogisticRegressionModel()

    assert model.activation_function == "sigmoid"
    assert model.class_weight is None
    assert model.model is None
    assert model.is_trained is False


def test_constructor_accepts_supported_activation_function():
    model = LogisticRegressionModel(activation_function="sigmoid")

    assert model.activation_function == "sigmoid"


def test_constructor_accepts_supported_class_weight():
    model = LogisticRegressionModel(class_weight="balanced")

    assert model.class_weight == "balanced"


def test_constructor_rejects_invalid_activation_function():
    with pytest.raises(ValueError, match="Activation function not supported"):
        LogisticRegressionModel(activation_function="invalid")


def test_constructor_rejects_invalid_class_weight():
    with pytest.raises(ValueError, match="Class weight not supported"):
        LogisticRegressionModel(class_weight="invalid")


def test_train_sets_model_and_is_trained(sample_X, sample_y):
    model = LogisticRegressionModel()

    model.train(sample_X, sample_y)

    assert model.model is not None
    assert model.is_trained is True


def test_train_uses_class_weight(sample_X, sample_y):
    model = LogisticRegressionModel(class_weight="balanced")

    model.train(sample_X, sample_y)

    assert model.model is not None
    assert model.model.class_weight == "balanced"
    assert model.is_trained is True


def test_train_rejects_empty_X(sample_y):
    model = LogisticRegressionModel()
    empty_X = pd.DataFrame()

    with pytest.raises(ValueError, match="Training feature data cannot be empty"):
        model.train(empty_X, sample_y)


def test_train_rejects_empty_y(sample_X):
    model = LogisticRegressionModel()
    empty_y = pd.Series(dtype=int)

    with pytest.raises(ValueError, match="Training target data cannot be empty"):
        model.train(sample_X, empty_y)


def test_train_rejects_mismatched_X_and_y(sample_X):
    model = LogisticRegressionModel()
    mismatched_y = pd.Series([0, 1])

    with pytest.raises(ValueError, match="same number of rows"):
        model.train(sample_X, mismatched_y)


def test_train_probit_not_implemented(sample_X, sample_y):
    model = LogisticRegressionModel(activation_function="probit")

    with pytest.raises(NotImplementedError, match="Probit training is not implemented yet"):
        model.train(sample_X, sample_y)


def test_predict_rejects_untrained_model(sample_X):
    model = LogisticRegressionModel()

    with pytest.raises(ValueError, match="Model must be trained before prediction"):
        model.predict(sample_X)


def test_predict_rejects_empty_X(sample_X, sample_y):
    model = LogisticRegressionModel()
    model.train(sample_X, sample_y)

    empty_X = pd.DataFrame()

    with pytest.raises(ValueError, match="Prediction feature data cannot be empty"):
        model.predict(empty_X)


def test_predict_rejects_threshold_below_zero(sample_X, sample_y):
    model = LogisticRegressionModel()
    model.train(sample_X, sample_y)

    with pytest.raises(ValueError, match="Threshold must be between 0 and 1"):
        model.predict(sample_X, threshold=-0.1)


def test_predict_rejects_threshold_above_one(sample_X, sample_y):
    model = LogisticRegressionModel()
    model.train(sample_X, sample_y)

    with pytest.raises(ValueError, match="Threshold must be between 0 and 1"):
        model.predict(sample_X, threshold=1.1)


def test_predict_returns_predictions_probabilities_and_threshold(sample_X, sample_y):
    model = LogisticRegressionModel()
    model.train(sample_X, sample_y)

    predictions, probabilities, threshold = model.predict(sample_X, threshold=0.5)

    assert isinstance(predictions, list)
    assert isinstance(probabilities, list)
    assert threshold == 0.5

    assert len(predictions) == len(sample_X)
    assert len(probabilities) == len(sample_X)

    assert all(prediction in [0, 1] for prediction in predictions)
    assert all(0 <= probability <= 1 for probability in probabilities)


def test_predict_uses_custom_threshold(sample_X, sample_y):
    model = LogisticRegressionModel()
    model.train(sample_X, sample_y)

    predictions_high_threshold, probabilities, threshold = model.predict(
        sample_X,
        threshold=1.0,
    )

    assert threshold == 1.0
    assert predictions_high_threshold == [
        1 if probability >= 1.0 else 0
        for probability in probabilities
    ]


def test_save_rejects_untrained_model(tmp_path):
    model = LogisticRegressionModel()
    model_path = tmp_path / "lr_model.joblib"

    with pytest.raises(ValueError, match="Model must be trained before saving"):
        model.save(str(model_path))


def test_save_creates_file(sample_X, sample_y, tmp_path):
    model = LogisticRegressionModel()
    model.train(sample_X, sample_y)

    model_path = tmp_path / "lr_model.joblib"
    model.save(str(model_path))

    assert model_path.exists()


def test_load_restores_trained_model(sample_X, sample_y, tmp_path):
    model = LogisticRegressionModel()
    model.train(sample_X, sample_y)

    model_path = tmp_path / "lr_model.joblib"
    model.save(str(model_path))

    loaded_model = LogisticRegressionModel()
    loaded_model.load(str(model_path))

    assert loaded_model.activation_function == "sigmoid"
    assert loaded_model.class_weight is None
    assert loaded_model.model is not None
    assert loaded_model.is_trained is True

    predictions, probabilities, threshold = loaded_model.predict(sample_X)

    assert len(predictions) == len(sample_X)
    assert len(probabilities) == len(sample_X)
    assert threshold == 0.5


def test_load_restores_class_weight(sample_X, sample_y, tmp_path):
    model = LogisticRegressionModel(class_weight="balanced")
    model.train(sample_X, sample_y)

    model_path = tmp_path / "lr_model_balanced.joblib"
    model.save(str(model_path))

    loaded_model = LogisticRegressionModel()
    loaded_model.load(str(model_path))

    assert loaded_model.activation_function == "sigmoid"
    assert loaded_model.class_weight == "balanced"
    assert loaded_model.model is not None
    assert loaded_model.model.class_weight == "balanced"
    assert loaded_model.is_trained is True