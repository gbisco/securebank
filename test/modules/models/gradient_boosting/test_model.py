import pytest
import pandas as pd

from securebank.modules.models.gradient_boosting.model import (
    HistGradientBoostingModel,
)


@pytest.fixture
def sample_X():
    return pd.DataFrame({
        "amt": [10.0, 500.0, 15.0, 800.0, 20.0, 900.0],
        "hour": [9, 23, 10, 1, 11, 2],
        "distance": [1.0, 50.0, 2.0, 80.0, 3.0, 90.0],
        "age": [35, 45, 30, 50, 33, 43],
    })


@pytest.fixture
def sample_y():
    return pd.Series([0, 1, 0, 1, 0, 1])


def test_constructor_sets_default_values():
    model = HistGradientBoostingModel()

    assert model.max_iter == 100
    assert model.learning_rate == 0.1
    assert model.max_leaf_nodes == 31
    assert model.max_depth is None
    assert model.min_samples_leaf == 20
    assert model.l2_regularization == 0.0
    assert model.class_weight is None
    assert model.random_state == 42
    assert model.model is None
    assert model.is_trained is False


def test_constructor_accepts_custom_hyperparameters():
    model = HistGradientBoostingModel(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=15,
        max_depth=5,
        min_samples_leaf=10,
        l2_regularization=0.1,
        class_weight="balanced",
        random_state=7,
    )

    assert model.max_iter == 200
    assert model.learning_rate == 0.05
    assert model.max_leaf_nodes == 15
    assert model.max_depth == 5
    assert model.min_samples_leaf == 10
    assert model.l2_regularization == 0.1
    assert model.class_weight == "balanced"
    assert model.random_state == 7


def test_constructor_rejects_invalid_class_weight():
    with pytest.raises(ValueError, match="Class weight not supported"):
        HistGradientBoostingModel(class_weight="invalid")


def test_train_sets_model_and_is_trained(sample_X, sample_y):
    model = HistGradientBoostingModel(
        max_iter=5,
        min_samples_leaf=1,
    )

    model.train(sample_X, sample_y)

    assert model.model is not None
    assert model.is_trained is True


def test_train_uses_hyperparameters(sample_X, sample_y):
    model = HistGradientBoostingModel(
        max_iter=5,
        learning_rate=0.05,
        max_leaf_nodes=7,
        max_depth=3,
        min_samples_leaf=1,
        l2_regularization=0.1,
        class_weight="balanced",
        random_state=7,
    )

    model.train(sample_X, sample_y)

    assert model.model.max_iter == 5
    assert model.model.learning_rate == 0.05
    assert model.model.max_leaf_nodes == 7
    assert model.model.max_depth == 3
    assert model.model.min_samples_leaf == 1
    assert model.model.l2_regularization == 0.1
    assert model.model.class_weight == "balanced"
    assert model.model.random_state == 7
    assert model.is_trained is True


def test_train_rejects_empty_X(sample_y):
    model = HistGradientBoostingModel()
    empty_X = pd.DataFrame()

    with pytest.raises(ValueError, match="Training feature data cannot be empty"):
        model.train(empty_X, sample_y)


def test_train_rejects_empty_y(sample_X):
    model = HistGradientBoostingModel()
    empty_y = pd.Series(dtype=int)

    with pytest.raises(ValueError, match="Training target data cannot be empty"):
        model.train(sample_X, empty_y)


def test_train_rejects_mismatched_X_and_y(sample_X):
    model = HistGradientBoostingModel()
    mismatched_y = pd.Series([0, 1])

    with pytest.raises(ValueError, match="same number of rows"):
        model.train(sample_X, mismatched_y)


def test_predict_rejects_untrained_model(sample_X):
    model = HistGradientBoostingModel()

    with pytest.raises(ValueError, match="Model must be trained before prediction"):
        model.predict(sample_X)


def test_predict_rejects_empty_X(sample_X, sample_y):
    model = HistGradientBoostingModel(
        max_iter=5,
        min_samples_leaf=1,
    )
    model.train(sample_X, sample_y)

    empty_X = pd.DataFrame()

    with pytest.raises(ValueError, match="Prediction feature data cannot be empty"):
        model.predict(empty_X)


def test_predict_rejects_threshold_below_zero(sample_X, sample_y):
    model = HistGradientBoostingModel(
        max_iter=5,
        min_samples_leaf=1,
    )
    model.train(sample_X, sample_y)

    with pytest.raises(ValueError, match="Threshold must be between 0 and 1"):
        model.predict(sample_X, threshold=-0.1)


def test_predict_rejects_threshold_above_one(sample_X, sample_y):
    model = HistGradientBoostingModel(
        max_iter=5,
        min_samples_leaf=1,
    )
    model.train(sample_X, sample_y)

    with pytest.raises(ValueError, match="Threshold must be between 0 and 1"):
        model.predict(sample_X, threshold=1.1)


def test_predict_returns_predictions_probabilities_and_threshold(sample_X, sample_y):
    model = HistGradientBoostingModel(
        max_iter=5,
        min_samples_leaf=1,
    )
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
    model = HistGradientBoostingModel(
        max_iter=5,
        min_samples_leaf=1,
    )
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
    model = HistGradientBoostingModel()
    model_path = tmp_path / "hgb_model.joblib"

    with pytest.raises(ValueError, match="Model must be trained before saving"):
        model.save(str(model_path))


def test_save_creates_file(sample_X, sample_y, tmp_path):
    model = HistGradientBoostingModel(
        max_iter=5,
        min_samples_leaf=1,
    )
    model.train(sample_X, sample_y)

    model_path = tmp_path / "hgb_model.joblib"
    model.save(str(model_path))

    assert model_path.exists()


def test_load_restores_trained_model(sample_X, sample_y, tmp_path):
    model = HistGradientBoostingModel(
        max_iter=5,
        learning_rate=0.05,
        max_leaf_nodes=7,
        max_depth=3,
        min_samples_leaf=1,
        l2_regularization=0.1,
        class_weight="balanced",
        random_state=7,
    )
    model.train(sample_X, sample_y)

    model_path = tmp_path / "hgb_model.joblib"
    model.save(str(model_path))

    loaded_model = HistGradientBoostingModel()
    loaded_model.load(str(model_path))

    assert loaded_model.max_iter == 5
    assert loaded_model.learning_rate == 0.05
    assert loaded_model.max_leaf_nodes == 7
    assert loaded_model.max_depth == 3
    assert loaded_model.min_samples_leaf == 1
    assert loaded_model.l2_regularization == 0.1
    assert loaded_model.class_weight == "balanced"
    assert loaded_model.random_state == 7
    assert loaded_model.model is not None
    assert loaded_model.is_trained is True

    predictions, probabilities, threshold = loaded_model.predict(sample_X)

    assert len(predictions) == len(sample_X)
    assert len(probabilities) == len(sample_X)
    assert threshold == 0.5