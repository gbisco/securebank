import pytest
import pandas as pd

from securebank.modules.evaluation.evaluator import ModelEvaluator


@pytest.fixture
def evaluator():
    return ModelEvaluator()


@pytest.fixture
def y_true():
    return pd.Series([0, 0, 1, 1])


@pytest.fixture
def predictions():
    return [0, 1, 1, 0]


@pytest.fixture
def probabilities():
    return [0.1, 0.7, 0.8, 0.3]


def test_accuracy(evaluator, y_true, predictions):
    result = evaluator.accuracy(y_true, predictions)

    assert result == 0.5


def test_balanced_accuracy(evaluator, y_true, predictions):
    result = evaluator.balanced_accuracy(y_true, predictions)

    assert result == 0.5


def test_precision(evaluator, y_true, predictions):
    result = evaluator.precision(y_true, predictions)

    assert result == 0.5


def test_recall(evaluator, y_true, predictions):
    result = evaluator.recall(y_true, predictions)

    assert result == 0.5


def test_f1(evaluator, y_true, predictions):
    result = evaluator.f1(y_true, predictions)

    assert result == 0.5


def test_roc_auc(evaluator, y_true, probabilities):
    result = evaluator.roc_auc(y_true, probabilities)

    assert result == 0.75


def test_average_precision(evaluator, y_true, probabilities):
    result = evaluator.average_precision(y_true, probabilities)

    assert result == pytest.approx(0.8333333333333333)


def test_confusion_matrix_values(evaluator, y_true, predictions):
    result = evaluator.confusion_matrix_values(y_true, predictions)

    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
    assert result["true_negative"] == 1
    assert result["false_positive"] == 1
    assert result["false_negative"] == 1
    assert result["true_positive"] == 1


def test_false_positive_rate(evaluator, y_true, predictions):
    result = evaluator.false_positive_rate(y_true, predictions)

    assert result == 0.5


def test_false_negative_rate(evaluator, y_true, predictions):
    result = evaluator.false_negative_rate(y_true, predictions)

    assert result == 0.5


def test_specificity(evaluator, y_true, predictions):
    result = evaluator.specificity(y_true, predictions)

    assert result == 0.5


def test_roc_auc_returns_none_when_only_one_class(evaluator):
    y_single_class = pd.Series([0, 0, 0])
    probabilities = [0.1, 0.2, 0.3]

    result = evaluator.roc_auc(y_single_class, probabilities)

    assert result is None


def test_average_precision_returns_none_when_only_one_class(evaluator):
    y_single_class = pd.Series([0, 0, 0])
    probabilities = [0.1, 0.2, 0.3]

    result = evaluator.average_precision(y_single_class, probabilities)

    assert result is None


def test_precision_returns_zero_when_no_positive_predictions(evaluator):
    y_true = pd.Series([0, 1, 1])
    predictions = [0, 0, 0]

    result = evaluator.precision(y_true, predictions)

    assert result == 0.0


def test_recall_returns_zero_when_no_positive_predictions(evaluator):
    y_true = pd.Series([0, 1, 1])
    predictions = [0, 0, 0]

    result = evaluator.recall(y_true, predictions)

    assert result == 0.0


def test_f1_returns_zero_when_no_positive_predictions(evaluator):
    y_true = pd.Series([0, 1, 1])
    predictions = [0, 0, 0]

    result = evaluator.f1(y_true, predictions)

    assert result == 0.0


def test_validate_label_inputs_rejects_empty_y_true(evaluator):
    y_empty = pd.Series(dtype=int)

    with pytest.raises(ValueError, match="True labels cannot be empty"):
        evaluator.accuracy(y_empty, [0])


def test_validate_label_inputs_rejects_empty_predictions(evaluator):
    y_true = pd.Series([0, 1])

    with pytest.raises(ValueError, match="Predictions cannot be empty"):
        evaluator.accuracy(y_true, [])


def test_validate_label_inputs_rejects_length_mismatch(evaluator):
    y_true = pd.Series([0, 1])

    with pytest.raises(ValueError, match="same length"):
        evaluator.accuracy(y_true, [0])


def test_validate_label_inputs_rejects_invalid_true_labels(evaluator):
    y_true = pd.Series([0, 2])

    with pytest.raises(ValueError, match="True labels must only contain 0 and 1"):
        evaluator.accuracy(y_true, [0, 1])


def test_validate_label_inputs_rejects_invalid_predictions(evaluator):
    y_true = pd.Series([0, 1])

    with pytest.raises(ValueError, match="Predictions must only contain 0 and 1"):
        evaluator.accuracy(y_true, [0, 2])


def test_validate_label_inputs_rejects_missing_true_labels(evaluator):
    y_true = pd.Series([0, None])

    with pytest.raises(ValueError, match="True labels cannot contain missing values"):
        evaluator.accuracy(y_true, [0, 1])


def test_validate_probability_inputs_rejects_empty_probabilities(evaluator):
    y_true = pd.Series([0, 1])

    with pytest.raises(ValueError, match="Probabilities cannot be empty"):
        evaluator.roc_auc(y_true, [])


def test_validate_probability_inputs_rejects_probability_length_mismatch(evaluator):
    y_true = pd.Series([0, 1])

    with pytest.raises(ValueError, match="same length"):
        evaluator.roc_auc(y_true, [0.2])


def test_validate_probability_inputs_rejects_invalid_probabilities(evaluator):
    y_true = pd.Series([0, 1])

    with pytest.raises(ValueError, match="Probabilities must be between 0 and 1"):
        evaluator.roc_auc(y_true, [0.2, 1.2])


def test_validate_probability_inputs_rejects_non_numeric_probabilities(evaluator):
    y_true = pd.Series([0, 1])

    with pytest.raises(ValueError, match="Probabilities must be between 0 and 1"):
        evaluator.roc_auc(y_true, [0.2, "bad"])


def test_false_positive_rate_returns_zero_when_no_negative_cases(evaluator):
    y_true = pd.Series([1, 1, 1])
    predictions = [1, 0, 1]

    result = evaluator.false_positive_rate(y_true, predictions)

    assert result == 0.0


def test_false_negative_rate_returns_zero_when_no_positive_cases(evaluator):
    y_true = pd.Series([0, 0, 0])
    predictions = [0, 1, 0]

    result = evaluator.false_negative_rate(y_true, predictions)

    assert result == 0.0


def test_specificity_returns_zero_when_no_negative_cases(evaluator):
    y_true = pd.Series([1, 1, 1])
    predictions = [1, 0, 1]

    result = evaluator.specificity(y_true, predictions)

    assert result == 0.0