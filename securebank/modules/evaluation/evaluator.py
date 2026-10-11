from __future__ import annotations

from typing import Any, Optional

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

class ModelEvaluator:
    """
    Computes classification metrics for SecureBank model outputs.

    This class does not train models, preprocess data, split data, or make predictions.
    It only evaluates predictions and probabilities against known labels.
    """

    def evaluate(
        self,
        y_true: pd.Series,
        predictions: list[int],
        probabilities: Optional[list[float]] = None,
        threshold: Optional[float]= None,
    ) -> dict[str, Any]:
        """
        Computes all supported evaluation metrics.

        Parameters:
            y_true (pd.Series): True fraud labels.
            predictions (list[int]): Predicted fraud labels.
            probabilities (list[float]  None): Fraud probabilities for class 1.
            threshold (float  None): Fraud probability threshold used for classification.

        Returns:
            dict[str, Any]: All supported evaluation metrics.
        """
        pass

    def accuracy(
        self,
        y_true: pd.Series,
        predictions: list[int],
    ) -> float:
        """
        Computes accuracy.
        """
        self._validate_label_inputs(y_true, predictions)

        return float(accuracy_score(y_true, predictions))

    def balanced_accuracy(
        self,
        y_true: pd.Series,
        predictions: list[int],
    ) -> float:
        """
        Computes balanced accuracy.
        """
        self._validate_label_inputs(y_true, predictions)

        return float(balanced_accuracy_score(y_true, predictions))

    def precision(
        self,
        y_true: pd.Series,
        predictions: list[int],
    ) -> float:
        """
        Computes precision for the fraud class.
        """
        self._validate_label_inputs(y_true, predictions)

        return float(
            precision_score(
                y_true,
                predictions,
                pos_label=1,
                zero_division=0,
            )
        )

    def recall(
        self,
        y_true: pd.Series,
        predictions: list[int],
    ) -> float:
        """
        Computes recall for the fraud class.
        """
        self._validate_label_inputs(y_true, predictions)

        return float(
            recall_score(
                y_true,
                predictions,
                pos_label=1,
                zero_division=0,
            )
        )

    def f1(
        self,
        y_true: pd.Series,
        predictions: list[int],
    ) -> float:
        """
        Computes F1 score for the fraud class.
        """
        self._validate_label_inputs(y_true, predictions)

        return float(
            f1_score(
                y_true,
                predictions,
                pos_label=1,
                zero_division=0,
            )
        )

    def roc_auc(
        self,
        y_true: pd.Series,
        probabilities: list[float],
    ) -> Optional[float]:
        """
        Computes ROC AUC using fraud probabilities.
        """
        self._validate_probability_inputs(y_true, probabilities)

        if y_true.nunique() < 2:
            return None

        return float(roc_auc_score(y_true, probabilities))

    def average_precision(
        self,
        y_true: pd.Series,
        probabilities: list[float],
    ) -> Optional[float]:
        """
        Computes average precision using fraud probabilities.
        """
        self._validate_probability_inputs(y_true, probabilities)

        if y_true.nunique() < 2:
            return None

        return float(average_precision_score(y_true, probabilities))

    def confusion_matrix_values(
        self,
        y_true: pd.Series,
        predictions: list[int],
    ) -> dict[str, Any]:
        """
        Computes the confusion matrix and named values:
        true_negative, false_positive, false_negative, true_positive.
        """
        self._validate_label_inputs(y_true, predictions)

        matrix = confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1],
        )

        true_negative, false_positive, false_negative, true_positive = matrix.ravel()

        return {
            "confusion_matrix": matrix.tolist(),
            "true_negative": int(true_negative),
            "false_positive": int(false_positive),
            "false_negative": int(false_negative),
            "true_positive": int(true_positive),
        }

    def false_positive_rate(
        self,
        y_true: pd.Series,
        predictions: list[int],
    ) -> float:
        """
        Computes false positive rate.
        """
        values = self.confusion_matrix_values(y_true, predictions)

        denominator = values["false_positive"] + values["true_negative"]

        if denominator == 0:
            return 0.0

        return float(values["false_positive"] / denominator)

    def false_negative_rate(
        self,
        y_true: pd.Series,
        predictions: list[int],
    ) -> float:
        """
        Computes false negative rate.
        """
        values = self.confusion_matrix_values(y_true, predictions)

        denominator = values["false_negative"] + values["true_positive"]

        if denominator == 0:
            return 0.0

        return float(values["false_negative"] / denominator)

    def specificity(
        self,
        y_true: pd.Series,
        predictions: list[int],
    ) -> float:
        """
        Computes specificity, also known as true negative rate.
        """
        values = self.confusion_matrix_values(y_true, predictions)

        denominator = values["true_negative"] + values["false_positive"]

        if denominator == 0:
            return 0.0

        return float(values["true_negative"] / denominator)

    def _validate_label_inputs(
        self,
        y_true: pd.Series,
        predictions: list[int],
    ) -> None:
        """
        Validates true labels and predicted labels.
        """
        if y_true.empty:
            raise ValueError("True labels cannot be empty")

        if not predictions:
            raise ValueError("Predictions cannot be empty")

        if len(y_true) != len(predictions):
            raise ValueError("True labels and predictions must have the same length")

        invalid_true_labels = set(y_true.dropna().unique()) - {0, 1}
        if invalid_true_labels:
            raise ValueError("True labels must only contain 0 and 1")

        invalid_predictions = set(predictions) - {0, 1}
        if invalid_predictions:
            raise ValueError("Predictions must only contain 0 and 1")

        if y_true.isna().any():
            raise ValueError("True labels cannot contain missing values")

    def _validate_probability_inputs(
        self,
        y_true: pd.Series,
        probabilities: list[float],
    ) -> None:
        """
        Validates true labels and predicted probabilities.
        """
        if y_true.empty:
            raise ValueError("True labels cannot be empty")

        if not probabilities:
            raise ValueError("Probabilities cannot be empty")

        if len(y_true) != len(probabilities):
            raise ValueError("True labels and probabilities must have the same length")

        invalid_true_labels = set(y_true.dropna().unique()) - {0, 1}
        if invalid_true_labels:
            raise ValueError("True labels must only contain 0 and 1")

        if y_true.isna().any():
            raise ValueError("True labels cannot contain missing values")

        invalid_probabilities = [
            probability
            for probability in probabilities
            if not isinstance(probability, (int, float))
            or probability < 0
            or probability > 1
        ]

        if invalid_probabilities:
            raise ValueError("Probabilities must be between 0 and 1")