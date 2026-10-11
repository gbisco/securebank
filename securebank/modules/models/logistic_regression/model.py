from __future__ import annotations

import pandas as pd
import joblib
from typing import Optional
from sklearn.linear_model import LogisticRegression
from securebank.modules.models.logistic_regression.adapter import (
    LogisticRegressionAdapter,
)


class LogisticRegressionModel:
    """
    Logistic regression-style model wrapper for SecureBank fraud detection.

    This class is responsible for training and prediction orchestration only.
    Model-specific feature preparation is delegated to LogisticRegressionAdapter.

    The activation_function parameter allows the wrapper to support different
    binary probability transformations later, such as sigmoid or probit.
    """

    SUPPORTED_ACTIVATION_FUNCTIONS = [
        "sigmoid",
        "probit",
    ]

    SUPPORTED_CLASS_WEIGHTS = [
        None,
        "balanced",
    ]

    def __init__(
        self,
        activation_function: str = "sigmoid",
        class_weight: Optional[str] = None,
    ):
        """
        Initializes the logistic regression model wrapper.

        Parameters:
            activation_function (str): Probability transformation to use.
                Supported values are "sigmoid" and "probit".
        """
        if activation_function not in self.SUPPORTED_ACTIVATION_FUNCTIONS:
            raise ValueError(
                f"Activation function not supported: {activation_function}. "
                f"Supported values are: {self.SUPPORTED_ACTIVATION_FUNCTIONS}"
            )

        if class_weight not in self.SUPPORTED_CLASS_WEIGHTS:
            raise ValueError(
                f"Class weight not supported: {class_weight}. "
                f"Supported values are: {self.SUPPORTED_CLASS_WEIGHTS}"
            )

        self.activation_function = activation_function
        self.class_weight = class_weight
        self.adapter = LogisticRegressionAdapter()
        self.model = None
        self.is_trained = False


    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Trains the model using model-ready feature data and target labels.

        Parameters:
            X (pd.DataFrame): Model-ready feature matrix.
            y (pd.Series): Target labels.
        """
        if X.empty:
            raise ValueError("Training feature data cannot be empty")

        if y.empty:
            raise ValueError("Training target data cannot be empty")

        if len(X) != len(y):
            raise ValueError("Training feature data and target data must have the same number of rows")

        if self.activation_function == "sigmoid":
            self.model = LogisticRegression(
                max_iter=1000,
                class_weight=self.class_weight,
            )

        elif self.activation_function == "probit":
            raise NotImplementedError("Probit training is not implemented yet.")

        else:
            raise ValueError(
                f"Activation function not supported: {self.activation_function}"
            )

        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> list[int]:
        """
        Predicts fraud labels using model-ready feature data and a custom threshold.

        Parameters:
            X (pd.DataFrame): Model-ready feature matrix.
            threshold (float): Fraud probability cutoff used to classify fraud.

        Returns:
            list[int]: Predicted fraud labels.
        """
        if threshold < 0 or threshold > 1:
            raise ValueError("Threshold must be between 0 and 1")

        fraud_probabilities = self.predict_proba(X)

        predictions = [
            1 if probability >= threshold else 0
            for probability in fraud_probabilities
        ]

        return predictions

    def predict(
        self,
        X: pd.DataFrame,
        threshold: float = 0.5,
    ) -> tuple[list[int], list[float], float]:
        """
        Predicts fraud labels and fraud probabilities using model-ready feature data.

        Parameters:
            X (pd.DataFrame): Model-ready feature matrix.
            threshold (float): Fraud probability cutoff used to classify fraud.

        Returns:
            tuple[list[int], list[float], float]:
                predictions, fraud_probabilities, threshold
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        if self.model is None:
            raise ValueError("Model has not been initialized")

        if X.empty:
            raise ValueError("Prediction feature data cannot be empty")

        if threshold < 0 or threshold > 1:
            raise ValueError("Threshold must be between 0 and 1")

        fraud_probabilities = self.model.predict_proba(X)[:, 1].tolist()

        predictions = [
            1 if probability >= threshold else 0
            for probability in fraud_probabilities
        ]

        return predictions, fraud_probabilities, threshold

    def save(self, path: str) -> None:
        """
        Saves the trained model object to disk.

        Parameters:
            path (str): File path where the model should be saved.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        if self.model is None:
            raise ValueError("Model has not been initialized")

        model_data = {
            "activation_function": self.activation_function,
            "class_weight": self.class_weight,
            "model": self.model,
            "is_trained": self.is_trained,
        }

        joblib.dump(model_data, path)
    
    def load(self, path: str) -> None:
        """
        Loads a trained model object from disk.

        Parameters:
            path (str): File path where the model should be loaded from.
        """
        model_data = joblib.load(path)

        self.activation_function = model_data["activation_function"]
        self.class_weight = model_data.get("class_weight")
        self.model = model_data["model"]
        self.is_trained = model_data["is_trained"]