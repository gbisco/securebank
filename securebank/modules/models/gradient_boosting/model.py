from __future__ import annotations

import pandas as pd
import joblib
from typing import Optional
from sklearn.ensemble import HistGradientBoostingClassifier


class HistGradientBoostingModel:
    """
    Histogram gradient boosting model wrapper for SecureBank fraud detection.

    This class is responsible for training and prediction orchestration only.
    Model-specific feature preparation is handled by the model wrapper adapter.

    The model parameters allow the wrapper to support tuning of the boosting
    process while keeping a consistent SecureBank model interface.
    """

    SUPPORTED_CLASS_WEIGHTS = [
        None,
        "balanced",
    ]

    def __init__(
        self,
        max_iter: int = 400,
        learning_rate: float = 0.02,
        max_leaf_nodes: int = 63,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 20,
        l2_regularization: float = 0.5,
        class_weight: Optional[str] = None,
        random_state: int = 42,
    ):
        """
        Initializes the histogram gradient boosting model wrapper.

        Parameters:
            max_iter (int): Maximum number of boosting iterations.
            learning_rate (float): Learning rate used for each boosting step.
            max_leaf_nodes (int): Maximum number of leaves per tree.
            max_depth (Optional[int] = None): Maximum depth of each tree.
            min_samples_leaf (int): Minimum number of samples per leaf.
            l2_regularization (float): L2 regularization strength.
            class_weight (Optional[str] = None): Class weighting strategy.
            random_state (int): Random state used for reproducible training.
        """
        if class_weight not in self.SUPPORTED_CLASS_WEIGHTS:
            raise ValueError(
                f"Class weight not supported: {class_weight}. "
                f"Supported values are: {self.SUPPORTED_CLASS_WEIGHTS}"
            )

        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.class_weight = class_weight
        self.random_state = random_state

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
            raise ValueError(
                "Training feature data and target data must have the same number of rows"
            )

        self.model = HistGradientBoostingClassifier(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_leaf_nodes=self.max_leaf_nodes,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            class_weight=self.class_weight,
            random_state=self.random_state,
        )

        self.model.fit(X, y)
        self.is_trained = True

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
            "max_iter": self.max_iter,
            "learning_rate": self.learning_rate,
            "max_leaf_nodes": self.max_leaf_nodes,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "l2_regularization": self.l2_regularization,
            "class_weight": self.class_weight,
            "random_state": self.random_state,
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

        self.max_iter = model_data["max_iter"]
        self.learning_rate = model_data["learning_rate"]
        self.max_leaf_nodes = model_data["max_leaf_nodes"]
        self.max_depth = model_data.get("max_depth")
        self.min_samples_leaf = model_data.get("min_samples_leaf", 20)
        self.l2_regularization = model_data["l2_regularization"]
        self.class_weight = model_data.get("class_weight")
        self.random_state = model_data["random_state"]
        self.model = model_data["model"]
        self.is_trained = model_data["is_trained"]