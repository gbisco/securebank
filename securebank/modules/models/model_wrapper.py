
from __future__ import annotations
from typing import Optional

import pandas as pd

from securebank.modules.models.logistic_regression.adapter import (
    LogisticRegressionAdapter,
)
from securebank.modules.models.logistic_regression.model import (
    LogisticRegressionModel,
)

from securebank.modules.models.gradient_boosting.adapter import (
    HistGradientBoostingAdapter,
)
from securebank.modules.models.gradient_boosting.model import (
    HistGradientBoostingModel,
)

import joblib

class ModelWrapper:
    """
    Routes model-specific preparation, training, prediction, saving, and loading
    to the selected SecureBank model implementation.
    """

    SUPPORTED_MODELS = [
        "logistic_regression",
        "gradient_boosting",
    ]

    def __init__(
        self,
        model_type: str = "logistic_regression",
        model_config: Optional[dict] = None,
    ):
        """
        Initializes the model wrapper with the selected model type.

        Parameters:
            model_type (str): Model implementation to use.
            model_config (OPtional[dict] = None): Optional model-specific configuration.
        """
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Model type not supported: {model_type}. "
                f"Supported values are: {self.SUPPORTED_MODELS}"
            )

        self.model_type = model_type
        self.model_config = model_config or {}

        if self.model_type == "logistic_regression":
            activation_function = self.model_config.get(
                "activation_function",
                "sigmoid",
            )

            class_weight = self.model_config.get("class_weight")

            self.adapter = LogisticRegressionAdapter()
            self.model = LogisticRegressionModel(
                activation_function=activation_function,
                class_weight=class_weight,
            )

        elif self.model_type == "gradient_boosting":
            max_iter = self.model_config.get("max_iter", 400)
            learning_rate = self.model_config.get("learning_rate", 0.02)
            max_leaf_nodes = self.model_config.get("max_leaf_nodes", 63)
            max_depth = self.model_config.get("max_depth")
            min_samples_leaf = self.model_config.get("min_samples_leaf", 20)
            l2_regularization = self.model_config.get("l2_regularization", 0.5)
            class_weight = self.model_config.get("class_weight")
            random_state = self.model_config.get("random_state", 42)

            self.adapter = HistGradientBoostingAdapter()
            self.model = HistGradientBoostingModel(
                max_iter=max_iter,
                learning_rate=learning_rate,
                max_leaf_nodes=max_leaf_nodes,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                l2_regularization=l2_regularization,
                class_weight=class_weight,
                random_state=random_state,
            )

    def prepare_training_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """
        Converts cleaned training data into model-ready X and y.
        """
        if df.empty:
            raise ValueError("Training data cannot be empty")

        return self.adapter.prepare_training_data(df)

    def prepare_prediction_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts cleaned prediction data into model-ready X.
        """
        if df.empty:
            raise ValueError("Prediction data cannot be empty")

        return self.adapter.prepare_prediction_data(df)

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Trains the selected model using model-ready X and y.
        """
        if X.empty:
            raise ValueError("Training feature data cannot be empty")

        if y.empty:
            raise ValueError("Training target data cannot be empty")

        if len(X) != len(y):
            raise ValueError(
                "Training feature data and target data must have the same number of rows"
            )

        self.model.train(X, y)

    def predict(
        self,
        X: pd.DataFrame,
        threshold: float = 0.5,
    ) -> tuple[list[int], list[float], float]:
        """
        Predicts labels and probabilities using the selected model.
        """
        if X.empty:
            raise ValueError("Prediction feature data cannot be empty")

        if threshold < 0 or threshold > 1:
            raise ValueError("Threshold must be between 0 and 1")

        return self.model.predict(X, threshold=threshold)

    def save(self, path: str) -> None:
        """
        Saves the selected model state.
        """
        model_state = {
            "model_type": self.model_type,
            "model_config": self.model_config,
            "adapter": self.adapter,
            "model": self.model,
        }

        joblib.dump(model_state, path)

    def load(self, path: str) -> None:
        """
        Loads the selected model state.
        """
        model_state = joblib.load(path)

        self.model_type = model_state["model_type"]
        self.model_config = model_state["model_config"]
        self.adapter = model_state["adapter"]
        self.model = model_state["model"]
