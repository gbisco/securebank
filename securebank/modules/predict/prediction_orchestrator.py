from __future__ import annotations

from typing import Any, Optional, Union

import pandas as pd
from pathlib import Path
from securebank.modules.data.preprocessing import TransactionPreprocessingPipeline
from securebank.modules.models.model_wrapper import ModelWrapper


class PredictionOrchestrator:
    """
    Coordinates the SecureBank prediction workflow.

    This class owns the inference sequence:
    - prediction preprocessing
    - loading trained model state
    - model-specific feature adaptation
    - fraud prediction

    It does not train models, split datasets, or compute evaluation metrics.
    """

    def __init__(
        self,
        model_path: str,
        storage_path: Optional[str] = None,
        model_type: str = "logistic_regression",
        model_config: Optional[dict[str, Any]] = None,
    ):
        """
        Initializes the prediction orchestrator.

        Parameters:
            model_path (str): Saved trained model wrapper path to load.
            storage_path (Optional[str]=None): Optional path to data sources for customer enrichment.
            model_type (str): Model implementation to use before loading saved state.
            model_config (Optional[dict[str, Any]] = None): Optional model-specific config.
        """
        self.model_path = model_path
        self.storage_path = storage_path
        self.model_type = model_type
        self.model_config = model_config or {}

        self.preprocessing_pipeline = TransactionPreprocessingPipeline(
            storage_path=self.storage_path
        )

        self.model_wrapper = ModelWrapper(
            model_type=self.model_type,
            model_config=self.model_config,
        )

        self._load_model(self.model_path)

    def run(
        self,
        input_data: Union[dict[str, Any], pd.DataFrame],
        customer_filename: Optional[str]= None,
        threshold: float = 0.5,
        preprocessed: bool = False,
    ) -> dict[str, Any]:
        """
        Runs the full prediction workflow.

        Parameters:
            input_data: Union[dict[str, Any], pd.DataFrame]: Raw or preprocessed prediction input.
            customer_filename (Optional[str]= None): Optional customer CSV filename for enrichment.
            threshold (float): Fraud probability threshold used for classification.
            preprocessed (bool): If True, skips general prediction preprocessing.

        Returns:
            dict[str, Any]: Prediction output with labels, probabilities, and threshold.
        """
        if preprocessed:
            if isinstance(input_data, pd.DataFrame):
                prediction_df = input_data.copy()
            else:
                prediction_df = pd.DataFrame([input_data])
        else:
            prediction_df = self._preprocess_prediction_data(
                input_data=input_data,
                customer_filename=customer_filename,
            )

        X_prediction = self._adapt_prediction_data(prediction_df)

        predictions, probabilities, prediction_threshold = self._predict(
            X_prediction=X_prediction,
            threshold=threshold,
        )

        return {
            "model_path": self.model_path,
            "model_type": self.model_wrapper.model_type,
            "threshold": prediction_threshold,
            "row_count": len(prediction_df),
            "preprocessed": preprocessed,
            "predictions": predictions,
            "probabilities": probabilities,
        }

    def _preprocess_prediction_data(
        self,
        input_data: Union[dict[str, Any], pd.DataFrame],
        customer_filename: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Runs general preprocessing for prediction input.
        """
        return self.preprocessing_pipeline.run_prediction_preprocessing(
            input_data=input_data,
            customer_filename=customer_filename,
        )

    def _adapt_prediction_data(
        self,
        prediction_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Converts cleaned prediction data into model-ready X.
        """
        if prediction_df.empty:
            raise ValueError("Prediction data cannot be empty")

        return self.model_wrapper.prepare_prediction_data(prediction_df)

    def _predict(
        self,
        X_prediction: pd.DataFrame,
        threshold: float,
    ) -> tuple[list[int], list[float], float]:
        """
        Runs fraud prediction using model-ready prediction data.
        """
        if X_prediction.empty:
            raise ValueError("Prediction feature data cannot be empty")

        if threshold < 0 or threshold > 1:
            raise ValueError("Threshold must be between 0 and 1")

        return self.model_wrapper.predict(
            X=X_prediction,
            threshold=threshold,
        )

    def _load_model(self, model_path: str) -> None:
        """
        Loads trained model wrapper state.
        """
        if not model_path:
            raise ValueError("model_path is required for prediction")

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model_wrapper.load(model_path)