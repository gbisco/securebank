from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from securebank.modules.log.logger import JsonLogger
from securebank.modules.predict.prediction_orchestrator import PredictionOrchestrator


class PredictionService:
    """
    App-level prediction service for the SecureBank Flask /predict endpoint.

    This service wraps PredictionOrchestrator without changing it. It handles:
    - reading default_model.json
    - loading the selected model path, model type, and threshold
    - passing transaction-only input into the prediction orchestrator
    - enriching the transaction with customer data through the preprocessing pipeline
    - converting numeric predictions into assignment-ready labels
    - logging request and prediction results
    """

    DEFAULT_CUSTOMER_FILE = "customer_release.csv"

    def __init__(
        self,
        data_sources_path: str,
        default_model_metadata_path: str,
        customer_filename: str = DEFAULT_CUSTOMER_FILE,
    ):
        self.data_sources_path = Path(data_sources_path)
        self.default_model_metadata_path = Path(default_model_metadata_path)
        self.customer_filename = customer_filename
        self.logger = JsonLogger()

    def predict_transaction(
        self,
        input_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Runs prediction for a single incoming transaction JSON payload.

        Returns a JSON-safe response containing at least:
        {
            "predict": "legitimate" , "fraudulent"
        }
        """
        request_data = {
            "input_data": input_data,
            "default_model_metadata_path": str(self.default_model_metadata_path),
            "customer_filename": self.customer_filename,
        }

        try:
            metadata = self._load_default_model_metadata()
            prediction_orchestrator = self._build_prediction_orchestrator(metadata)

            threshold = float(metadata["selected_threshold"])

            prediction_result = prediction_orchestrator.run(
                input_data=input_data,
                customer_filename=self.customer_filename,
                threshold=threshold,
                preprocessed=False,
            )

            predictions = prediction_result.get("predictions", [])
            probabilities = prediction_result.get("probabilities", [])

            if not predictions or not probabilities:
                raise ValueError("Prediction orchestrator returned no predictions")

            numeric_prediction = int(predictions[0])
            probability = float(probabilities[0])

            response_data = {
                "predict": self._convert_prediction_label(numeric_prediction),
                "prediction": numeric_prediction,
                "fraud_probability": probability,
                "threshold": threshold,
                "model_type": prediction_result["model_type"],
                "model_path": prediction_result["model_path"],
            }

            self.logger.write_log(
                endpoint="predict",
                request_data=request_data,
                response_data=response_data,
                status="success",
            )

            return response_data

        except Exception as error:
            self.logger.write_log(
                endpoint="predict",
                request_data=request_data,
                status="error",
                error=str(error),
            )
            raise

    def _load_default_model_metadata(self) -> dict[str, Any]:
        """
        Loads default_model.json so the service knows which model to use.
        """
        if not self.default_model_metadata_path.exists():
            raise FileNotFoundError(
                f"Default model metadata not found: {self.default_model_metadata_path}"
            )

        with open(self.default_model_metadata_path, "r") as file:
            metadata = json.load(file)

        required_keys = [
            "model_path",
            "model_type",
            "selected_threshold",
        ]

        missing_keys = [
            key for key in required_keys
            if key not in metadata
        ]

        if missing_keys:
            raise ValueError(
                f"Default model metadata is missing required keys: {missing_keys}"
            )

        return metadata

    def _build_prediction_orchestrator(
        self,
        metadata: dict[str, Any],
    ) -> PredictionOrchestrator:
        """
        Builds the lower-level prediction orchestrator from default model metadata.
        """
        return PredictionOrchestrator(
            model_path=metadata["model_path"],
            storage_path=str(self.data_sources_path),
            model_type=metadata["model_type"],
            model_config=metadata.get("model_config", {}),
        )

    def _convert_prediction_label(
        self,
        prediction: int,
    ) -> str:
        """
        Converts numeric model output into assignment-required label.
        """
        if int(prediction) == 1:
            return "fraudulent"

        return "legitimate"