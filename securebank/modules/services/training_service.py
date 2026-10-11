from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import json
from datetime import datetime

from securebank.modules.predict.prediction_orchestrator import PredictionOrchestrator
from securebank.modules.train.training_orchestrator import TrainingOrchestrator
from securebank.modules.evaluation.evaluator import ModelEvaluator
from securebank.modules.log.logger import JsonLogger

class TrainingService:
    """
    App-level training service for the SecureBank Flask /train_model endpoint.

    This service should wrap the existing TrainingOrchestrator without changing it.
    It will eventually handle:
    - calling the training orchestrator
    - evaluating validation thresholds
    - selecting the best threshold
    - evaluating final test metrics
    - saving model metadata
    - updating default_model.json
    """

    DEFAULT_THRESHOLDS = [
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.875,
        0.90,
        0.95,
        0.99,
    ]

    DEFAULT_CUSTOMER_FILE = "customer_release.csv"
    DEFAULT_TRANSACTION_FILE = "transactions_release.parquet"
    DEFAULT_FRAUD_FILE = "fraud_release.json"

    def __init__(
        self,
        data_sources_path: str,
        model_storage_path: str,
        default_model_metadata_path: str,
        model_type: str = "gradient_boosting",
        model_config: Optional[dict[str, Any]] = None,
    ):
        self.data_sources_path = Path(data_sources_path)
        self.model_storage_path = Path(model_storage_path)
        self.default_model_metadata_path = Path(default_model_metadata_path)

        self.model_type = model_type
        self.model_config = model_config or {}

        self.evaluator = ModelEvaluator()
        self.logger = JsonLogger()

        self.model_storage_path.mkdir(parents=True, exist_ok=True)
        self.default_model_metadata_path.parent.mkdir(parents=True, exist_ok=True)

    def train_and_select_model(
        self,
        customer_filename: str = DEFAULT_CUSTOMER_FILE,
        transaction_filename: str = DEFAULT_TRANSACTION_FILE,
        fraud_filename: str = DEFAULT_FRAUD_FILE,
        split_strategy: str = "holdout",
        thresholds: Optional[list[float]] = None,
        min_precision: float = 0.70,
        min_recall: float = 0.70,
        register_as_default: bool = True,
    ) -> dict[str, Any]:
        """
        Main public method used by the /train_model endpoint.

        This should eventually:
        1. Run TrainingOrchestrator.
        2. Evaluate thresholds on validation data.
        3. Select the best threshold.
        4. Evaluate the selected threshold on test data.
        5. Save model metadata.
        6. Optionally update default_model.json.
        7. Return a compact JSON-safe response.
        """
        thresholds = thresholds or self.DEFAULT_THRESHOLDS

        request_data = {
            "customer_filename": customer_filename,
            "transaction_filename": transaction_filename,
            "fraud_filename": fraud_filename,
            "split_strategy": split_strategy,
            "thresholds": thresholds,
            "min_precision": min_precision,
            "min_recall": min_recall,
            "register_as_default": register_as_default,
            "model_type": self.model_type,
            "model_config": self.model_config,
        }

        try:
            timestamp = self._timestamp()
            model_path = self._build_model_path(timestamp)

            training_result = self._run_training_orchestrator(
                model_path=model_path,
                customer_filename=customer_filename,
                transaction_filename=transaction_filename,
                fraud_filename=fraud_filename,
                split_strategy=split_strategy,
            )

            splits = training_result.get("splits")

            if not splits:
                raise ValueError("Training result did not include dataset splits")

            validation_df = splits.get("validation_df")
            test_df = splits.get("test_df")

            if validation_df is None:
                raise ValueError("Training result did not include validation_df")

            if test_df is None:
                raise ValueError("Training result did not include test_df")

            validation_results = self._evaluate_thresholds(
                model_path=str(model_path),
                evaluation_df=validation_df,
                thresholds=thresholds,
            )

            selected_validation_result = self._select_best_threshold(
                threshold_results=validation_results,
                min_precision=min_precision,
                min_recall=min_recall,
            )

            selected_threshold = selected_validation_result["threshold"]

            test_result = self._evaluate_single_threshold(
                model_path=str(model_path),
                evaluation_df=test_df,
                threshold=selected_threshold,
            )

            metadata = {
                "model_path": str(model_path),
                "model_type": self.model_type,
                "model_config": self.model_config,
                "created_at": timestamp,
                "split_strategy": split_strategy,
                "thresholds": thresholds,
                "selected_threshold": selected_threshold,
                "min_precision": min_precision,
                "min_recall": min_recall,
                "selection_rule": selected_validation_result.get("selection_rule"),
                "selected_validation_result": selected_validation_result,
                "validation_results": validation_results,
                "test_result": test_result,
                "training_summary": self._compact_training_summary(training_result),
            }

            metadata_path = self._save_model_metadata(
                metadata=metadata,
                timestamp=timestamp,
            )

            if register_as_default:
                self._save_default_model_metadata(metadata)

            response_data = {
                "status": "success",
                "model_path": str(model_path),
                "metadata_path": str(metadata_path),
                "default_model_updated": register_as_default,
                "selected_threshold": selected_threshold,
                "selected_validation_result": selected_validation_result,
                "test_result": test_result,
            }

            self.logger.write_log(
                endpoint="train_model",
                request_data=request_data,
                response_data=response_data,
                status="success",
            )

            return response_data

        except Exception as error:
            self.logger.write_log(
                endpoint="train_model",
                request_data=request_data,
                status="error",
                error=str(error),
            )
            raise
        
    def _run_training_orchestrator(
        self,
        model_path: Path,
        customer_filename: str,
        transaction_filename: str,
        fraud_filename: str,
        split_strategy: str,
    ) -> dict[str, Any]:
        """
        Calls the existing TrainingOrchestrator.

        This method should stay thin so we do not disturb the working training flow.
        """
        split_config = {
            "strategy": split_strategy,
            "validation_size": 0.1,
            "test_size": 0.2,
            "random_state": 42,
            "stratify_column": "is_fraud",
        }

        orchestrator = TrainingOrchestrator(
            storage_path=str(self.data_sources_path),
            model_type=self.model_type,
            model_config=self.model_config,
            split_config=split_config,
        )

        return orchestrator.run(
            customer_filename=customer_filename,
            transaction_filename=transaction_filename,
            fraud_filename=fraud_filename,
            model_save_path=str(model_path),
        )

    def _evaluate_thresholds(
        self,
        model_path: str,
        evaluation_df,
        thresholds: list[float],
    ) -> list[dict[str, Any]]:
        """
        Evaluates the saved model across candidate thresholds.
        """
        threshold_results = []

        for threshold in thresholds:
            result = self._evaluate_single_threshold(
                model_path=model_path,
                evaluation_df=evaluation_df,
                threshold=threshold,
            )

            threshold_results.append(result)

        return threshold_results

    def _evaluate_single_threshold(
        self,
        model_path: str,
        evaluation_df,
        threshold: float,
    ) -> dict[str, Any]:
        """
        Evaluates one threshold on a preprocessed validation or test dataframe.
        """
        prediction_orchestrator = PredictionOrchestrator(
            model_path=model_path,
            storage_path=str(self.data_sources_path),
            model_type=self.model_type,
            model_config=self.model_config,
        )

        prediction_result = prediction_orchestrator.run(
            input_data=evaluation_df,
            threshold=threshold,
            preprocessed=True,
        )

        y_true = evaluation_df["is_fraud"]
        predictions = prediction_result["predictions"]
        probabilities = prediction_result["probabilities"]

        confusion_values = self.evaluator.confusion_matrix_values(
            y_true=y_true,
            predictions=predictions,
        )

        return {
            "threshold": threshold,
            "accuracy": self.evaluator.accuracy(y_true, predictions),
            "balanced_accuracy": self.evaluator.balanced_accuracy(y_true, predictions),
            "precision": self.evaluator.precision(y_true, predictions),
            "recall": self.evaluator.recall(y_true, predictions),
            "f1": self.evaluator.f1(y_true, predictions),
            "roc_auc": self.evaluator.roc_auc(y_true, probabilities),
            "average_precision": self.evaluator.average_precision(y_true, probabilities),
            "true_positive": confusion_values["true_positive"],
            "false_positive": confusion_values["false_positive"],
            "false_negative": confusion_values["false_negative"],
            "true_negative": confusion_values["true_negative"],
        }

    def _select_best_threshold(
        self,
        threshold_results: list[dict[str, Any]],
        min_precision: float,
        min_recall: float,
    ) -> dict[str, Any]:
        """
        Selects the best validation result.

        Intended rule:
        highest F1 among thresholds where precision >= min_precision
        and recall >= min_recall. If none qualify, fall back to highest F1.
        """
        if not threshold_results:
            raise ValueError("threshold_results cannot be empty")

        eligible_results = [
            result
            for result in threshold_results
            if (
                result.get("precision", 0) >= min_precision
                and result.get("recall", 0) >= min_recall
            )
        ]

        candidates = eligible_results if eligible_results else threshold_results

        best_result = max(
            candidates,
            key=lambda result: (
                result.get("f1", 0),
                result.get("recall", 0),
                result.get("precision", 0),
                result.get("balanced_accuracy", 0),
            ),
        )

        return {
            **best_result,
            "met_minimum_precision_recall": bool(eligible_results),
            "selection_rule": (
                "highest_f1_with_min_precision_recall"
                if eligible_results
                else "fallback_highest_f1"
            ),
        }

    def _save_model_metadata(
        self,
        metadata: dict[str, Any],
        timestamp: str,
    ) -> Path:
        """
        Saves versioned model metadata beside the model.
        """
        metadata_path = (
            self.model_storage_path
            / f"model_{self.model_type}_{timestamp}_metadata.json"
        )

        self._write_json(
            path=metadata_path,
            data=metadata,
        )

        return metadata_path

    def _save_default_model_metadata(
        self,
        metadata: dict[str, Any],
    ) -> None:
        """
        Updates default_model.json so /predict knows which model and threshold to use.
        """
        default_metadata = {
            "model_path": metadata["model_path"],
            "model_type": metadata["model_type"],
            "model_config": metadata.get("model_config", {}),
            "selected_threshold": metadata["selected_threshold"],
            "created_at": metadata["created_at"],
            "selection_rule": metadata.get("selection_rule"),
            "selected_validation_result": metadata.get("selected_validation_result"),
            "test_result": metadata.get("test_result"),
        }

        self._write_json(
            path=self.default_model_metadata_path,
            data=default_metadata,
        )

    def _compact_training_summary(
        self,
        training_result: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Removes large DataFrames from the training result before returning/saving metadata.
        """
        compact_summary = {}

        for key, value in training_result.items():
            if key == "splits":
                continue

            compact_summary[key] = value

        if "splits" in training_result:
            compact_summary["split_shapes"] = {}

            for split_name, split_df in training_result["splits"].items():
                if hasattr(split_df, "shape"):
                    compact_summary["split_shapes"][split_name] = {
                        "rows": int(split_df.shape[0]),
                        "columns": int(split_df.shape[1]),
                    }

        return compact_summary
    
        
    def _build_model_path(
        self,
        timestamp: str,
    ) -> Path:
        """
        Builds the versioned model path.
        """
        return self.model_storage_path / f"model_{self.model_type}_{timestamp}.joblib"


    def _timestamp(self) -> str:
        """
        Creates a filesystem-safe timestamp.
        """
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _write_json(
        self,
        path: Path,
        data: dict[str, Any],
    ) -> None:
        """
        Writes JSON data to disk.
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as file:
            json.dump(data, file, indent=2, default=str)