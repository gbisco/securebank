from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from securebank.modules.data.preprocessing import TransactionPreprocessingPipeline
from securebank.modules.log.logger import JsonLogger


class DatasetService:
    """
    App-level dataset service for the SecureBank Flask /create_dataset endpoint.

    This service wraps TransactionPreprocessingPipeline without changing it.
    It handles:
    - running the training preprocessing pipeline
    - saving the cleaned dataset to storage/datasets
    - returning compact dataset metadata
    - logging success and error cases
    """

    DEFAULT_CUSTOMER_FILE = "customer_release.csv"
    DEFAULT_TRANSACTION_FILE = "transactions_release.parquet"
    DEFAULT_FRAUD_FILE = "fraud_release.json"

    def __init__(
        self,
        data_sources_path: str,
        dataset_storage_path: str,
    ):
        self.data_sources_path = Path(data_sources_path)
        self.dataset_storage_path = Path(dataset_storage_path)

        self.logger = JsonLogger()

        self.dataset_storage_path.mkdir(parents=True, exist_ok=True)

    def create_dataset(
        self,
        customer_filename: str = DEFAULT_CUSTOMER_FILE,
        transaction_filename: str = DEFAULT_TRANSACTION_FILE,
        fraud_filename: str = DEFAULT_FRAUD_FILE,
        partition_type: str = "full",
    ) -> dict[str, Any]:
        """
        Main public method used by the /create_dataset endpoint.

        This should:
        1. Run TransactionPreprocessingPipeline.run_training_preprocessing().
        2. Save the cleaned dataset as a CSV.
        3. Return dataset path and quality summary.
        4. Log success/error.
        """
        request_data = {
            "customer_filename": customer_filename,
            "transaction_filename": transaction_filename,
            "fraud_filename": fraud_filename,
            "partition_type": partition_type,
        }

        try:
            timestamp = self._timestamp()

            dataset_df = self._run_preprocessing_pipeline(
                customer_filename=customer_filename,
                transaction_filename=transaction_filename,
                fraud_filename=fraud_filename,
            )

            dataset_path = self._build_dataset_path(
                partition_type=partition_type,
                timestamp=timestamp,
            )

            dataset_df.to_csv(dataset_path, index=False)

            dataset_summary = self._summarize_dataset(dataset_df)

            response_data = {
                "status": "success",
                "dataset_path": str(dataset_path),
                "partition_type": partition_type,
                "created_at": timestamp,
                "dataset_summary": dataset_summary,
            }

            self.logger.write_log(
                endpoint="create_dataset",
                request_data=request_data,
                response_data=response_data,
                status="success",
            )

            return response_data

        except Exception as error:
            self.logger.write_log(
                endpoint="create_dataset",
                request_data=request_data,
                status="error",
                error=str(error),
            )
            raise

    def _run_preprocessing_pipeline(
        self,
        customer_filename: str,
        transaction_filename: str,
        fraud_filename: str,
    ):
        """
        Calls the existing TransactionPreprocessingPipeline.
        """
        pipeline = TransactionPreprocessingPipeline(
            storage_path=str(self.data_sources_path)
        )

        return pipeline.run_training_preprocessing(
            customer_filename=customer_filename,
            transaction_filename=transaction_filename,
            fraud_filename=fraud_filename,
        )

    def _build_dataset_path(
        self,
        partition_type: str,
        timestamp: str,
    ) -> Path:
        """
        Builds the assignment-required dataset output path.
        """
        safe_partition_type = (
            partition_type
            .strip()
            .lower()
            .replace(" ", "_")
        )

        return (
            self.dataset_storage_path
            / f"dataset_{safe_partition_type}_{timestamp}.csv"
        )

    def _summarize_dataset(
        self,
        dataset_df,
    ) -> dict[str, Any]:
        """
        Creates a compact dataset quality summary.
        """
        summary = {
            "row_count": int(dataset_df.shape[0]),
            "column_count": int(dataset_df.shape[1]),
            "total_missing_values": int(dataset_df.isna().sum().sum()),
            "columns": dataset_df.columns.tolist(),
        }

        if "is_fraud" in dataset_df.columns and not dataset_df.empty:
            fraud_counts = dataset_df["is_fraud"].value_counts(dropna=False).to_dict()

            summary["fraud_rate"] = float(dataset_df["is_fraud"].mean())
            summary["fraud_label_counts"] = {
                str(label): int(count)
                for label, count in fraud_counts.items()
            }

        if "trans_num" in dataset_df.columns:
            summary["unique_transactions"] = int(dataset_df["trans_num"].nunique())
            summary["duplicate_transactions"] = int(
                dataset_df["trans_num"].duplicated().sum()
            )

        if "cc_num" in dataset_df.columns:
            summary["unique_customers"] = int(dataset_df["cc_num"].nunique())

        return summary

    def _timestamp(self) -> str:
        """
        Creates a filesystem-safe timestamp.
        """
        return datetime.now().strftime("%Y%m%d_%H%M%S")