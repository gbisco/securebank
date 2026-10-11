from __future__ import annotations

from typing import Any, Optional
import pandas as pd
from securebank.modules.data.preprocessing import TransactionPreprocessingPipeline
from securebank.modules.models.model_wrapper import ModelWrapper
from securebank.modules.train.splitter import DataSplitter


class TrainingOrchestrator:
    """
    Coordinates the SecureBank model training workflow.

    This class owns the training sequence:
    - general preprocessing
    - train/validation/test splitting
    - model-specific data preparation
    - model training
    - validation/test prediction
    - optional model saving

    It does not directly perform low-level data loading, feature engineering,
    model fitting, or metric calculation.
    """

    def __init__(
        self,
        storage_path: str,
        model_type: str = "logistic_regression",
        model_config: Optional[dict[str, Any]] = None,
        split_config: Optional[dict[str, Any]] = None,
    ):
        """
        Initializes the training orchestrator.

        Parameters:
            storage_path (str): Path to the raw SecureBank data source directory.
            model_type (str): Model implementation to use.
            model_config (Optional[dict[str, Any]] = None): Optional model-specific config.
            split_config (Optional[dict[str, Any]] = None): Optional splitter config.
        """
        self.storage_path = storage_path
        self.model_type = model_type
        self.model_config = model_config or {}
        self.split_config = split_config or {}

        self.preprocessing_pipeline = TransactionPreprocessingPipeline(
            storage_path=self.storage_path
        )

        self.model_wrapper = ModelWrapper(
            model_type=self.model_type,
            model_config=self.model_config,
        )

        self.splitter = DataSplitter(**self.split_config)

    def run(
        self,
        customer_filename: str,
        transaction_filename: str,
        fraud_filename: str,
        model_save_path: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Runs the training workflow.

        Parameters:
            customer_filename (str): Customer CSV filename.
            transaction_filename (str): Transaction Parquet filename.
            fraud_filename (str): Fraud JSON filename.
            model_save_path (Optional[str] = None): Optional path to save trained model state.

        Returns:
            dict[str, Any]: Training result metadata and preprocessed data splits.
        """
        cleaned_df = self._preprocess_training_data(
            customer_filename=customer_filename,
            transaction_filename=transaction_filename,
            fraud_filename=fraud_filename,
        )

        train_df, validation_df, test_df = self._split_data(cleaned_df)

        X_train, y_train = self._adapt_training_data(train_df)

        self._train_model(
            X_train=X_train,
            y_train=y_train,
        )

        self._save_model(model_save_path)

        return {
            "trained": True,
            "model_type": self.model_type,
            "model_path": model_save_path,
            "row_counts": {
                "cleaned": len(cleaned_df),
                "train": len(train_df),
                "validation": len(validation_df),
                "test": len(test_df),
            },
            "splits": {
                "train_df": train_df,
                "validation_df": validation_df,
                "test_df": test_df,
            },
        }


    def _preprocess_training_data(
        self,
        customer_filename: str,
        transaction_filename: str,
        fraud_filename: str,
    ) -> pd.DataFrame:
        """
        Runs general preprocessing for the raw training files.
        """
        return self.preprocessing_pipeline.run_training_preprocessing(
            customer_filename=customer_filename,
            transaction_filename=transaction_filename,
            fraud_filename=fraud_filename,
        )

    def _split_data(
        self,
        cleaned_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits cleaned data into train, validation, and test DataFrames.
        """
        return self.splitter.split(cleaned_df)

    def _adapt_training_data(
        self,
        train_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Converts cleaned training data into model-ready X and y.

        This is the only model-specific adaptation performed during training.
        Validation and test splits are returned untouched for downstream prediction
        and evaluation.
        """
        return self.model_wrapper.prepare_training_data(train_df)

    def _train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> None:
        """
        Trains the selected model.
        """
        self.model_wrapper.train(X_train, y_train)

    def _save_model(self, model_save_path: Optional[str] = None) -> None:
        """
        Saves the trained model when a save path is provided.
        """
        if model_save_path is None:
            return

        self.model_wrapper.save(model_save_path)
