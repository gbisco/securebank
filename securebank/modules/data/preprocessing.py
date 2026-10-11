from typing import Any, Optional, Union

import pandas as pd

from securebank.modules.data.raw_data_handler import RawDataHandlerV2
from securebank.modules.data.schemas import (
    TrainingTransactionSchema,
    PredictionTransactionSchema,
)
from securebank.modules.log.logger import JsonLogger


class TransactionPreprocessingPipeline:
    """
    Orchestrates SecureBank preprocessing for training and prediction workflows.

    This class coordinates raw data loading, raw frame construction,
    missing-value handling, schema validation, and date feature creation.

    It does not perform model-specific feature engineering, training, or prediction.
    """

    def __init__(self, storage_path: Optional[str] = None):
        self.raw_data_handler = RawDataHandlerV2(storage_path=storage_path)
        self.logger = JsonLogger()

    def run_training_preprocessing(
        self,
        customer_filename: str,
        transaction_filename: str,
        fraud_filename: str,
    ) -> pd.DataFrame:
        """
        Runs preprocessing for training data.
        """
        request_data = {
            "customer_filename": customer_filename,
            "transaction_filename": transaction_filename,
            "fraud_filename": fraud_filename,
        }

        try:
            customer_df = self.raw_data_handler.load_customer_data(customer_filename)
            transaction_df = self.raw_data_handler.load_transaction_data(transaction_filename)
            fraud_df = self.raw_data_handler.load_fraud_data(fraud_filename)

            raw_summary = {
                "customer": self._summarize_dataframe(
                    customer_df,
                    important_columns=["cc_num", "dob", "sex", "state"],
                ),
                "transaction": self._summarize_dataframe(
                    transaction_df,
                    important_columns=[
                        "trans_date_trans_time",
                        "cc_num",
                        "unix_time",
                        "merchant",
                        "category",
                        "amt",
                        "merch_lat",
                        "merch_long",
                    ],
                ),
                "fraud": self._summarize_dataframe(
                    fraud_df,
                    important_columns=["trans_id", "is_fraud"],
                ),
            }

            training_df = self.raw_data_handler.build_training_frame(
                customer_df=customer_df,
                transaction_df=transaction_df,
                fraud_df=fraud_df,
            )

            merged_summary = self._summarize_dataframe(
                training_df,
                important_columns=[
                    "trans_num",
                    "trans_date_trans_time",
                    "cc_num",
                    "unix_time",
                    "merchant",
                    "category",
                    "amt",
                    "dob",
                    "is_fraud",
                ],
            )

            training_df = self.raw_data_handler.fill_missing_values(
                training_df,
                fill_values=TrainingTransactionSchema.fill_values,
            )

            filled_summary = self._summarize_dataframe(
                training_df,
                important_columns=[
                    "trans_num",
                    "trans_date_trans_time",
                    "cc_num",
                    "unix_time",
                    "merchant",
                    "category",
                    "amt",
                    "dob",
                    "is_fraud",
                ],
            )

            TrainingTransactionSchema.validate(training_df)

            training_df = self.raw_data_handler.convert_dates(training_df)

            final_summary = self._summarize_dataframe(
                training_df,
                important_columns=[
                    "cc_num",
                    "unix_time",
                    "merchant",
                    "category",
                    "amt",
                    "dob",
                    "is_fraud",
                    "day_of_week",
                    "hour",
                    "month_date",
                    "year_date",
                ],
            )

            fraud_rate = None
            if "is_fraud" in training_df.columns and not training_df.empty:
                fraud_rate = float(training_df["is_fraud"].mean())

            self.logger.write_log(
                endpoint="training_preprocessing",
                request_data=request_data,
                response_data={
                    "raw_summary": raw_summary,
                    "merged_summary": merged_summary,
                    "filled_summary": filled_summary,
                    "final_summary": final_summary,
                    "fraud_rate": fraud_rate,
                },
                status="success",
            )

            return training_df

        except Exception as error:
            self.logger.write_log(
                endpoint="training_preprocessing",
                request_data=request_data,
                status="error",
                error=str(error),
            )
            raise

    def run_prediction_preprocessing(
        self,
        input_data: Union[dict[str, Any], pd.DataFrame],
        customer_filename: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Runs preprocessing for prediction input.

        If customer_filename is provided, the prediction input is enriched with
        customer profile data using cc_num.
        """
        request_data = {
            "input_type": type(input_data).__name__,
            "customer_filename": customer_filename,
            "customer_enrichment_requested": customer_filename is not None,
        }

        try:
            prediction_df = self.raw_data_handler.build_prediction_frame(input_data)

            input_summary = self._summarize_dataframe(
                prediction_df,
                important_columns=[
                    "trans_date_trans_time",
                    "cc_num",
                    "unix_time",
                    "merchant",
                    "category",
                    "amt",
                    "merch_lat",
                    "merch_long",
                ],
            )

            PredictionTransactionSchema.validate(prediction_df)

            customer_summary = None

            if customer_filename is not None:
                customer_df = self.raw_data_handler.load_customer_data(customer_filename)

                customer_summary = self._summarize_dataframe(
                    customer_df,
                    important_columns=["cc_num", "dob", "sex", "state"],
                )

                prediction_df = self.raw_data_handler.merge_customer_data(
                    transaction_df=prediction_df,
                    customer_df=customer_df,
                )

            merged_summary = self._summarize_dataframe(
                prediction_df,
                important_columns=[
                    "trans_date_trans_time",
                    "cc_num",
                    "unix_time",
                    "merchant",
                    "category",
                    "amt",
                    "dob",
                    "sex",
                    "state",
                ],
            )

            prediction_df = self.raw_data_handler.fill_missing_values(
                prediction_df,
                fill_values=PredictionTransactionSchema.fill_values,
            )

            filled_summary = self._summarize_dataframe(
                prediction_df,
                important_columns=[
                    "trans_date_trans_time",
                    "cc_num",
                    "unix_time",
                    "merchant",
                    "category",
                    "amt",
                    "dob",
                    "sex",
                    "state",
                ],
            )

            prediction_df = self.raw_data_handler.convert_dates(prediction_df)

            final_summary = self._summarize_dataframe(
                prediction_df,
                important_columns=[
                    "cc_num",
                    "unix_time",
                    "merchant",
                    "category",
                    "amt",
                    "dob",
                    "day_of_week",
                    "hour",
                    "month_date",
                    "year_date",
                ],
            )

            self.logger.write_log(
                endpoint="prediction_preprocessing",
                request_data=request_data,
                response_data={
                    "input_summary": input_summary,
                    "customer_summary": customer_summary,
                    "merged_summary": merged_summary,
                    "filled_summary": filled_summary,
                    "final_summary": final_summary,
                    "customer_enriched": customer_filename is not None,
                },
                status="success",
            )

            return prediction_df

        except Exception as error:
            self.logger.write_log(
                endpoint="prediction_preprocessing",
                request_data=request_data,
                status="error",
                error=str(error),
            )
            raise

    def _summarize_dataframe(
        self,
        df: pd.DataFrame,
        important_columns: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Returns a compact monitoring summary for a DataFrame.
        """
        important_columns = important_columns or []

        return {
            "row_count": int(df.shape[0]),
            "column_count": int(df.shape[1]),
            "total_missing_values": int(df.isna().sum().sum()),
            "important_missing_values": {
                column: int(df[column].isna().sum())
                for column in important_columns
                if column in df.columns
            },
        }