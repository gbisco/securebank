from dataclasses import dataclass
from typing import ClassVar
import pandas as pd


@dataclass(frozen=True)
class TrainingTransactionSchema:
    required_columns: ClassVar[list[str]] = [
        "cc_num",
        "amt",
        "trans_date_trans_time",
        "is_fraud",
    ]

    fill_values: ClassVar[dict] = {
        "unix_time": -1,
        "merchant": "unknown",
        "category": "unknown",
        "merch_lat": -1,
        "merch_long": -1,
    }

    @staticmethod
    def validate(df: pd.DataFrame) -> None:
        if df.empty:
            raise ValueError("Dataset is empty")

        missing = [
            col for col in TrainingTransactionSchema.required_columns
            if col not in df.columns
        ]

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if df[TrainingTransactionSchema.required_columns].isna().any().any():
            raise ValueError("Required columns contain missing values")

        pd.to_numeric(df["cc_num"], errors="raise")
        pd.to_numeric(df["amt"], errors="raise")
        pd.to_numeric(df["is_fraud"], errors="raise")
        pd.to_datetime(df["trans_date_trans_time"], errors="raise")

@dataclass(frozen=True)
class PredictionTransactionSchema:
    required_columns: ClassVar[list[str]] = [
        "cc_num",
        "amt",
        "trans_date_trans_time",
    ]

    fill_values: ClassVar[dict] = {
        "unix_time": -1,
        "merchant": "unknown",
        "category": "unknown",
        "merch_lat": -1,
        "merch_long": -1,
    }

    @staticmethod
    def validate(df: pd.DataFrame) -> None:
        if df.empty:
            raise ValueError("Prediction input is empty")

        missing = [
            col for col in PredictionTransactionSchema.required_columns
            if col not in df.columns
        ]

        if missing:
            raise ValueError(f"Missing required prediction columns: {missing}")

        if df[PredictionTransactionSchema.required_columns].isna().any().any():
            raise ValueError("Required prediction columns contain missing values")

        pd.to_numeric(df["cc_num"], errors="raise")
        pd.to_numeric(df["amt"], errors="raise")
        pd.to_datetime(df["trans_date_trans_time"], errors="raise")