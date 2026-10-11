from __future__ import annotations

import pandas as pd
from typing import Optional
from sklearn.model_selection import train_test_split


class DataSplitter:
    """
    Splits cleaned SecureBank data for model training, validation, and testing.

    This class owns dataset splitting strategy only. It does not perform
    preprocessing, feature engineering, model training, prediction, or evaluation.

    Supported strategies:
    - holdout: random train/validation/test split
    - time_based: trains on older rows, validates on middle rows, tests on newest rows
    """

    SUPPORTED_STRATEGIES = [
        "holdout",
        "time_based",
    ]

    def __init__(
        self,
        strategy: str = "holdout",
        validation_size: float = 0.1,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify_column: Optional[str] = "is_fraud",
        time_column: str = "unix_time",
    ):
        """
        Initializes the data splitter.

        Parameters:
            strategy (str): Validation splitting strategy to use.
                Supported values are "holdout" and "time_based".
            validation_size (float): Proportion of data to use for validation.
            test_size (float): Proportion of data to use for final testing.
            random_state (int): Random seed for reproducible holdout splits.
            stratify_column (Optional[str]=None): Column used for stratified holdout splitting.
            time_column (str): Column used to sort rows for time-based splitting.
        """
        if strategy not in self.SUPPORTED_STRATEGIES:
            raise ValueError(
                f"Split strategy not supported: {strategy}. "
                f"Supported values are: {self.SUPPORTED_STRATEGIES}"
            )

        if validation_size < 0 or validation_size >= 1:
            raise ValueError(
                "validation_size must be greater than or equal to 0 and less than 1"
            )

        if test_size <= 0 or test_size >= 1:
            raise ValueError("test_size must be greater than 0 and less than 1")

        if validation_size + test_size >= 1:
            raise ValueError("validation_size + test_size must be less than 1")

        self.strategy = strategy
        self.validation_size = validation_size
        self.test_size = test_size
        self.random_state = random_state
        self.stratify_column = stratify_column
        self.time_column = time_column

    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits cleaned data into train, validation, and test DataFrames.

        Parameters:
            df (pd.DataFrame): Cleaned training DataFrame.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
                train_df, validation_df, test_df
        """
        if df.empty:
            raise ValueError("Cannot split an empty DataFrame")

        if self.strategy == "holdout":
            return self._holdout_split(df)

        if self.strategy == "time_based":
            return self._time_based_split(df)

        raise ValueError(f"Unsupported split strategy: {self.strategy}")

    def _holdout_split(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Performs a random train/validation/test split.
        """
        stratify_values = self._get_stratify_values(df)

        train_validation_df, test_df = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_values,
        )

        if train_validation_df.empty or test_df.empty:
            raise ValueError("Split produced an empty train or test dataset")

        if self.validation_size == 0:
            validation_df = pd.DataFrame(columns=df.columns)

            return (
                train_validation_df.reset_index(drop=True),
                validation_df.reset_index(drop=True),
                test_df.reset_index(drop=True),
            )

        relative_validation_size = self.validation_size / (1 - self.test_size)

        train_stratify_values = self._get_stratify_values(train_validation_df)

        train_df, validation_df = train_test_split(
            train_validation_df,
            test_size=relative_validation_size,
            random_state=self.random_state,
            stratify=train_stratify_values,
        )

        if train_df.empty or test_df.empty:
            raise ValueError("Split produced an empty train or test dataset")

        if self.validation_size > 0 and validation_df.empty:
            raise ValueError("Split produced an empty validation dataset")

        return (
            train_df.reset_index(drop=True),
            validation_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
        )

    def _time_based_split(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Performs a time-based train/validation/test split.

        Older rows become training data, middle rows become validation data,
        and newest rows become test data.
        """
        if self.time_column not in df.columns:
            raise ValueError(f"Time column not found: {self.time_column}")

        sorted_df = df.sort_values(by=self.time_column).reset_index(drop=True)

        train_end = int(len(sorted_df) * (1 - self.validation_size - self.test_size))
        validation_end = int(len(sorted_df) * (1 - self.test_size))

        train_df = sorted_df.iloc[:train_end].copy()
        validation_df = sorted_df.iloc[train_end:validation_end].copy()
        test_df = sorted_df.iloc[validation_end:].copy()

        if train_df.empty or test_df.empty:
            raise ValueError("Split produced an empty train or test dataset")

        if self.validation_size > 0 and validation_df.empty:
            raise ValueError("Split produced an empty validation dataset")

        return (
            train_df.reset_index(drop=True),
            validation_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
        )

    def _get_stratify_values(self, df: pd.DataFrame) -> Optional[pd.Series]:
        """
        Returns stratification values if stratification is enabled.
        """
        if self.stratify_column is None:
            return None

        if self.stratify_column not in df.columns:
            raise ValueError(f"Stratify column not found: {self.stratify_column}")

        return df[self.stratify_column]