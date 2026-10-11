from __future__ import annotations

import pandas as pd
from typing import Optional
from sklearn.preprocessing import StandardScaler

class LogisticRegressionAdapter:
    """
    Prepares preprocessed SecureBank transaction data for a logistic regression model.

    This adapter is responsible for model-specific feature preparation only.
    It assumes the input DataFrame has already gone through the generic
    preprocessing pipeline.

    Responsibilities may include:
    - separating features and target
    - dropping non-useful columns
    - transforming special columns
    - encoding categorical features
    - scaling numerical features
    - aligning prediction features with training features
    """

    DROP_COLUMNS = [
        "trans_num",
        "index",
        "index_customer",
        "cc_num",
        "first",
        "last",
        "street",
        "city",
        "zip",
        "city_pop",
        "merchant",
        "minute",
        "seconds",
    ]

    NUMERIC_COLUMNS = [
        "unix_time",
        "amt",
        "merch_lat",
        "merch_long",
        "lat",
        "long",
        "hour",
        "day_date",
        "year_date",
        "age",
        "job_frequency",
        "minutes_since_previous_transaction",
        "repeat_within_5_minutes",
        "repeat_within_30_minutes",
        "repeat_within_1_hour",
        "minute_of_day",
        "is_night",
        "is_evening",
        "is_weekend",
    ]

    CATEGORICAL_COLUMNS = [
        "category",
        "sex",
        "state",
        "job",
        "day_of_week",
        "month_date",
    ]

    TARGET_COLUMN = "is_fraud"

    def __init__(self):
        self.feature_columns: Optional[list[str]] = None
        self.scaler = StandardScaler()
        self.frequency_maps: dict[str, dict[str, float]] = {}
        self.median_minutes_since_previous_transaction: float = 0

    def _scale_numeric_columns(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """
        Scales numeric feature columns for logistic regression.
        """
        scaled_df = df.copy()

        numeric_columns = [
            column for column in self.NUMERIC_COLUMNS
            if column in scaled_df.columns
        ]

        if not numeric_columns:
            return scaled_df

        if fit:
            scaled_df[numeric_columns] = self.scaler.fit_transform(
                scaled_df[numeric_columns]
            )
        else:
            scaled_df[numeric_columns] = self.scaler.transform(
                scaled_df[numeric_columns]
            )

        return scaled_df

    def _frequency_encode_column(
        self,
        df: pd.DataFrame,
        column: str,
        fit: bool,
    ) -> pd.DataFrame:
        """
        Frequency encodes one high-cardinality categorical column.
        """
        encoded_df = df.copy()

        if column not in encoded_df.columns:
            return encoded_df

        encoded_df[column] = (
            encoded_df[column]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        if fit:
            frequency_map = encoded_df[column].value_counts(normalize=True).to_dict()
            self.frequency_maps[column] = frequency_map
        else:
            frequency_map = self.frequency_maps.get(column, {})

        encoded_df[f"{column}_frequency"] = (
            encoded_df[column]
            .map(frequency_map)
            .fillna(0)
        )

        return encoded_df.drop(columns=[column])

    def _encode_categorical_columns(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """
        Encodes categorical columns into numeric model features.
        """
        encoded_df = df.copy()

        for col in self.CATEGORICAL_COLUMNS:
            if col not in encoded_df.columns:
                continue

            encoded_df[col] = (
                encoded_df[col]
                .astype(str)
                .str.strip()
                .str.lower()
            )

            if col == "job":
                encoded_df = self._frequency_encode_column(encoded_df, col, fit=fit)
            else:
                dummies = pd.get_dummies(encoded_df[col], prefix=col).astype(int)
                encoded_df = pd.concat([encoded_df, dummies], axis=1)
                encoded_df = encoded_df.drop(columns=[col])

        return encoded_df

    def _create_age_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates an age feature from the dob column and drops dob.
        """
        feature_df = df.copy()

        if "dob" not in feature_df.columns:
            return feature_df

        dob = pd.to_datetime(
            feature_df["dob"],
            errors="coerce",
            format="mixed",
        )

        feature_df["age"] = feature_df["year_date"] - dob.dt.year

        median_age = feature_df["age"].median()

        if pd.isna(median_age):
            median_age = 0

        feature_df["age"] = feature_df["age"].fillna(median_age)

        feature_df = feature_df.drop(columns=["dob"])

        return feature_df

    def _create_time_delta_features(
        self,
        df: pd.DataFrame,
        fit: bool,
    ) -> pd.DataFrame:
        """
        Creates transaction time-delta features using cc_num and unix_time.

        During training, the feature is calculated from the training batch and
        the median time delta is stored. During prediction, single-row inputs
        fall back to the stored training median.
        """
        feature_df = df.copy()

        required_columns = ["cc_num", "unix_time"]

        if not all(column in feature_df.columns for column in required_columns):
            return feature_df

        original_index = feature_df.index

        feature_df = feature_df.sort_values(
            by=["cc_num", "unix_time"]
        ).copy()

        feature_df["seconds_since_previous_transaction"] = pd.NA

        valid_unix_time = feature_df["unix_time"] >= 0
        valid_index = feature_df[valid_unix_time].index

        feature_df.loc[valid_index, "seconds_since_previous_transaction"] = (
            feature_df
            .loc[valid_index]
            .groupby("cc_num")["unix_time"]
            .diff()
        )

        feature_df["minutes_since_previous_transaction"] = (
            pd.to_numeric(
                feature_df["seconds_since_previous_transaction"],
                errors="coerce",
            ) / 60
        )

        if fit:
            median_minutes = feature_df["minutes_since_previous_transaction"].median()

            if pd.isna(median_minutes):
                median_minutes = 0

            self.median_minutes_since_previous_transaction = float(median_minutes)

        feature_df["minutes_since_previous_transaction"] = (
            feature_df["minutes_since_previous_transaction"]
            .fillna(self.median_minutes_since_previous_transaction)
            .clip(lower=0)
        )

        feature_df["repeat_within_5_minutes"] = (
            feature_df["minutes_since_previous_transaction"] <= 5
        ).astype(int)

        feature_df["repeat_within_30_minutes"] = (
            feature_df["minutes_since_previous_transaction"] <= 30
        ).astype(int)

        feature_df["repeat_within_1_hour"] = (
            feature_df["minutes_since_previous_transaction"] <= 60
        ).astype(int)

        feature_df = feature_df.drop(
            columns=["seconds_since_previous_transaction"],
            errors="ignore",
        )

        feature_df = feature_df.loc[original_index]

        return feature_df

    def _create_time_behavior_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates simple time behavior features from existing preprocessing outputs.
        """
        feature_df = df.copy()

        if "hour" in feature_df.columns:
            feature_df["is_night"] = feature_df["hour"].between(0, 5).astype(int)
            feature_df["is_evening"] = feature_df["hour"].between(18, 23).astype(int)

        if "hour" in feature_df.columns and "minute" in feature_df.columns:
            feature_df["minute_of_day"] = (
                feature_df["hour"] * 60 + feature_df["minute"]
            )

        if "day_of_week" in feature_df.columns:
            normalized_day = (
                feature_df["day_of_week"]
                .astype(str)
                .str.strip()
                .str.lower()
            )

            feature_df["is_weekend"] = normalized_day.isin(
                ["saturday", "sunday"]
            ).astype(int)

        return feature_df

    def _align_prediction_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aligns prediction features to match the training feature columns.
        """
        if self.feature_columns is None:
            raise ValueError("Adapter must be fitted with training data before prediction")

        return df.reindex(columns=self.feature_columns, fill_value=0)

    def prepare_training_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """
        Converts a preprocessed training DataFrame into model-ready X and y.
        """
        training_df = df.copy()

        if self.TARGET_COLUMN not in training_df.columns:
            raise ValueError(
                f"Training data must contain {self.TARGET_COLUMN} target column"
            )

        y = training_df[self.TARGET_COLUMN]
        X = training_df.drop(columns=[self.TARGET_COLUMN])

        X = self._create_time_delta_features(X, fit=True)
        X = self._create_time_behavior_features(X)
        X = self._create_age_feature(X)

        X = X.drop(columns=self.DROP_COLUMNS, errors="ignore")

        X = self._encode_categorical_columns(X, fit=True)
        X = self._scale_numeric_columns(X, fit=True)

        self.feature_columns = X.columns.tolist()

        return X, y

    def prepare_prediction_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts a preprocessed prediction DataFrame into model-ready X.
        """
        prediction_df = df.copy()

        prediction_df = prediction_df.drop(
            columns=[self.TARGET_COLUMN],
            errors="ignore",
        )

        X = self._create_time_delta_features(prediction_df, fit=False)
        X = self._create_time_behavior_features(X)
        X = self._create_age_feature(X)

        X = X.drop(columns=self.DROP_COLUMNS, errors="ignore")

        X = self._encode_categorical_columns(X, fit=False)
        X = self._align_prediction_columns(X)
        X = self._scale_numeric_columns(X, fit=False)

        return X