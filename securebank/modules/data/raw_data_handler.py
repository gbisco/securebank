# Imports
import os
import json
import pandas as pd
from pandas.errors import ParserError
#######################################
# LEGACY FOR GRADER ONLY
#######################################
#######################################
class RawDataHandler:
    """
    Handles extraction, transformation, and description 
    of raw data for machine learning preprocessing.
    """

    # Custom constructor
    def __init__(self, storage_path: str, save_path: str):
        """
        Initializes data handler with following parameters:
        Parameters:
        - `storage_path`: Path where the raw data files are stored.  
        - `save_path`: Path where cleaned data will be saved.  
        """
        self.storage_path = storage_path
        self.save_path = save_path

    # Methods
    def extract(self, customer_information_filename:str, transaction_filename:str,fraud_information_filename:str) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
        """
        **Purpose:** Reads raw data files and returns them as DataFrames.  

        **Parameters:**
        - `customer_information_filename` (CSV)
        - `transaction_filename` (Parquet)  
        - `fraud_information_filename` (JSON)  

        **Output:**  
        - `customer_information` (DataFrame)  
        - `transaction_information` (DataFrame)  
        - `fraud_information` (DataFrame)
        """
        # Read csv and assign df
        try:
            customer_information = pd.read_csv(os.path.join(self.storage_path,customer_information_filename))
        # Except file not found
        except FileNotFoundError:
            print(f"Error: The file '{customer_information_filename}' was not found.")
        # Except parsing error
        except ParserError as err:
             print(f"Error parsing '{customer_information_filename}': {err}")

        # Read Parquet and assign df
        try:
            transaction_information = pd.read_parquet(os.path.join(self.storage_path,transaction_filename))
        # Except file not found
        except FileNotFoundError:
            print(f"Error: The file '{transaction_filename}' was not found.")
        # Except parsing error
        except ParserError as err:
             print(f"Error parsing '{transaction_filename}': {err}")

        # Read JSON and assign df (requires mapping)
        try:
            with open(os.path.join(self.storage_path,fraud_information_filename)) as f:
                fraud_dict = json.load(f)

            fraud_information = pd.DataFrame({
                "trans_id": list(fraud_dict.keys()),
                "is_fraud": list(fraud_dict.values())
            })
        # Except file not found
        except FileNotFoundError as err:
            raise FileNotFoundError(f"Fraud file not found: {fraud_information_filename}") from err
        # Except invalid JSON
        except json.JSONDecodeError as err:
            raise ValueError(f"Fraud JSON is invalid: {fraud_information_filename}") from err
    
        # Return dataframes
        return customer_information, transaction_information, fraud_information
    
    def convert_dates(self, df: pd.DataFrame)-> pd.DataFrame:
        """
        **Purpose:** Converts `trans_date_trans_time` into seven distinct columns:  
        - `day_of_week` (string, e.g. Monday)  
        - `hour` (int 0-23)  
        - `minute` (int 0-59)  
        - `seconds` (int 0-59)  
        - `day_date` (int 1-31)  
        - `month_date` (string, e.g. January)  
        - `year_date` (int, e.g. 2025)

        **Input:** DataFrame with `trans_date_trans_time`.

        **Output:** Modified DataFrame with these new columns appended (and `trans_date_trans_time` removed).
        """
        # If not trans_date_trans_time return original
        if "trans_date_trans_time" not in df.columns:
            return df
        
        # Safe copy df
        new_df = df.copy()

        # Modify column to datetime type
        new_df['trans_date_trans_time'] = pd.to_datetime(new_df['trans_date_trans_time'], errors="coerce") # Coerce errors to NaT

        # Append new columns
        new_df['day_of_week'] = new_df['trans_date_trans_time'].dt.day_name()
        new_df['hour'] = new_df['trans_date_trans_time'].dt.hour
        new_df['minute'] = new_df['trans_date_trans_time'].dt.minute
        new_df['seconds'] = new_df['trans_date_trans_time'].dt.second
        new_df['day_date'] = new_df['trans_date_trans_time'].dt.day
        new_df['month_date'] = new_df['trans_date_trans_time'].dt.month_name()
        new_df['year_date'] = new_df['trans_date_trans_time'].dt.year

        # Drop date column and return
        new_df.drop(columns=['trans_date_trans_time'],inplace=True)
        return new_df
    
    def transform(self, customer_information:pd.DataFrame, transaction_information:pd.DataFrame, fraud_information:pd.DataFrame)-> pd.DataFrame:
        """
        **Purpose:** Prepares and cleans data by:  
        - Merging the three data sources  
        - Imputing/dropping missing values  
        - Dropping duplicate rows  

        **Input:**  
        - `customer_information`  
        - `transaction_information`  
        - `fraud_information`  

        **Output:** A cleaned and merged DataFrame where each row is a **unique transaction** with all relevant information.
        """

        # Safe copy
        customer_copy = customer_information.copy()
        transactions_copy = transaction_information.copy()
        fraud_copy = fraud_information.copy()

        transactions_copy = transactions_copy.reset_index(drop=True)
        fraud_copy = fraud_copy.reset_index(drop=True)

        if len(transactions_copy) != len(fraud_copy):
            raise ValueError("Transaction and fraud datasets must have the same number of rows.")

        transactions_copy["is_fraud"] = fraud_copy["is_fraud"].values

        new_df = transactions_copy.merge(
            customer_copy,
            on="cc_num",
            how="left",
            suffixes=("", "_customer")
        )

        new_df = new_df.drop_duplicates()

        # Fill optional fields
        new_df["merchant"] = new_df["merchant"].fillna("unknown")
        new_df["category"] = new_df["category"].fillna("unknown")
        new_df["unix_time"] = new_df["unix_time"].fillna(-1)

        # Only enforce truly required fields
        required_columns = [
            "cc_num",
            "amt",
            "is_fraud",
        ]

        new_df = new_df.dropna(subset=required_columns)

        return new_df
    
    def describe(self, raw_data:pd.DataFrame)-> dict:
        """
        **Purpose:** Produces a summary of quality metrics. 

        **Input:** `raw_data` (cleaned DataFrame)

        **Output:** Dictionary with:
        - `"number_of_records"` (int)  
        - `"number_of_columns"` (int)  
        - `"feature_names"` (list of str)  
        - `"number_missing_values"` (int)  
        - `"column_data_types"` (list of str)
        """

        return{
            "number_of_records": int(raw_data.shape[0]),
            "number_of_columns": int(raw_data.shape[1]),
            "feature_names": raw_data.columns.tolist(),
            "number_missing_values": int(raw_data.isna().sum().sum()),
            "column_data_types": [str(dtype) for dtype in raw_data.dtypes.tolist()],}

#######################################
 
#######################################
# # V2 COMPATIBLE 
#######################################
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from securebank.modules.data.schemas import TrainingTransactionSchema
from securebank.modules.data.schemas import PredictionTransactionSchema

class RawDataHandlerV2:
    """
    Handles low-level SecureBank data loading and raw DataFrame shaping.

    This class prepares raw training and prediction data into a consistent
    transaction-level DataFrame structure.

    It does not perform full preprocessing, model-specific feature engineering,
    training, prediction, or orchestration.
    """

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path) if storage_path else None

        if self.storage_path:
            if not self.storage_path.exists():
                raise ValueError(f"Storage path does not exist: {self.storage_path}")

            if not self.storage_path.is_dir():
                raise ValueError(f"Storage path is not a directory: {self.storage_path}")

    # =========================
    # File loading
    # =========================

    def load_customer_data(self, filename: str) -> pd.DataFrame:
        """Loads customer data from CSV."""
        if not self.storage_path:
            raise ValueError("Storage path is not set for loading customer data")

        file_path = self.storage_path / filename

        if not file_path.exists():
            raise FileNotFoundError(f"Customer file not found: {file_path}")

        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"Failed to read customer CSV: {e}") from e


    def load_transaction_data(self, filename: str) -> pd.DataFrame:
        """Loads transaction data from Parquet."""
        if not self.storage_path:
            raise ValueError("Storage path is not set for loading transaction data")

        file_path = self.storage_path / filename

        if not file_path.exists():
            raise FileNotFoundError(f"Transaction file not found: {file_path}")

        try:
            return pd.read_parquet(file_path)
        except Exception as e:
            raise ValueError(f"Failed to read transaction Parquet: {e}") from e


    def load_fraud_data(self, filename: str) -> pd.DataFrame:
        """Loads fraud labels from JSON and returns a two-column DataFrame."""
        if not self.storage_path:
            raise ValueError("Storage path is not set for loading fraud data")

        file_path = self.storage_path / filename

        if not file_path.exists():
            raise FileNotFoundError(f"Fraud file not found: {file_path}")

        try:
            with open(file_path) as f:
                fraud_dict = json.load(f)

            fraud_df = pd.DataFrame({
                "trans_id": list(fraud_dict.keys()),
                "is_fraud": list(fraud_dict.values())
            })

            return fraud_df

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in fraud file: {file_path}") from e

    # =========================
    # Training frame construction
    # =========================

    def build_training_frame(
        self,
        customer_df: pd.DataFrame,
        transaction_df: pd.DataFrame,
        fraud_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Builds one transaction-level training DataFrame from customer,
        transaction, and fraud data.
        """
        if customer_df.empty:
            raise ValueError("Customer dataset is empty")

        if transaction_df.empty:
            raise ValueError("Transaction dataset is empty")

        if fraud_df.empty:
            raise ValueError("Fraud dataset is empty")

        transaction_df = self.normalize_transaction_index(transaction_df)

        transaction_df = self.attach_fraud_labels(
            transaction_df=transaction_df,
            fraud_df=fraud_df,
        )

        training_df = self.merge_customer_data(
            transaction_df=transaction_df,
            customer_df=customer_df,
        )

        training_df = self.drop_duplicate_transactions(training_df)

        training_df = self.drop_missing_required(
            training_df,
            required_columns=TrainingTransactionSchema.required_columns,
        )

        TrainingTransactionSchema.validate(training_df)

        return training_df

    def normalize_transaction_index(self, transaction_df: pd.DataFrame) -> pd.DataFrame:
        """Moves transaction id/index into a normal trans_num column."""

        if transaction_df.empty:
            raise ValueError("Transaction dataset is empty")

        normalized_df = transaction_df.copy()

        if "trans_num" in normalized_df.columns:
            return normalized_df

        normalized_df = normalized_df.reset_index()

        if "trans_num" not in normalized_df.columns:
            normalized_df = normalized_df.rename(columns={"index": "trans_num"})

        return normalized_df

    def attach_fraud_labels(
        self,
        transaction_df: pd.DataFrame,
        fraud_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Attaches is_fraud labels to transaction rows."""

        if transaction_df.empty:
            raise ValueError("Transaction dataset is empty")

        if fraud_df.empty:
            raise ValueError("Fraud dataset is empty")

        if "trans_num" not in transaction_df.columns:
            raise ValueError("Transaction data must contain trans_num before attaching fraud labels")

        required_fraud_columns = ["trans_id", "is_fraud"]
        missing = [
            col for col in required_fraud_columns
            if col not in fraud_df.columns
        ]

        if missing:
            raise ValueError(f"Fraud data missing required columns: {missing}")

        merged_df = transaction_df.merge(
            fraud_df[required_fraud_columns],
            left_on="trans_num",
            right_on="trans_id",
            how="left",
        )

        merged_df = merged_df.drop(columns=["trans_id"])

        return merged_df

    def merge_customer_data(
        self,
        transaction_df: pd.DataFrame,
        customer_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merges customer profile data onto transaction rows using cc_num."""

        if transaction_df.empty:
            raise ValueError("Transaction dataset is empty")

        if customer_df.empty:
            raise ValueError("Customer dataset is empty")

        if "cc_num" not in transaction_df.columns:
            raise ValueError("Transaction data missing cc_num")

        if "cc_num" not in customer_df.columns:
            raise ValueError("Customer data missing cc_num")

        merged_df = transaction_df.merge(
            customer_df,
            on="cc_num",
            how="left",
            suffixes=("", "_customer"),
        )

        return merged_df

    # =========================
    # Prediction frame construction
    # =========================

    def build_prediction_frame(self, input_data: dict[str, Any]) -> pd.DataFrame:
        """
        Converts one prediction request dictionary into a transaction DataFrame.
        """
        prediction_df = self.ensure_dataframe(input_data)

        prediction_df = self.drop_missing_required(
            prediction_df,
            required_columns=PredictionTransactionSchema.required_columns,
        )

        PredictionTransactionSchema.validate(prediction_df)

        return prediction_df

    def ensure_dataframe(self, data: Any) -> pd.DataFrame:
        """Converts dict/list/DataFrame input into a DataFrame."""
        if isinstance(data, pd.DataFrame):
            return data.copy()

        if isinstance(data, dict):
            return pd.DataFrame([data])

        if isinstance(data, list):
            return pd.DataFrame(data)

        raise ValueError("Input data must be a dictionary, list of dictionaries, or DataFrame")

    # =========================
    # Shared cleaning / shaping
    # =========================

    def fill_missing_values(
        self,
        df: pd.DataFrame,
        fill_values: Optional[dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Fills missing values using provided defaults."""
        
        if df.empty:
            raise ValueError("Cannot fill missing values in an empty DataFrame")

        cleaned_df = df.copy()

        if fill_values:
            for column, value in fill_values.items():
                if column in cleaned_df.columns:
                    cleaned_df[column] = cleaned_df[column].fillna(value)

        return cleaned_df

    def drop_duplicate_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drops duplicate transaction rows."""
        if df.empty:
            raise ValueError("Cannot drop duplicates from an empty DataFrame")

        cleaned_df = df.copy()

        if "trans_num" in cleaned_df.columns:
            return cleaned_df.drop_duplicates(subset=["trans_num"])

        return cleaned_df.drop_duplicates()

    def drop_missing_required(
        self,
        df: pd.DataFrame,
        required_columns: list[str],
    ) -> pd.DataFrame:
        """Drops rows missing required fields."""
        if df.empty:
            raise ValueError("Cannot drop missing values from an empty DataFrame")

        missing_columns = [
            column for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        cleaned_df = df.copy()
        cleaned_df = cleaned_df.dropna(subset=required_columns)

        if cleaned_df.empty:
            raise ValueError("No rows remain after dropping missing required values")

        return cleaned_df

    def convert_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Expands trans_date_trans_time into date/time feature columns."""

        if df.empty:
            raise ValueError("Cannot convert dates on an empty DataFrame")

        if "trans_date_trans_time" not in df.columns:
            return df

        new_df = df.copy()

        new_df["trans_date_trans_time"] = pd.to_datetime(
            new_df["trans_date_trans_time"],
            errors="coerce"
        )

        new_df["day_of_week"] = new_df["trans_date_trans_time"].dt.day_name()
        new_df["hour"] = new_df["trans_date_trans_time"].dt.hour
        new_df["minute"] = new_df["trans_date_trans_time"].dt.minute
        new_df["seconds"] = new_df["trans_date_trans_time"].dt.second
        new_df["day_date"] = new_df["trans_date_trans_time"].dt.day
        new_df["month_date"] = new_df["trans_date_trans_time"].dt.month_name()
        new_df["year_date"] = new_df["trans_date_trans_time"].dt.year

        new_df = new_df.drop(columns=["trans_date_trans_time"])

        return new_df

    def describe(self, df: pd.DataFrame) -> dict:
        """Returns basic DataFrame quality summary."""

        if df.empty:
            raise ValueError("Cannot describe an empty DataFrame")

        return {
            "number_of_records": int(df.shape[0]),
            "number_of_columns": int(df.shape[1]),
            "feature_names": df.columns.tolist(),
            "number_missing_values": int(df.isna().sum().sum()),
            "column_data_types": [str(dtype) for dtype in df.dtypes.tolist()],
        }





        