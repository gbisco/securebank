import unittest
import pandas as pd
import json
import os
import tempfile
import numpy as np
from unittest.mock import patch, mock_open
from securebank.modules.data.raw_data_handler import RawDataHandler  # Assuming the class is saved in raw_data_handler.py
 
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
 
class TestRawDataHandler(unittest.TestCase):
 
    @classmethod  
    def setUpClass(self):
        
        # Dynamically construct absolute paths to data and output
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        securebank_dir = os.path.join(repo_root, "securebank")
        
        self.storage_path = os.path.join(securebank_dir, "data_sources")
        self.save_path = os.path.join(securebank_dir, "output")

        self.handler = RawDataHandler(self.storage_path, self.save_path)

        self.customer, self.transaction, self.fraud = self.handler.extract(
            'customer_release.csv',
            'transactions_release.parquet',
            'fraud_release.json'
        )

        self.transformed_df = self.handler.transform(self.customer, self.transaction, self.fraud)
        self.covert_dates_df = self.handler.convert_dates(self.transformed_df)
        self.description = self.handler.describe(self.covert_dates_df)


    def test_check_methods_exist(self):
        # check that all methods exist
        self.assertTrue(hasattr(RawDataHandler, '__init__'), "No constructor method")
        self.assertTrue(hasattr(RawDataHandler, 'extract'), "No extract method")
        self.assertTrue(hasattr(RawDataHandler, 'convert_dates'), "No convert_dates method")
        self.assertTrue(hasattr(RawDataHandler, 'transform'), "No transform method")
        self.assertTrue(hasattr(RawDataHandler, 'describe'), "No describe method")

    def test_extract_success(self):
        self.assertEqual(len(self.customer), 750, "Customer information not properly parsed")
        self.assertEqual(len(self.transaction), 1647542, "Transaction information not properly parsed")
        self.assertEqual(len(self.fraud), 1647542, "Fraud event information not properly parsed")

    def create_set(self, start, end):
        """Creates a set of numbers between start and end (inclusive)."""
        return set(range(start, end + 1))

    def test_convert_dates_day_of_week_column(self):
        self.assertIn('day_of_week', self.covert_dates_df.columns, "No day_of_week column, (-5 pts)")

    def test_convert_dates_hour_column(self):
        self.assertIn('hour', self.covert_dates_df.columns, "No hour column, (-5 pts)")

    def test_convert_dates_minute_column(self):
        self.assertIn('minute', self.covert_dates_df.columns, "No minute column, (-5 pts)")

    def test_convert_dates_seconds_column(self):
        self.assertIn('seconds', self.covert_dates_df.columns, "No seconds column, (-5 pts)")

    def test_convert_dates_day_date_column(self):
        self.assertIn('day_date', self.covert_dates_df.columns, "No day_date column, (-5 pts)")

    def test_convert_dates_month_date_column(self):
        self.assertIn('month_date', self.covert_dates_df.columns, "No month_date column, (-5 pts)")

    def test_convert_dates_year_date_column(self):
        self.assertIn('year_date', self.covert_dates_df.columns, "No year_date column, (-5 pts)")

    def test_convert_dates_trans_date_trans_time_column(self):
        
        self.assertNotIn('trans_date_trans_time', self.covert_dates_df.columns, "Existing trans_date_trans_time column, (-5 pts)")

    def test_date_ranges_check(self):
        month_names = {'July','April','October','February','November','January','December','June','May','March','September','August'}
        days_of_weeks = {"Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"}
        possible_years = {2019, 2020}
        self.assertSetEqual(set(self.covert_dates_df.day_of_week.unique()), days_of_weeks, "Column day_of_week not correct, (-5 pts)")
        self.assertSetEqual(set(self.covert_dates_df.hour.unique()), self.create_set(0, 23), "Column hour not correct, (-5 pts)")
        self.assertSetEqual(set(self.covert_dates_df.minute.unique()), self.create_set(0, 59), "Column minute not correct, (-5 pts)")
        self.assertSetEqual(set(self.covert_dates_df.seconds.unique()), self.create_set(0, 59), "Column seconds not correct, (-5 pts)")
        self.assertSetEqual(set(self.covert_dates_df.day_date.unique()), self.create_set(1, 31), "Column day_date not correct, (-5 pts)")
        self.assertSetEqual(set(self.covert_dates_df.month_date.unique()), month_names, "Column month_date not correct, (-5 pts)")
        self.assertSetEqual(set(self.covert_dates_df.year_date.unique()), possible_years, "Column year_date not correct, (-5 pts)")

    def test_drop_na_is_fraud(self):
        self.assertGreaterEqual(len(self.covert_dates_df.is_fraud.unique()), 2, "Column is_fraud not properly formatted, (-5 pts)")

    def test_transform_success(self):
        print(self.transformed_df)
        self.assertLessEqual(len(self.transformed_df), 1647542)
        self.assertIn('is_fraud', self.transformed_df.columns, "Column is_fraud not found in output DataFrame, (-5 pts)")
        self.assertIn('cc_num', self.transformed_df.columns)
        self.assertIn('unix_time', self.transformed_df.columns)
        self.assertIn('amt', self.transformed_df.columns)

    def test_describe_values(self):    
        self.assertLessEqual(self.description["number_of_records"], 1647542)
        self.assertGreaterEqual(len(self.description["feature_names"]), 3)

    def test_describe_output(self):
        necessary_columns = {"number_of_records", "number_of_columns", "feature_names", "number_missing_values", "column_data_types"}
        self.assertTrue(set(self.description.keys()).issuperset(necessary_columns),
                        f"Missing columns: Only has {set(self.description.keys()).intersection(necessary_columns)}, (-5 pts)")
        
if __name__ == '__main__':
    unittest.main()
