"""
Utility helper functions for CortexX sales forecasting platform.
Provides common utility functions used across the application.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union
import logging
from datetime import datetime, timedelta
import os
import json

logger = logging.getLogger(__name__)

class DataValidator:
    """
    A class to provide data validation utilities.
    """
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_columns: List[str] = None) -> Dict[str, Any]:
        """
        Validate DataFrame structure and content.
        
        Args:
            df (pd.DataFrame): DataFrame to validate
            required_columns (List[str]): List of required columns
            
        Returns:
            Dict[str, Any]: Validation results
        """
        validation_result = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'summary': {}
        }
        
        try:
            # Check if DataFrame is empty
            if df.empty:
                validation_result['is_valid'] = False
                validation_result['issues'].append("DataFrame is empty")
                return validation_result
            
            # Check required columns
            if required_columns:
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    validation_result['is_valid'] = False
                    validation_result['issues'].append(f"Missing required columns: {missing_columns}")
            
            # Check for excessive missing values
            missing_percentage = (df.isnull().sum() / len(df)) * 100
            high_missing = missing_percentage[missing_percentage > 30].index.tolist()
            if high_missing:
                validation_result['warnings'].append(f"High missing values in: {high_missing}")
            
            # Check for constant columns
            constant_columns = df.columns[df.nunique() <= 1].tolist()
            if constant_columns:
                validation_result['warnings'].append(f"Constant columns: {constant_columns}")
            
            # Create summary
            validation_result['summary'] = {
                'shape': df.shape,
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
                'data_types': dict(df.dtypes.value_counts()),
                'missing_values_total': df.isnull().sum().sum()
            }
            
            return validation_result
            
        except Exception as e:
            validation_result['is_valid'] = False
            validation_result['issues'].append(f"Validation error: {str(e)}")
            return validation_result
    
    @staticmethod
    def detect_anomalies(df: pd.DataFrame, column: str, method: str = 'iqr') -> pd.Series:
        """
        Detect anomalies in a column using specified method.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            column (str): Column to check for anomalies
            method (str): Anomaly detection method
            
        Returns:
            pd.Series: Boolean series indicating anomalies
        """
        try:
            if method == 'iqr':
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                return (df[column] < lower_bound) | (df[column] > upper_bound)
            
            elif method == 'zscore':
                from scipy import stats
                z_scores = np.abs(stats.zscore(df[column]))
                return z_scores > 3
            
            else:
                logger.warning(f"Unknown anomaly detection method: {method}")
                return pd.Series([False] * len(df))
                
        except Exception as e:
            logger.error(f"Error detecting anomalies: {str(e)}")
            return pd.Series([False] * len(df))

class DateHandler:
    """
    A class to handle date and time operations.
    """
    
    @staticmethod
    def ensure_datetime(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """
        Ensure specified column is datetime type.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            date_column (str): Date column name
            
        Returns:
            pd.DataFrame: DataFrame with datetime column
        """
        try:
            df_copy = df.copy()
            if date_column in df_copy.columns:
                df_copy[date_column] = pd.to_datetime(df_copy[date_column])
            return df_copy
        except Exception as e:
            logger.error(f"Error converting to datetime: {str(e)}")
            return df
    
    @staticmethod
    def create_date_range(start_date: Union[str, datetime], 
                         end_date: Union[str, datetime], 
                         freq: str = 'D') -> pd.DatetimeIndex:
        """
        Create a date range between start and end dates.
        
        Args:
            start_date (Union[str, datetime]): Start date
            end_date (Union[str, datetime]): End date
            freq (str): Frequency string
            
        Returns:
            pd.DatetimeIndex: Date range
        """
        try:
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            return pd.date_range(start=start, end=end, freq=freq)
        except Exception as e:
            logger.error(f"Error creating date range: {str(e)}")
            return pd.DatetimeIndex([])
    
    @staticmethod
    def extract_date_features(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """
        Extract common date features from datetime column.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            date_column (str): Date column name
            
        Returns:
            pd.DataFrame: DataFrame with date features added
        """
        try:
            df_copy = df.copy()
            df_copy[date_column] = pd.to_datetime(df_copy[date_column])
            
            # Extract common date features
            df_copy['year'] = df_copy[date_column].dt.year
            df_copy['month'] = df_copy[date_column].dt.month
            df_copy['quarter'] = df_copy[date_column].dt.quarter
            df_copy['day_of_week'] = df_copy[date_column].dt.dayofweek
            df_copy['day_of_year'] = df_copy[date_column].dt.dayofyear
            df_copy['is_weekend'] = df_copy['day_of_week'].isin([5, 6]).astype(int)
            
            return df_copy
        except Exception as e:
            logger.error(f"Error extracting date features: {str(e)}")
            return df

class FileManager:
    """
    A class to handle file operations.
    """
    
    @staticmethod
    def ensure_directory(directory_path: str) -> bool:
        """
        Ensure directory exists, create if it doesn't.
        
        Args:
            directory_path (str): Directory path
            
        Returns:
            bool: True if directory exists or was created successfully
        """
        try:
            os.makedirs(directory_path, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Error creating directory {directory_path}: {str(e)}")
            return False
    
    @staticmethod
    def save_dataframe(df: pd.DataFrame, file_path: str, **kwargs) -> bool:
        """
        Save DataFrame to file with error handling.
        
        Args:
            df (pd.DataFrame): DataFrame to save
            file_path (str): Target file path
            **kwargs: Additional arguments for pandas to_* methods
            
        Returns:
            bool: True if save was successful
        """
        try:
            # Ensure directory exists
            directory = os.path.dirname(file_path)
            if directory and not FileManager.ensure_directory(directory):
                return False
            
            # Save based on file extension
            if file_path.endswith('.csv'):
                df.to_csv(file_path, **kwargs)
            elif file_path.endswith('.xlsx'):
                df.to_excel(file_path, **kwargs)
            elif file_path.endswith('.json'):
                df.to_json(file_path, **kwargs)
            elif file_path.endswith('.parquet'):
                df.to_parquet(file_path, **kwargs)
            else:
                logger.error(f"Unsupported file format: {file_path}")
                return False
            
            logger.info(f"DataFrame saved to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving DataFrame to {file_path}: {str(e)}")
            return False
    
    @staticmethod
    def load_dataframe(file_path: str, **kwargs) -> Optional[pd.DataFrame]:
        """
        Load DataFrame from file with error handling.
        
        Args:
            file_path (str): Source file path
            **kwargs: Additional arguments for pandas read_* methods
            
        Returns:
            Optional[pd.DataFrame]: Loaded DataFrame or None if error
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return None
            
            # Load based on file extension
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path, **kwargs)
            elif file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path, **kwargs)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path, **kwargs)
            elif file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path, **kwargs)
            else:
                logger.error(f"Unsupported file format: {file_path}")
                return None
            
            logger.info(f"DataFrame loaded from {file_path}, shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading DataFrame from {file_path}: {str(e)}")
            return None
    
    @staticmethod
    def save_json(data: Dict[str, Any], file_path: str) -> bool:
        """
        Save dictionary as JSON file.
        
        Args:
            data (Dict[str, Any]): Data to save
            file_path (str): Target file path
            
        Returns:
            bool: True if save was successful
        """
        try:
            directory = os.path.dirname(file_path)
            if directory and not FileManager.ensure_directory(directory):
                return False
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"JSON data saved to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving JSON to {file_path}: {str(e)}")
            return False
    
    @staticmethod
    def load_json(file_path: str) -> Optional[Dict[str, Any]]:
        """
        Load JSON file as dictionary.
        
        Args:
            file_path (str): Source file path
            
        Returns:
            Optional[Dict[str, Any]]: Loaded data or None if error
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return None
            
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            logger.info(f"JSON data loaded from {file_path}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading JSON from {file_path}: {str(e)}")
            return None