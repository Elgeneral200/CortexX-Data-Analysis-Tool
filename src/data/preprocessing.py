"""
Data preprocessing module for CortexX sales forecasting platform.
Handles data cleaning, missing value treatment, and feature encoding.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    A class to handle comprehensive data preprocessing for sales forecasting.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'interpolate', 
                            columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Handle missing values in the dataset using specified strategy.
        
        Args:
            df (pd.DataFrame): Input dataframe
            strategy (str): Strategy for handling missing values
            columns (List[str], optional): Specific columns to process
            
        Returns:
            pd.DataFrame: Dataframe with handled missing values
        """
        try:
            df_clean = df.copy()
            columns_to_process = columns if columns else df_clean.columns
            
            for column in columns_to_process:
                if column not in df_clean.columns:
                    continue
                    
                if strategy == 'interpolate':
                    if pd.api.types.is_numeric_dtype(df_clean[column]):
                        df_clean[column] = df_clean[column].interpolate(method='linear')
                    else:
                        df_clean[column] = df_clean[column].fillna(method='ffill')
                elif strategy == 'ffill':
                    df_clean[column] = df_clean[column].fillna(method='ffill')
                elif strategy == 'bfill':
                    df_clean[column] = df_clean[column].fillna(method='bfill')
                elif strategy == 'mean':
                    if pd.api.types.is_numeric_dtype(df_clean[column]):
                        df_clean[column] = df_clean[column].fillna(df_clean[column].mean())
                elif strategy == 'median':
                    if pd.api.types.is_numeric_dtype(df_clean[column]):
                        df_clean[column] = df_clean[column].fillna(df_clean[column].median())
                elif strategy == 'drop':
                    df_clean = df_clean.dropna(subset=[column])
                else:
                    self.logger.warning(f"Unknown strategy: {strategy}. Using forward fill.")
                    df_clean[column] = df_clean[column].fillna(method='ffill')
            
            self.logger.info(f"Handled missing values using {strategy} strategy")
            return df_clean
            
        except Exception as e:
            self.logger.error(f"Error handling missing values: {str(e)}")
            raise ValueError(f"Missing value handling failed: {str(e)}")
    
    def remove_outliers(self, df: pd.DataFrame, column: str, method: str = 'iqr', 
                       threshold: float = 1.5) -> pd.DataFrame:
        """
        Remove outliers from specified column using various methods.
        
        Args:
            df (pd.DataFrame): Input dataframe
            column (str): Column name to process
            method (str): Method for outlier detection
            threshold (float): Threshold for outlier detection
            
        Returns:
            pd.DataFrame: Dataframe with outliers removed
        """
        try:
            if column not in df.columns:
                self.logger.warning(f"Column {column} not found in dataframe")
                return df.copy()
            
            df_clean = df.copy()
            
            if method == 'iqr':
                Q1 = df_clean[column].quantile(0.25)
                Q3 = df_clean[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                initial_count = len(df_clean)
                df_clean = df_clean[
                    (df_clean[column] >= lower_bound) & 
                    (df_clean[column] <= upper_bound)
                ]
                removed_count = initial_count - len(df_clean)
                
            elif method == 'zscore':
                from scipy import stats
                z_scores = np.abs(stats.zscore(df_clean[column]))
                initial_count = len(df_clean)
                df_clean = df_clean[z_scores < threshold]
                removed_count = initial_count - len(df_clean)
                
            elif method == 'percentile':
                lower_bound = df_clean[column].quantile(0.01)
                upper_bound = df_clean[column].quantile(0.99)
                initial_count = len(df_clean)
                df_clean = df_clean[
                    (df_clean[column] >= lower_bound) & 
                    (df_clean[column] <= upper_bound)
                ]
                removed_count = initial_count - len(df_clean)
            else:
                self.logger.warning(f"Unknown outlier method: {method}")
                return df_clean
            
            self.logger.info(f"Removed {removed_count} outliers from {column} using {method} method")
            return df_clean
            
        except Exception as e:
            self.logger.error(f"Error removing outliers: {str(e)}")
            raise ValueError(f"Outlier removal failed: {str(e)}")
    
    def encode_categorical_variables(self, df: pd.DataFrame, columns: Optional[List[str]] = None, 
                                   method: str = 'onehot') -> pd.DataFrame:
        """
        Encode categorical variables using specified encoding method.
        
        Args:
            df (pd.DataFrame): Input dataframe
            columns (List[str], optional): Categorical columns to encode
            method (str): Encoding method ('onehot', 'label', 'target')
            
        Returns:
            pd.DataFrame: Dataframe with encoded categorical variables
        """
        try:
            df_encoded = df.copy()
            
            if columns is None:
                # Auto-detect categorical columns
                categorical_columns = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
            else:
                categorical_columns = [col for col in columns if col in df_encoded.columns]
            
            if not categorical_columns:
                self.logger.info("No categorical columns found for encoding")
                return df_encoded
            
            if method == 'onehot':
                df_encoded = pd.get_dummies(df_encoded, columns=categorical_columns, prefix=categorical_columns)
            elif method == 'label':
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                for col in categorical_columns:
                    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            else:
                self.logger.warning(f"Unknown encoding method: {method}. Using one-hot encoding.")
                df_encoded = pd.get_dummies(df_encoded, columns=categorical_columns, prefix=categorical_columns)
            
            self.logger.info(f"Encoded {len(categorical_columns)} categorical columns using {method} encoding")
            return df_encoded
            
        except Exception as e:
            self.logger.error(f"Error encoding categorical variables: {str(e)}")
            raise ValueError(f"Categorical encoding failed: {str(e)}")
    
    def normalize_data(self, df: pd.DataFrame, columns: Optional[List[str]] = None,
                      method: str = 'standardize') -> pd.DataFrame:
        """
        Normalize numerical data using specified method.
        
        Args:
            df (pd.DataFrame): Input dataframe
            columns (List[str], optional): Numerical columns to normalize
            method (str): Normalization method ('standardize', 'minmax', 'robust')
            
        Returns:
            pd.DataFrame: Dataframe with normalized numerical data
        """
        try:
            from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
            
            df_normalized = df.copy()
            
            if columns is None:
                numerical_columns = df_normalized.select_dtypes(include=[np.number]).columns.tolist()
            else:
                numerical_columns = [col for col in columns if col in df_normalized.columns and 
                                   pd.api.types.is_numeric_dtype(df_normalized[col])]
            
            if not numerical_columns:
                self.logger.info("No numerical columns found for normalization")
                return df_normalized
            
            if method == 'standardize':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            elif method == 'robust':
                scaler = RobustScaler()
            else:
                self.logger.warning(f"Unknown normalization method: {method}. Using standard scaling.")
                scaler = StandardScaler()
            
            df_normalized[numerical_columns] = scaler.fit_transform(df_normalized[numerical_columns])
            
            self.logger.info(f"Normalized {len(numerical_columns)} numerical columns using {method} method")
            return df_normalized
            
        except Exception as e:
            self.logger.error(f"Error normalizing data: {str(e)}")
            raise ValueError(f"Data normalization failed: {str(e)}")