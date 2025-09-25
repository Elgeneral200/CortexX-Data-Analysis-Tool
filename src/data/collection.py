"""
Data collection module for CortexX sales forecasting platform.
Handles data acquisition from multiple sources including CSV, databases, and APIs.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, Dict, Any
import logging
from io import StringIO
import re

logger = logging.getLogger(__name__)

class DataCollector:
    """
    A class to handle data collection from various sources for sales forecasting.
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize DataCollector with configuration.
        
        Args:
            config (dict, optional): Configuration dictionary for data sources
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
    
    def load_csv_data(self, file_path: Union[str, object], **kwargs) -> pd.DataFrame:
        """
        Load sales data from CSV file or file-like object with automatic date detection.
        
        Args:
            file_path (Union[str, object]): Path to CSV file or file-like object
            **kwargs: Additional arguments for pandas read_csv
            
        Returns:
            pd.DataFrame: Loaded sales data with auto-detected date columns
        """
        try:
            self.logger.info(f"Loading data from {file_path}")
            
            if isinstance(file_path, str):
                df = pd.read_csv(file_path, **kwargs)
            else:
                # Handle file-like objects (Streamlit uploads)
                df = pd.read_csv(file_path, **kwargs)
            
            # Auto-detect and convert date columns
            df = self._auto_detect_dates(df)
            
            self.logger.info(f"Successfully loaded data with shape {df.shape}")
            return df
            
        except FileNotFoundError as e:
            self.logger.error(f"CSV file not found: {file_path}")
            raise e
        except Exception as e:
            self.logger.error(f"Error loading CSV data: {str(e)}")
            raise ValueError(f"Failed to load data: {str(e)}")
    
    def _auto_detect_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Automatically detect and convert date columns in the dataframe.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with date columns converted
        """
        df_processed = df.copy()
        date_columns_found = []
        
        # Common date column patterns (case insensitive)
        date_patterns = [
            r'date', r'time', r'day', r'month', r'year', r'timestamp',
            r'created', r'modified', r'period', r'week', r'quarter'
        ]
        
        for col in df_processed.columns:
            col_lower = col.lower()
            
            # Check if column name matches date patterns
            is_date_like = any(pattern in col_lower for pattern in date_patterns)
            
            if is_date_like:
                # Try to convert to datetime
                converted_col, success = self._try_convert_to_datetime(df_processed[col])
                if success:
                    df_processed[col] = converted_col
                    date_columns_found.append(col)
                    self.logger.info(f"Auto-converted column '{col}' to datetime")
                    continue
            
            # If not detected by name, check content
            if not is_date_like and df_processed[col].dtype == 'object':
                converted_col, success = self._try_convert_to_datetime(df_processed[col])
                if success and self._is_likely_date_column(converted_col):
                    df_processed[col] = converted_col
                    date_columns_found.append(col)
                    self.logger.info(f"Auto-detected date column '{col}' from content")
        
        if date_columns_found:
            self.logger.info(f"Found date columns: {date_columns_found}")
        else:
            self.logger.warning("No date columns detected automatically")
            
        return df_processed
    
    def _try_convert_to_datetime(self, series: pd.Series) -> tuple:
        """
        Try to convert a series to datetime using multiple methods.
        
        Args:
            series (pd.Series): Series to convert
            
        Returns:
            tuple: (converted_series, success_flag)
        """
        original_dtype = series.dtype
        
        # If already datetime, return as is
        if pd.api.types.is_datetime64_any_dtype(series):
            return series, True
        
        # Try direct conversion
        try:
            converted = pd.to_datetime(series, errors='coerce')
            # Check if conversion was successful (not all NaT)
            if not converted.isna().all():
                return converted, True
        except:
            pass
        
        # Try with different date formats
        date_formats = [
            '%Y-%m-%d', '%d-%m-%Y', '%m/%d/%Y', '%Y/%m/%d',
            '%d/%m/%Y', '%m-%d-%Y', '%Y%m%d', '%d%m%Y',
            '%b %d, %Y', '%d %b %Y', '%B %d, %Y', '%d %B %Y'
        ]
        
        for fmt in date_formats:
            try:
                converted = pd.to_datetime(series, format=fmt, errors='coerce')
                if not converted.isna().all():
                    return converted, True
            except:
                continue
        
        # Try inferring datetime
        try:
            converted = pd.to_datetime(series, infer_datetime_format=True, errors='coerce')
            if not converted.isna().all():
                return converted, True
        except:
            pass
        
        return series, False
    
    def _is_likely_date_column(self, series: pd.Series, threshold: float = 0.8) -> bool:
        """
        Check if a series is likely a date column based on content patterns.
        
        Args:
            series (pd.Series): Series to check
            threshold (float): Minimum proportion of valid dates required
            
        Returns:
            bool: True if likely a date column
        """
        if not pd.api.types.is_datetime64_any_dtype(series):
            return False
        
        # Check if most values are valid dates
        valid_dates_ratio = 1 - series.isna().mean()
        
        # Check for reasonable date range (not all same date, not too wide range)
        if valid_dates_ratio > threshold:
            unique_dates = series.dropna().nunique()
            date_range = series.dropna().max() - series.dropna().min()
            
            # Reasonable criteria for a date column
            if unique_dates > 1 and date_range.days > 0:
                return True
        
        return False
    
    def generate_sample_data(self, periods: int = 365*3, products: int = 3) -> pd.DataFrame:
        """
        Generate synthetic sales data for demonstration purposes.
        
        Args:
            periods (int): Number of time periods to generate
            products (int): Number of different products to generate
            
        Returns:
            pd.DataFrame: Synthetic sales data with realistic patterns
        """
        try:
            self.logger.info(f"Generating sample data for {periods} periods and {products} products")
            
            dates = pd.date_range(start='2020-01-01', periods=periods, freq='D')
            product_ids = [f'Product_{chr(65+i)}' for i in range(products)]
            
            data = []
            for date in dates:
                for product_id in product_ids:
                    # Base trend with slight upward trajectory
                    trend = 100 + (date - dates[0]).days * 0.1
                    
                    # Seasonal patterns
                    seasonal = 50 * np.sin(2 * np.pi * (date.dayofyear - 1) / 365)
                    
                    # Weekly pattern
                    weekly = 20 * np.sin(2 * np.pi * date.dayofweek / 7)
                    
                    # Product-specific variations
                    product_factor = ord(product_id[-1]) - 64  # A=1, B=2, etc.
                    product_base = trend * (0.8 + 0.4 * product_factor / len(product_ids))
                    
                    # Random noise
                    noise = np.random.normal(0, 15)
                    
                    # Promotional effects (random)
                    promotion = np.random.choice([0, 1], p=[0.85, 0.15])
                    promo_effect = 40 if promotion else 0
                    
                    sales = max(0, product_base + seasonal + weekly + noise + promo_effect)
                    
                    data.append({
                        'date': date,
                        'product_id': product_id,
                        'sales': sales,
                        'price': np.random.uniform(10, 100),
                        'promotion': promotion,
                        'category': f'Category_{(ord(product_id[-1]) - 65) % 3 + 1}'
                    })
            
            df = pd.DataFrame(data)
            self.logger.info(f"Generated sample data with shape {df.shape}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating sample data: {str(e)}")
            raise ValueError(f"Failed to generate sample data: {str(e)}")
    
    def validate_data_structure(self, df: pd.DataFrame, required_columns: list = None) -> Dict[str, Any]:
        """
        Validate the structure and quality of the loaded data.
        
        Args:
            df (pd.DataFrame): Dataframe to validate
            required_columns (list, optional): List of required columns
            
        Returns:
            Dict[str, Any]: Validation results
        """
        validation_result = {
            'is_valid': True,
            'issues': [],
            'summary': {},
            'date_columns': []
        }
        
        try:
            # Check for empty dataframe
            if df.empty:
                validation_result['is_valid'] = False
                validation_result['issues'].append("Dataframe is empty")
                return validation_result
            
            # Find date columns
            date_columns = df.select_dtypes(include=['datetime64']).columns.tolist()
            validation_result['date_columns'] = date_columns
            
            # Basic statistics
            validation_result['summary'] = {
                'shape': df.shape,
                'columns': list(df.columns),
                'data_types': dict(df.dtypes),
                'missing_values': df.isnull().sum().to_dict(),
                'duplicate_rows': df.duplicated().sum(),
                'date_columns_found': date_columns
            }
            
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
                validation_result['issues'].append(f"High missing values in: {high_missing}")
            
            # Warn if no date columns found
            if not date_columns:
                validation_result['issues'].append("No date columns detected. Time series analysis may be limited.")
            
            self.logger.info("Data validation completed")
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Error during data validation: {str(e)}")
            validation_result['is_valid'] = False
            validation_result['issues'].append(f"Validation error: {str(e)}")
            return validation_result