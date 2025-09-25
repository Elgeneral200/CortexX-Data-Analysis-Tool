"""
Advanced feature engineering module for CortexX sales forecasting platform.
Creates time-based features, lag features, and rolling statistics.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
import logging
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    A class to handle advanced feature engineering for time series forecasting.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create_time_features(self, df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """
        Create comprehensive time-based features from datetime column.
        
        Args:
            df (pd.DataFrame): Input dataframe
            date_col (str): Date column name
            
        Returns:
            pd.DataFrame: Dataframe with time features added
        """
        try:
            df_eng = df.copy()
            
            # Ensure date column is datetime
            df_eng[date_col] = pd.to_datetime(df_eng[date_col])
            
            # Basic time features
            df_eng['year'] = df_eng[date_col].dt.year
            df_eng['month'] = df_eng[date_col].dt.month
            df_eng['quarter'] = df_eng[date_col].dt.quarter
            df_eng['week'] = df_eng[date_col].dt.isocalendar().week.astype(int)
            df_eng['day_of_week'] = df_eng[date_col].dt.dayofweek
            df_eng['day_of_year'] = df_eng[date_col].dt.dayofyear
            df_eng['is_weekend'] = df_eng['day_of_week'].isin([5, 6]).astype(int)
            df_eng['is_month_start'] = df_eng[date_col].dt.is_month_start.astype(int)
            df_eng['is_month_end'] = df_eng[date_col].dt.is_month_end.astype(int)
            df_eng['is_quarter_start'] = df_eng[date_col].dt.is_quarter_start.astype(int)
            df_eng['is_quarter_end'] = df_eng[date_col].dt.is_quarter_end.astype(int)
            df_eng['is_year_start'] = df_eng[date_col].dt.is_year_start.astype(int)
            df_eng['is_year_end'] = df_eng[date_col].dt.is_year_end.astype(int)
            
            self.logger.info("Created basic time features")
            return df_eng
            
        except Exception as e:
            self.logger.error(f"Error creating time features: {str(e)}")
            raise ValueError(f"Time feature creation failed: {str(e)}")
    
    def create_lag_features(self, df: pd.DataFrame, value_col: str, 
                           lags: List[int] = [1, 7, 30, 90]) -> pd.DataFrame:
        """
        Create lag features for time series forecasting.
        
        Args:
            df (pd.DataFrame): Input dataframe
            value_col (str): Value column to create lags for
            lags (List[int]): List of lag periods
            
        Returns:
            pd.DataFrame: Dataframe with lag features added
        """
        try:
            df_lags = df.copy()
            
            # Sort by date if available to ensure proper lag calculation
            date_cols = df_lags.select_dtypes(include=['datetime64']).columns
            if len(date_cols) > 0:
                df_lags = df_lags.sort_values(date_cols[0])
            
            for lag in lags:
                df_lags[f'{value_col}_lag_{lag}'] = df_lags[value_col].shift(lag)
            
            # Create lag differences (change from previous period)
            for lag in [1, 7, 30]:
                if f'{value_col}_lag_{lag}' in df_lags.columns:
                    df_lags[f'{value_col}_diff_{lag}'] = (
                        df_lags[value_col] - df_lags[f'{value_col}_lag_{lag}']
                    )
                    df_lags[f'{value_col}_pct_change_{lag}'] = (
                        (df_lags[value_col] - df_lags[f'{value_col}_lag_{lag}']) / 
                        df_lags[f'{value_col}_lag_{lag}'].replace(0, np.nan)
                    )
            
            self.logger.info(f"Created lag features for lags: {lags}")
            return df_lags
            
        except Exception as e:
            self.logger.error(f"Error creating lag features: {str(e)}")
            raise ValueError(f"Lag feature creation failed: {str(e)}")
    
    def create_rolling_features(self, df: pd.DataFrame, value_col: str,
                               windows: List[int] = [7, 30, 90]) -> pd.DataFrame:
        """
        Create rolling statistics features.
        
        Args:
            df (pd.DataFrame): Input dataframe
            value_col (str): Value column for rolling stats
            windows (List[int]): Rolling window sizes
            
        Returns:
            pd.DataFrame: Dataframe with rolling features added
        """
        try:
            df_roll = df.copy()
            
            # Sort by date if available
            date_cols = df_roll.select_dtypes(include=['datetime64']).columns
            if len(date_cols) > 0:
                df_roll = df_roll.sort_values(date_cols[0])
            
            for window in windows:
                # Basic rolling statistics
                df_roll[f'{value_col}_roll_mean_{window}'] = (
                    df_roll[value_col].rolling(window=window, min_periods=1).mean()
                )
                df_roll[f'{value_col}_roll_std_{window}'] = (
                    df_roll[value_col].rolling(window=window, min_periods=1).std()
                )
                df_roll[f'{value_col}_roll_min_{window}'] = (
                    df_roll[value_col].rolling(window=window, min_periods=1).min()
                )
                df_roll[f'{value_col}_roll_max_{window}'] = (
                    df_roll[value_col].rolling(window=window, min_periods=1).max()
                )
                df_roll[f'{value_col}_roll_median_{window}'] = (
                    df_roll[value_col].rolling(window=window, min_periods=1).median()
                )
                
                # Rolling volatility
                df_roll[f'{value_col}_roll_volatility_{window}'] = (
                    df_roll[value_col].rolling(window=window, min_periods=1).std() / 
                    df_roll[value_col].rolling(window=window, min_periods=1).mean()
                )
                
                # Exponential moving averages
                df_roll[f'{value_col}_ema_{window}'] = (
                    df_roll[value_col].ewm(span=window, min_periods=1).mean()
                )
            
            self.logger.info(f"Created rolling features for windows: {windows}")
            return df_roll
            
        except Exception as e:
            self.logger.error(f"Error creating rolling features: {str(e)}")
            raise ValueError(f"Rolling feature creation failed: {str(e)}")
    
    def encode_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode cyclical features (like months, days) using sine/cosine transformation.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with cyclical features encoded
        """
        try:
            df_cyclic = df.copy()
            
            # Monthly cyclical encoding
            if 'month' in df_cyclic.columns:
                df_cyclic['month_sin'] = np.sin(2 * np.pi * df_cyclic['month'] / 12)
                df_cyclic['month_cos'] = np.cos(2 * np.pi * df_cyclic['month'] / 12)
            
            # Day of week cyclical encoding
            if 'day_of_week' in df_cyclic.columns:
                df_cyclic['day_sin'] = np.sin(2 * np.pi * df_cyclic['day_of_week'] / 7)
                df_cyclic['day_cos'] = np.cos(2 * np.pi * df_cyclic['day_of_week'] / 7)
            
            # Day of year cyclical encoding
            if 'day_of_year' in df_cyclic.columns:
                df_cyclic['day_of_year_sin'] = np.sin(2 * np.pi * df_cyclic['day_of_year'] / 365)
                df_cyclic['day_of_year_cos'] = np.cos(2 * np.pi * df_cyclic['day_of_year'] / 365)
            
            # Hour cyclical encoding (if applicable)
            if 'hour' in df_cyclic.columns:
                df_cyclic['hour_sin'] = np.sin(2 * np.pi * df_cyclic['hour'] / 24)
                df_cyclic['hour_cos'] = np.cos(2 * np.pi * df_cyclic['hour'] / 24)
            
            self.logger.info("Created cyclical feature encodings")
            return df_cyclic
            
        except Exception as e:
            self.logger.error(f"Error encoding cyclical features: {str(e)}")
            raise ValueError(f"Cyclical feature encoding failed: {str(e)}")
    
    def create_interaction_features(self, df: pd.DataFrame, 
                                  feature_pairs: List[tuple]) -> pd.DataFrame:
        """
        Create interaction features between pairs of features.
        
        Args:
            df (pd.DataFrame): Input dataframe
            feature_pairs (List[tuple]): List of feature pairs to create interactions for
            
        Returns:
            pd.DataFrame: Dataframe with interaction features added
        """
        try:
            df_interact = df.copy()
            
            for feat1, feat2 in feature_pairs:
                if feat1 in df_interact.columns and feat2 in df_interact.columns:
                    # Multiplication interaction
                    df_interact[f'{feat1}_x_{feat2}'] = df_interact[feat1] * df_interact[feat2]
                    
                    # Ratio interaction (avoid division by zero)
                    if (df_interact[feat2] != 0).all():
                        df_interact[f'{feat1}_div_{feat2}'] = df_interact[feat1] / df_interact[feat2]
                    else:
                        df_interact[f'{feat1}_div_{feat2}'] = np.where(
                            df_interact[feat2] != 0, 
                            df_interact[feat1] / df_interact[feat2], 
                            np.nan
                        )
            
            self.logger.info(f"Created interaction features for {len(feature_pairs)} pairs")
            return df_interact
            
        except Exception as e:
            self.logger.error(f"Error creating interaction features: {str(e)}")
            raise ValueError(f"Interaction feature creation failed: {str(e)}")
    
    def create_fourier_features(self, df: pd.DataFrame, date_col: str, 
                               value_col: str, periods: List[int] = [365, 30, 7]) -> pd.DataFrame:
        """
        Create Fourier features for capturing seasonal patterns.
        
        Args:
            df (pd.DataFrame): Input dataframe
            date_col (str): Date column name
            value_col (str): Value column name
            periods (List[int]): Seasonal periods for Fourier terms
            
        Returns:
            pd.DataFrame: Dataframe with Fourier features added
        """
        try:
            df_fourier = df.copy()
            df_fourier[date_col] = pd.to_datetime(df_fourier[date_col])
            
            # Create time index
            df_fourier = df_fourier.sort_values(date_col)
            time_index = (df_fourier[date_col] - df_fourier[date_col].min()).dt.days
            
            for period in periods:
                # Fourier sine and cosine terms
                df_fourier[f'fourier_sin_{period}'] = np.sin(2 * np.pi * time_index / period)
                df_fourier[f'fourier_cos_{period}'] = np.cos(2 * np.pi * time_index / period)
            
            self.logger.info(f"Created Fourier features for periods: {periods}")
            return df_fourier
            
        except Exception as e:
            self.logger.error(f"Error creating Fourier features: {str(e)}")
            raise ValueError(f"Fourier feature creation failed: {str(e)}")