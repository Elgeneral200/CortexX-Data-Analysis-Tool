"""
Data exploration module for CortexX sales forecasting platform.
Provides statistical analysis and insights generation capabilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from scipy import stats

logger = logging.getLogger(__name__)

class DataExplorer:
    """
    A class to perform comprehensive exploratory data analysis for sales forecasting.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_summary_statistics(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Generate comprehensive summary statistics for the dataset.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            Dict[str, any]: Dictionary containing summary statistics
        """
        try:
            summary = {
                'dataset_shape': df.shape,
                'data_types': dict(df.dtypes.value_counts()),
                'missing_values': df.isnull().sum().to_dict(),
                'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
                'duplicate_rows': df.duplicated().sum(),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
            }
            
            # Numerical columns statistics
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            if not numerical_cols.empty:
                summary['numerical_stats'] = df[numerical_cols].describe().to_dict()
                summary['skewness'] = df[numerical_cols].skew().to_dict()
                summary['kurtosis'] = df[numerical_cols].kurtosis().to_dict()
            
            # Categorical columns statistics
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            if not categorical_cols.empty:
                categorical_stats = {}
                for col in categorical_cols:
                    categorical_stats[col] = {
                        'unique_count': df[col].nunique(),
                        'top_categories': df[col].value_counts().head(5).to_dict(),
                        'missing_count': df[col].isnull().sum()
                    }
                summary['categorical_stats'] = categorical_stats
            
            self.logger.info("Generated comprehensive summary statistics")
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating summary statistics: {str(e)}")
            raise ValueError(f"Summary statistics generation failed: {str(e)}")
    
    def analyze_time_series_patterns(self, df: pd.DataFrame, date_column: str, 
                                   value_column: str) -> Dict[str, any]:
        """
        Analyze time series patterns including trends, seasonality, and stationarity.
        
        Args:
            df (pd.DataFrame): Input dataframe with time series data
            date_column (str): Name of the date column
            value_column (str): Name of the value column
            
        Returns:
            Dict[str, any]: Time series analysis results
        """
        try:
            if date_column not in df.columns or value_column not in df.columns:
                raise ValueError("Required columns not found in dataframe")
            
            # Ensure date column is datetime
            df_ts = df.copy()
            df_ts[date_column] = pd.to_datetime(df_ts[date_column])
            df_ts = df_ts.sort_values(date_column)
            
            analysis = {
                'time_period': {
                    'start_date': df_ts[date_column].min(),
                    'end_date': df_ts[date_column].max(),
                    'total_days': (df_ts[date_column].max() - df_ts[date_column].min()).days,
                    'data_points': len(df_ts)
                },
                'basic_stats': {
                    'mean': df_ts[value_column].mean(),
                    'std': df_ts[value_column].std(),
                    'min': df_ts[value_column].min(),
                    'max': df_ts[value_column].max(),
                    'trend_strength': self._calculate_trend_strength(df_ts, date_column, value_column)
                }
            }
            
            # Seasonality analysis
            analysis['seasonality'] = self._analyze_seasonality(df_ts, date_column, value_column)
            
            # Stationarity test
            analysis['stationarity'] = self._test_stationarity(df_ts[value_column])
            
            # Autocorrelation analysis
            analysis['autocorrelation'] = self._analyze_autocorrelation(df_ts[value_column])
            
            self.logger.info("Completed time series pattern analysis")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing time series patterns: {str(e)}")
            raise ValueError(f"Time series analysis failed: {str(e)}")
    
    def _calculate_trend_strength(self, df: pd.DataFrame, date_column: str, 
                                value_column: str) -> float:
        """Calculate the strength of trend in time series data."""
        try:
            # Convert dates to numerical values for regression
            dates_numeric = (df[date_column] - df[date_column].min()).dt.days.values
            values = df[value_column].values
            
            # Calculate trend using linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(dates_numeric, values)
            return abs(r_value)  # Return absolute correlation coefficient as trend strength
            
        except:
            return 0.0
    
    def _analyze_seasonality(self, df: pd.DataFrame, date_column: str, 
                           value_column: str) -> Dict[str, any]:
        """Analyze seasonal patterns in time series data."""
        try:
            df_seasonal = df.copy()
            df_seasonal['year'] = df_seasonal[date_column].dt.year
            df_seasonal['month'] = df_seasonal[date_column].dt.month
            df_seasonal['day_of_week'] = df_seasonal[date_column].dt.dayofweek
            df_seasonal['week'] = df_seasonal[date_column].dt.isocalendar().week
            
            seasonality = {
                'monthly': df_seasonal.groupby('month')[value_column].mean().to_dict(),
                'weekly': df_seasonal.groupby('day_of_week')[value_column].mean().to_dict(),
                'yearly': df_seasonal.groupby('year')[value_column].mean().to_dict()
            }
            
            # Calculate seasonality strength
            monthly_var = np.array(list(seasonality['monthly'].values())).var()
            seasonality['strength'] = monthly_var / df_seasonal[value_column].var() if df_seasonal[value_column].var() > 0 else 0
            
            return seasonality
            
        except Exception as e:
            self.logger.warning(f"Seasonality analysis partial failure: {str(e)}")
            return {'monthly': {}, 'weekly': {}, 'yearly': {}, 'strength': 0}
    
    def _test_stationarity(self, series: pd.Series, max_lags: int = 10) -> Dict[str, any]:
        """Test stationarity of time series using Augmented Dickey-Fuller test."""
        try:
            from statsmodels.tsa.stattools import adfuller
            
            # Remove NaN values
            series_clean = series.dropna()
            
            if len(series_clean) < max_lags + 1:
                return {'is_stationary': False, 'p_value': 1.0, 'test_statistic': 0}
            
            # Perform ADF test
            result = adfuller(series_clean, maxlag=max_lags)
            
            return {
                'is_stationary': result[1] <= 0.05,
                'p_value': result[1],
                'test_statistic': result[0],
                'critical_values': result[4]
            }
            
        except Exception as e:
            self.logger.warning(f"Stationarity test failed: {str(e)}")
            return {'is_stationary': False, 'p_value': 1.0, 'test_statistic': 0}
    
    def _analyze_autocorrelation(self, series: pd.Series, max_lags: int = 20) -> Dict[str, any]:
        """Analyze autocorrelation in time series data."""
        try:
            from statsmodels.tsa.stattools import acf, pacf
            
            series_clean = series.dropna()
            
            if len(series_clean) < max_lags + 1:
                return {'acf': [], 'pacf': [], 'significant_lags': []}
            
            # Calculate ACF and PACF
            acf_values = acf(series_clean, nlags=max_lags)
            pacf_values = pacf(series_clean, nlags=max_lags)
            
            # Find significant lags (outside 95% confidence interval)
            significant_acf = [i for i, val in enumerate(acf_values) if abs(val) > 1.96/np.sqrt(len(series_clean))]
            significant_pacf = [i for i, val in enumerate(pacf_values) if abs(val) > 1.96/np.sqrt(len(series_clean))]
            
            return {
                'acf': acf_values.tolist(),
                'pacf': pacf_values.tolist(),
                'significant_lags_acf': significant_acf,
                'significant_lags_pacf': significant_pacf
            }
            
        except Exception as e:
            self.logger.warning(f"Autocorrelation analysis failed: {str(e)}")
            return {'acf': [], 'pacf': [], 'significant_lags': []}
    
    def identify_data_issues(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Identify potential data quality issues in the dataset.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            Dict[str, List[str]]: Dictionary of issues by category
        """
        issues = {
            'missing_data': [],
            'outliers': [],
            'inconsistencies': [],
            'data_quality': []
        }
        
        try:
            # Check for missing values
            missing_cols = df.columns[df.isnull().any()].tolist()
            if missing_cols:
                issues['missing_data'].extend([f"Missing values in {col}" for col in missing_cols])
            
            # Check for constant columns
            constant_cols = df.columns[df.nunique() <= 1].tolist()
            if constant_cols:
                issues['data_quality'].extend([f"Constant column: {col}" for col in constant_cols])
            
            # Check for high cardinality categorical columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                if df[col].nunique() > 100:  # Arbitrary threshold
                    issues['data_quality'].append(f"High cardinality in {col}: {df[col].nunique()} unique values")
            
            # Check for potential outliers in numerical columns
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outlier_count = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
                if outlier_count > 0:
                    issues['outliers'].append(f"Potential outliers in {col}: {outlier_count} values")
            
            self.logger.info(f"Identified {sum(len(v) for v in issues.values())} data issues")
            return issues
            
        except Exception as e:
            self.logger.error(f"Error identifying data issues: {str(e)}")
            issues['inconsistencies'].append(f"Error during issue identification: {str(e)}")
            return issues