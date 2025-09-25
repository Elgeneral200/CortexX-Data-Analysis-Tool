"""
Feature selection module for CortexX sales forecasting platform.
Handles feature importance calculation and feature selection.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import StandardScaler
import warnings

logger = logging.getLogger(__name__)

class FeatureSelector:
    """
    A class to handle feature selection and importance calculation.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        warnings.filterwarnings('ignore')
    
    def calculate_feature_importance(self, df: pd.DataFrame, target_col: str, 
                                   method: str = 'random_forest') -> pd.DataFrame:
        """
        Calculate feature importance using specified method.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_col (str): Target column name
            method (str): Method for importance calculation
            
        Returns:
            pd.DataFrame: DataFrame with feature importance scores
        """
        try:
            # Prepare data
            X = df.drop(columns=[target_col], errors='ignore')
            y = df[target_col]
            
            # Remove non-numeric columns for importance calculation
            X_numeric = X.select_dtypes(include=[np.number])
            
            if X_numeric.empty:
                self.logger.warning("No numeric features found for importance calculation")
                return pd.DataFrame(columns=['feature', 'importance'])
            
            # Handle missing values
            X_numeric = X_numeric.fillna(X_numeric.mean())
            y = y.fillna(y.mean())
            
            if method == 'random_forest':
                importance_scores = self._random_forest_importance(X_numeric, y)
            elif method == 'f_regression':
                importance_scores = self._f_regression_importance(X_numeric, y)
            elif method == 'mutual_info':
                importance_scores = self._mutual_info_importance(X_numeric, y)
            else:
                self.logger.warning(f"Unknown method {method}, using random forest")
                importance_scores = self._random_forest_importance(X_numeric, y)
            
            # Create results dataframe
            importance_df = pd.DataFrame({
                'feature': X_numeric.columns,
                'importance': importance_scores
            }).sort_values('importance', ascending=False)
            
            self.logger.info(f"Calculated feature importance using {method} method")
            return importance_df
            
        except Exception as e:
            self.logger.error(f"Error calculating feature importance: {str(e)}")
            return pd.DataFrame(columns=['feature', 'importance'])
    
    def _random_forest_importance(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Calculate feature importance using Random Forest."""
        try:
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X, y)
            return rf.feature_importances_
        except:
            return np.zeros(X.shape[1])
    
    def _f_regression_importance(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Calculate feature importance using F-regression."""
        try:
            f_scores, _ = f_regression(X, y)
            return f_scores / np.sum(f_scores)  # Normalize to sum to 1
        except:
            return np.zeros(X.shape[1])
    
    def _mutual_info_importance(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Calculate feature importance using mutual information."""
        try:
            mi_scores = mutual_info_regression(X, y, random_state=42)
            return mi_scores / np.sum(mi_scores)  # Normalize to sum to 1
        except:
            return np.zeros(X.shape[1])
    
    def select_features(self, df: pd.DataFrame, target_col: str, 
                       n_features: int = 20, method: str = 'random_forest') -> List[str]:
        """
        Select top n features based on importance.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_col (str): Target column name
            n_features (int): Number of features to select
            method (str): Feature selection method
            
        Returns:
            List[str]: List of selected feature names
        """
        try:
            importance_df = self.calculate_feature_importance(df, target_col, method)
            
            if importance_df.empty:
                self.logger.warning("No features selected due to empty importance results")
                return []
            
            # Select top n features
            selected_features = importance_df.head(n_features)['feature'].tolist()
            
            self.logger.info(f"Selected top {len(selected_features)} features")
            return selected_features
            
        except Exception as e:
            self.logger.error(f"Error selecting features: {str(e)}")
            return []
    
    def remove_correlated_features(self, df: pd.DataFrame, threshold: float = 0.95) -> List[str]:
        """
        Remove highly correlated features to reduce multicollinearity.
        
        Args:
            df (pd.DataFrame): Input dataframe
            threshold (float): Correlation threshold for removal
            
        Returns:
            List[str]: List of features to keep
        """
        try:
            # Select only numeric columns
            numeric_df = df.select_dtypes(include=[np.number])
            
            if numeric_df.empty:
                return df.columns.tolist()
            
            # Calculate correlation matrix
            corr_matrix = numeric_df.corr().abs()
            
            # Create a mask to identify highly correlated features
            upper_tri = corr_matrix.where(
                np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            )
            
            # Find features with correlation greater than threshold
            to_drop = [
                column for column in upper_tri.columns 
                if any(upper_tri[column] > threshold)
            ]
            
            # Features to keep
            features_to_keep = [col for col in numeric_df.columns if col not in to_drop]
            
            # Add non-numeric columns back
            non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
            features_to_keep.extend(non_numeric_cols)
            
            self.logger.info(f"Removed {len(to_drop)} highly correlated features")
            return features_to_keep
            
        except Exception as e:
            self.logger.error(f"Error removing correlated features: {str(e)}")
            return df.columns.tolist()
    
    def create_feature_selection_report(self, df: pd.DataFrame, target_col: str) -> Dict[str, any]:
        """
        Create comprehensive feature selection report.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_col (str): Target column name
            
        Returns:
            Dict[str, any]: Feature selection report
        """
        try:
            report = {}
            
            # Basic information
            report['total_features'] = len(df.columns) - 1  # Exclude target
            report['numeric_features'] = len(df.select_dtypes(include=[np.number]).columns) - 1
            report['categorical_features'] = len(df.select_dtypes(include=['object', 'category']).columns)
            
            # Feature importance using different methods
            methods = ['random_forest', 'f_regression', 'mutual_info']
            importance_results = {}
            
            for method in methods:
                importance_df = self.calculate_feature_importance(df, target_col, method)
                importance_results[method] = importance_df.to_dict('records')
            
            report['importance_results'] = importance_results
            
            # Correlation analysis
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                report['correlation_with_target'] = corr_matrix[target_col].sort_values(
                    key=abs, ascending=False
                ).to_dict()
            else:
                report['correlation_with_target'] = {}
            
            # Recommended features
            recommended_features = self.select_features(df, target_col, n_features=15)
            report['recommended_features'] = recommended_features
            
            # Multicollinearity check
            features_to_keep = self.remove_correlated_features(df)
            report['features_after_correlation'] = features_to_keep
            
            self.logger.info("Created comprehensive feature selection report")
            return report
            
        except Exception as e:
            self.logger.error(f"Error creating feature selection report: {str(e)}")
            return {'error': str(e)}