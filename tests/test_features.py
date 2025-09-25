"""
Unit tests for features modules of CortexX sales forecasting platform.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.features.engineering import FeatureEngineer
from src.features.selection import FeatureSelector

class TestFeatureEngineer:
    """Test cases for FeatureEngineer class."""
    
    def setup_method(self):
        """Set up test data."""
        self.engineer = FeatureEngineer()
        self.test_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100),
            'sales': np.random.normal(100, 20, 100),
            'price': np.random.uniform(10, 50, 100)
        })
    
    def test_create_time_features(self):
        """Test time feature creation."""
        engineered = self.engineer.create_time_features(self.test_data, 'date')
        
        # Check that new time features were added
        expected_features = ['year', 'month', 'quarter', 'day_of_week', 'is_weekend']
        for feature in expected_features:
            assert feature in engineered.columns
        
        # Verify data types
        assert pd.api.types.is_numeric_dtype(engineered['year'])
        assert pd.api.types.is_numeric_dtype(engineered['month'])
    
    def test_create_lag_features(self):
        """Test lag feature creation."""
        engineered = self.engineer.create_lag_features(self.test_data, 'sales', [1, 7])
        
        # Check that lag features were created
        assert 'sales_lag_1' in engineered.columns
        assert 'sales_lag_7' in engineered.columns
        
        # Check that difference features were created
        assert 'sales_diff_1' in engineered.columns
        assert 'sales_pct_change_1' in engineered.columns
        
        # Verify lag calculations
        assert engineered['sales_lag_1'].iloc[1] == engineered['sales'].iloc[0]
    
    def test_create_rolling_features(self):
        """Test rolling feature creation."""
        engineered = self.engineer.create_rolling_features(self.test_data, 'sales', [7, 30])
        
        # Check that rolling features were created
        expected_features = [
            'sales_roll_mean_7', 'sales_roll_std_7',
            'sales_roll_mean_30', 'sales_roll_std_30'
        ]
        
        for feature in expected_features:
            assert feature in engineered.columns
        
        # Verify rolling calculations
        assert not pd.isna(engineered['sales_roll_mean_7'].iloc[6])  # First complete window
    
    def test_encode_cyclical_features(self):
        """Test cyclical feature encoding."""
        # First create time features
        with_time_features = self.engineer.create_time_features(self.test_data, 'date')
        cyclical_encoded = self.engineer.encode_cyclical_features(with_time_features)
        
        # Check that cyclical features were created
        assert 'month_sin' in cyclical_encoded.columns
        assert 'month_cos' in cyclical_encoded.columns
        assert 'day_sin' in cyclical_encoded.columns
        assert 'day_cos' in cyclical_encoded.columns
        
        # Verify cyclical properties (values between -1 and 1)
        assert cyclical_encoded['month_sin'].between(-1, 1).all()
        assert cyclical_encoded['month_cos'].between(-1, 1).all()

class TestFeatureSelector:
    """Test cases for FeatureSelector class."""
    
    def setup_method(self):
        """Set up test data."""
        self.selector = FeatureSelector()
        
        # Create realistic test data with correlations
        np.random.seed(42)
        n_samples = 100
        
        self.test_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=n_samples),
            'target': np.random.normal(100, 20, n_samples),
            'feature1': np.random.normal(50, 10, n_samples),
            'feature2': np.random.normal(30, 5, n_samples),
            'feature3': np.random.normal(70, 15, n_samples),
            'categorical': np.random.choice(['A', 'B', 'C'], n_samples)
        })
        
        # Create some correlation with target
        self.test_data['feature1'] = self.test_data['target'] * 0.7 + np.random.normal(0, 5, n_samples)
    
    def test_calculate_feature_importance(self):
        """Test feature importance calculation."""
        importance_df = self.selector.calculate_feature_importance(
            self.test_data, 'target', method='random_forest'
        )
        
        assert isinstance(importance_df, pd.DataFrame)
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
        
        # Should have importance scores for numeric features
        numeric_features = self.test_data.select_dtypes(include=[np.number]).columns
        numeric_features = [f for f in numeric_features if f != 'target']
        assert len(importance_df) == len(numeric_features)
        
        # Importance scores should be positive and sum to approximately 1
        assert importance_df['importance'].sum() == pytest.approx(1.0, abs=0.1)
    
    def test_select_features(self):
        """Test feature selection."""
        selected_features = self.selector.select_features(
            self.test_data, 'target', n_features=2, method='random_forest'
        )
        
        assert isinstance(selected_features, list)
        assert len(selected_features) == 2
        
        # Selected features should be from the original dataset
        for feature in selected_features:
            assert feature in self.test_data.columns
            assert feature != 'target'
    
    def test_remove_correlated_features(self):
        """Test correlated feature removal."""
        # Add highly correlated feature
        self.test_data['feature1_highly_correlated'] = self.test_data['feature1'] + np.random.normal(0, 0.1, len(self.test_data))
        
        features_to_keep = self.selector.remove_correlated_features(self.test_data, threshold=0.95)
        
        assert isinstance(features_to_keep, list)
        
        # The highly correlated feature should be removed
        assert 'feature1_highly_correlated' not in features_to_keep
        assert 'feature1' in features_to_keep  # Original should be kept
    
    def test_create_feature_selection_report(self):
        """Test feature selection report generation."""
        report = self.selector.create_feature_selection_report(self.test_data, 'target')
        
        assert isinstance(report, dict)
        assert 'total_features' in report
        assert 'importance_results' in report
        assert 'recommended_features' in report
        
        # Report should contain results from multiple methods
        assert 'random_forest' in report['importance_results']
        assert 'f_regression' in report['importance_results']
        assert 'mutual_info' in report['importance_results']

if __name__ == '__main__':
    pytest.main([__file__, '-v'])