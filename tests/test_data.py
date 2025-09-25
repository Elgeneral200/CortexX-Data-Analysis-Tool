"""
Unit tests for data modules of CortexX sales forecasting platform.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.data.collection import DataCollector
from src.data.preprocessing import DataPreprocessor
from src.data.exploration import DataExplorer

class TestDataCollector:
    """Test cases for DataCollector class."""
    
    def test_init(self):
        """Test DataCollector initialization."""
        collector = DataCollector()
        assert collector is not None
        assert isinstance(collector.config, dict)
    
    def test_generate_sample_data(self):
        """Test sample data generation."""
        collector = DataCollector()
        
        # Test basic generation
        sample_data = collector.generate_sample_data(periods=100, products=2)
        assert isinstance(sample_data, pd.DataFrame)
        assert len(sample_data) == 200  # 100 periods * 2 products
        assert 'date' in sample_data.columns
        assert 'sales' in sample_data.columns
        assert 'product_id' in sample_data.columns
        
        # Test data types
        assert pd.api.types.is_datetime64_any_dtype(sample_data['date'])
        assert pd.api.types.is_numeric_dtype(sample_data['sales'])
    
    def test_validate_data_structure(self):
        """Test data structure validation."""
        collector = DataCollector()
        
        # Create test data
        test_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10),
            'sales': range(10),
            'category': ['A'] * 10
        })
        
        validation = collector.validate_data_structure(test_data)
        assert validation['is_valid'] == True
        assert 'shape' in validation['summary']
        
        # Test with required columns
        validation = collector.validate_data_structure(test_data, ['date', 'sales'])
        assert validation['is_valid'] == True
        
        # Test with missing required columns
        validation = collector.validate_data_structure(test_data, ['missing_column'])
        assert validation['is_valid'] == False

class TestDataPreprocessor:
    """Test cases for DataPreprocessor class."""
    
    def setup_method(self):
        """Set up test data."""
        self.preprocessor = DataPreprocessor()
        self.test_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100),
            'sales': np.random.normal(100, 20, 100),
            'category': ['A', 'B'] * 50
        })
        
        # Introduce some missing values and outliers
        self.test_data.loc[10:15, 'sales'] = np.nan
        self.test_data.loc[90, 'sales'] = 1000  # Outlier
    
    def test_handle_missing_values(self):
        """Test missing value handling."""
        # Test different strategies
        strategies = ['interpolate', 'ffill', 'bfill', 'mean', 'drop']
        
        for strategy in strategies:
            processed = self.preprocessor.handle_missing_values(
                self.test_data, strategy=strategy, columns=['sales']
            )
            assert processed is not None
            assert isinstance(processed, pd.DataFrame)
            
            if strategy != 'drop':
                assert len(processed) == len(self.test_data)
    
    def test_remove_outliers(self):
        """Test outlier removal."""
        # Test IQR method
        processed = self.preprocessor.remove_outliers(self.test_data, 'sales', method='iqr')
        assert len(processed) <= len(self.test_data)
        
        # Test that outlier was removed
        assert processed['sales'].max() < 1000
    
    def test_encode_categorical_variables(self):
        """Test categorical variable encoding."""
        # Test one-hot encoding
        encoded = self.preprocessor.encode_categorical_variables(
            self.test_data, columns=['category'], method='onehot'
        )
        assert 'category_A' in encoded.columns
        assert 'category_B' in encoded.columns
        
        # Test label encoding
        encoded = self.preprocessor.encode_categorical_variables(
            self.test_data, columns=['category'], method='label'
        )
        assert 'category' in encoded.columns
        assert pd.api.types.is_numeric_dtype(encoded['category'])

class TestDataExplorer:
    """Test cases for DataExplorer class."""
    
    def setup_method(self):
        """Set up test data."""
        self.explorer = DataExplorer()
        self.test_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=365),
            'sales': 100 + 50 * np.sin(2 * np.pi * np.arange(365) / 365) + np.random.normal(0, 10, 365),
            'product_id': ['A'] * 365
        })
    
    def test_generate_summary_statistics(self):
        """Test summary statistics generation."""
        summary = self.explorer.generate_summary_statistics(self.test_data)
        
        assert isinstance(summary, dict)
        assert 'dataset_shape' in summary
        assert 'data_types' in summary
        assert 'missing_values' in summary
        
        # Verify shape
        assert summary['dataset_shape'] == (365, 3)
    
    def test_analyze_time_series_patterns(self):
        """Test time series pattern analysis."""
        analysis = self.explorer.analyze_time_series_patterns(
            self.test_data, 'date', 'sales'
        )
        
        assert isinstance(analysis, dict)
        assert 'time_period' in analysis
        assert 'basic_stats' in analysis
        assert 'seasonality' in analysis
        assert 'stationarity' in analysis
        
        # Verify time period calculations
        assert 'start_date' in analysis['time_period']
        assert 'end_date' in analysis['time_period']
    
    def test_identify_data_issues(self):
        """Test data issue identification."""
        # Create data with issues
        problematic_data = self.test_data.copy()
        problematic_data.loc[10:20, 'sales'] = np.nan  # Missing values
        problematic_data['constant_col'] = 1  # Constant column
        
        issues = self.explorer.identify_data_issues(problematic_data)
        
        assert isinstance(issues, dict)
        assert 'missing_data' in issues
        assert 'data_quality' in issues
        
        # Should identify the constant column
        assert any('constant_col' in issue for issue in issues['data_quality'])

if __name__ == '__main__':
    pytest.main([__file__, '-v'])