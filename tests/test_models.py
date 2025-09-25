"""
Unit tests for models modules of CortexX sales forecasting platform.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.models.training import ModelTrainer
from src.models.evaluation import ModelEvaluator

class TestModelTrainer:
    """Test cases for ModelTrainer class."""
    
    def setup_method(self):
        """Set up test data."""
        self.trainer = ModelTrainer()
        
        # Create realistic time series data
        np.random.seed(42)
        n_samples = 200
        
        dates = pd.date_range('2023-01-01', periods=n_samples, freq='D')
        trend = np.linspace(100, 200, n_samples)
        seasonal = 20 * np.sin(2 * np.pi * np.arange(n_samples) / 30)
        noise = np.random.normal(0, 10, n_samples)
        
        self.test_data = pd.DataFrame({
            'date': dates,
            'sales': trend + seasonal + noise,
            'feature1': np.random.normal(50, 10, n_samples),
            'feature2': np.random.normal(30, 5, n_samples),
            'promotion': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        })
    
    def test_train_test_split(self):
        """Test time-based train-test split."""
        train_df, test_df = self.trainer.train_test_split(
            self.test_data, 'date', 'sales', test_size=0.2
        )
        
        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(test_df, pd.DataFrame)
        
        # Test split proportions
        total_len = len(self.test_data)
        expected_train_len = int(total_len * 0.8)
        expected_test_len = total_len - expected_train_len
        
        assert len(train_df) == expected_train_len
        assert len(test_df) == expected_test_len
        
        # Test that split is time-based (no future data in train)
        assert train_df['date'].max() <= test_df['date'].min()
    
    def test_train_xgboost(self):
        """Test XGBoost model training."""
        # Create train-test split
        train_df, test_df = self.trainer.train_test_split(
            self.test_data, 'date', 'sales', test_size=0.2
        )
        
        model, results = self.trainer.train_xgboost(train_df, test_df, 'date', 'sales')
        
        assert model is not None
        assert isinstance(results, dict)
        
        # Check that required keys are in results
        expected_keys = ['model', 'training_time', 'actual', 'predictions']
        for key in expected_keys:
            assert key in results
        
        # Verify predictions shape matches test data
        assert len(results['predictions']) == len(test_df)
    
    def test_train_lightgbm(self):
        """Test LightGBM model training."""
        train_df, test_df = self.trainer.train_test_split(
            self.test_data, 'date', 'sales', test_size=0.2
        )
        
        model, results = self.trainer.train_lightgbm(train_df, test_df, 'date', 'sales')
        
        assert model is not None
        assert isinstance(results, dict)
        assert 'model' in results
        assert results['model'] == 'LightGBM'
        
        # Verify basic functionality
        assert len(results['predictions']) == len(test_df)
    
    def test_train_ensemble(self):
        """Test ensemble model training."""
        train_df, test_df = self.trainer.train_test_split(
            self.test_data, 'date', 'sales', test_size=0.2
        )
        
        model, results = self.trainer.train_ensemble(train_df, test_df, 'date', 'sales')
        
        assert model is not None
        assert isinstance(results, dict)
        assert 'model' in results
        assert results['model'] == 'Ensemble'
        
        # Ensemble should produce predictions
        assert len(results['predictions']) == len(test_df)

class TestModelEvaluator:
    """Test cases for ModelEvaluator class."""
    
    def setup_method(self):
        """Set up test data."""
        self.evaluator = ModelEvaluator()
        
        # Create sample predictions and actual values
        np.random.seed(42)
        n_samples = 100
        
        self.y_true = np.random.normal(100, 20, n_samples)
        self.y_pred = self.y_true + np.random.normal(0, 5, n_samples)  # Good predictions
        
        # Create sample model results
        self.model_results = {
            'model': 'TestModel',
            'training_time': 10.5,
            'actual': self.y_true,
            'predictions': self.y_pred,
            'dates': pd.date_range('2023-01-01', periods=n_samples)
        }
    
    def test_calculate_metrics(self):
        """Test metric calculation."""
        metrics = self.evaluator.calculate_metrics(self.y_true, self.y_pred)
        
        assert isinstance(metrics, dict)
        
        expected_metrics = ['rmse', 'mae', 'r2', 'mse', 'mape']
        for metric in expected_metrics:
            assert metric in metrics
        
        # For good predictions, R² should be high
        assert metrics['r2'] > 0.5
        # RMSE and MAE should be reasonable
        assert 0 < metrics['rmse'] < 50
        assert 0 < metrics['mae'] < 50
    
    def test_evaluate_model(self):
        """Test model evaluation."""
        evaluation = self.evaluator.evaluate_model(self.model_results)
        
        assert isinstance(evaluation, dict)
        assert evaluation['model_name'] == 'TestModel'
        assert 'rmse' in evaluation
        assert 'r2' in evaluation
        assert 'residuals' in evaluation
        
        # Should have residual statistics
        assert 'residual_stats' in evaluation
        assert 'mean' in evaluation['residual_stats']
        assert 'std' in evaluation['residual_stats']
    
    def test_compare_models(self):
        """Test model comparison."""
        # Create multiple model results
        models_results = {
            'Model1': self.model_results,
            'Model2': {
                'model': 'Model2',
                'training_time': 8.2,
                'actual': self.y_true,
                'predictions': self.y_true + np.random.normal(0, 8, len(self.y_true)),
                'dates': self.model_results['dates']
            }
        }
        
        comparison_df = self.evaluator.compare_models(models_results)
        
        assert isinstance(comparison_df, pd.DataFrame)
        assert not comparison_df.empty
        assert 'Model' in comparison_df.columns
        assert 'RMSE' in comparison_df.columns
        assert 'R²' in comparison_df.columns
        
        # Should have both models
        assert len(comparison_df) == 2
    
    def test_create_residual_analysis(self):
        """Test residual analysis."""
        analysis = self.evaluator.create_residual_analysis(self.model_results)
        
        assert isinstance(analysis, dict)
        assert 'residual_mean' in analysis
        assert 'residual_std' in analysis
        assert 'normality_test' in analysis
        assert 'autocorrelation_test' in analysis
        
        # For good model, residual mean should be close to 0
        assert abs(analysis['residual_mean']) < 10
    
    def test_generate_evaluation_report(self):
        """Test evaluation report generation."""
        models_results = {
            'TestModel': self.model_results
        }
        
        report = self.evaluator.generate_evaluation_report(models_results)
        
        assert isinstance(report, dict)
        assert 'model_comparison' in report
        assert 'best_model' in report
        assert 'detailed_evaluations' in report
        assert 'recommendations' in report
        
        # Should have recommendations
        assert isinstance(report['recommendations'], list)
        assert len(report['recommendations']) > 0

if __name__ == '__main__':
    pytest.main([__file__, '-v'])