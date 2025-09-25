"""
CortexX - Enterprise Sales Forecasting Platform
Professional sales forecasting and demand prediction system.
"""

__version__ = "1.0.0"
__author__ = "CortexX Team"
__email__ = "info@cortexx.ai"

from .data.collection import DataCollector
from .data.preprocessing import DataPreprocessor
from .data.exploration import DataExplorer
from .features.engineering import FeatureEngineer
from .features.selection import FeatureSelector
from .models.training import ModelTrainer
from .models.evaluation import ModelEvaluator
from .visualization.dashboard import VisualizationEngine

__all__ = [
    "DataCollector",
    "DataPreprocessor", 
    "DataExplorer",
    "FeatureEngineer",
    "FeatureSelector",
    "ModelTrainer",
    "ModelEvaluator",
    "VisualizationEngine",
]