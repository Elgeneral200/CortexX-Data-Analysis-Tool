"""
Configuration management module for CortexX sales forecasting platform.
Handles application configuration and settings.
"""

import os
import json
from typing import Dict, Any, Optional
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class Config:
    """
    A class to manage application configuration and settings.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path (str, optional): Path to configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path or "config.json"
        self.settings = self._load_config()
        
        # Load environment variables
        load_dotenv()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file or create default configuration.
        
        Returns:
            Dict[str, Any]: Configuration settings
        """
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                self.logger.info(f"Loaded configuration from {self.config_path}")
            else:
                config = self._create_default_config()
                self._save_config(config)
                self.logger.info("Created default configuration")
            
            return config
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
            return self._create_default_config()
    
    def _create_default_config(self) -> Dict[str, Any]:
        """
        Create default configuration settings.
        
        Returns:
            Dict[str, Any]: Default configuration
        """
        return {
            "data_settings": {
                "default_date_format": "%Y-%m-%d",
                "max_file_size_mb": 100,
                "supported_formats": [".csv", ".xlsx", ".json"],
                "encoding": "utf-8"
            },
            "model_settings": {
                "default_test_size": 0.2,
                "cross_validation_folds": 5,
                "random_state": 42,
                "max_training_time_minutes": 60
            },
            "feature_settings": {
                "default_lags": [1, 7, 30, 90],
                "default_rolling_windows": [7, 30, 90],
                "max_features": 100,
                "correlation_threshold": 0.95
            },
            "visualization_settings": {
                "default_theme": "plotly_white",
                "chart_height": 500,
                "color_palette": "Viridis",
                "animation_duration": 500
            },
            "api_settings": {
                "host": "localhost",
                "port": 8000,
                "debug": False,
                "log_level": "INFO"
            },
            "storage_settings": {
                "model_directory": "models",
                "data_directory": "data",
                "log_directory": "logs",
                "cache_directory": "cache"
            }
        }
    
    def _save_config(self, config: Dict[str, Any]) -> None:
        """
        Save configuration to file.
        
        Args:
            config (Dict[str, Any]): Configuration to save
        """
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving configuration: {str(e)}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key (str): Configuration key (e.g., "data_settings.default_date_format")
            default (Any): Default value if key not found
            
        Returns:
            Any: Configuration value
        """
        try:
            keys = key.split('.')
            value = self.settings
            
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            
            return value
            
        except Exception as e:
            self.logger.error(f"Error getting configuration key {key}: {str(e)}")
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.
        
        Args:
            key (str): Configuration key
            value (Any): Value to set
        """
        try:
            keys = key.split('.')
            current_level = self.settings
            
            # Navigate to the appropriate level
            for k in keys[:-1]:
                if k not in current_level:
                    current_level[k] = {}
                current_level = current_level[k]
            
            # Set the value
            current_level[keys[-1]] = value
            
            # Save the updated configuration
            self._save_config(self.settings)
            
            self.logger.info(f"Updated configuration: {key} = {value}")
            
        except Exception as e:
            self.logger.error(f"Error setting configuration key {key}: {str(e)}")
    
    def get_environment_variable(self, key: str, default: Any = None) -> Any:
        """
        Get value from environment variable.
        
        Args:
            key (str): Environment variable name
            default (Any): Default value if not found
            
        Returns:
            Any: Environment variable value
        """
        return os.getenv(key, default)
    
    def validate_config(self) -> Dict[str, Any]:
        """
        Validate configuration settings.
        
        Returns:
            Dict[str, Any]: Validation results
        """
        validation_result = {
            'is_valid': True,
            'issues': [],
            'warnings': []
        }
        
        try:
            # Validate data settings
            data_settings = self.get('data_settings', {})
            if not data_settings:
                validation_result['issues'].append("Missing data settings")
            
            # Validate model settings
            model_settings = self.get('model_settings', {})
            test_size = model_settings.get('default_test_size', 0.2)
            if not (0 < test_size < 1):
                validation_result['issues'].append("Invalid test size")
            
            # Check for environment variables
            required_env_vars = ['DATABASE_URL', 'API_KEY']  # Example required vars
            for env_var in required_env_vars:
                if not self.get_environment_variable(env_var):
                    validation_result['warnings'].append(f"Missing environment variable: {env_var}")
            
            validation_result['is_valid'] = len(validation_result['issues']) == 0
            return validation_result
            
        except Exception as e:
            validation_result['is_valid'] = False
            validation_result['issues'].append(f"Validation error: {str(e)}")
            return validation_result