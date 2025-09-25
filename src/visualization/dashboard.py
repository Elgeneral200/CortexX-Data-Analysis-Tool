"""
Interactive visualization module for CortexX sales forecasting platform.
Provides advanced EDA capabilities with Plotly visualizations.
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class VisualizationEngine:
    """
    A class to create interactive visualizations for sales forecasting.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create_sales_trend_plot(self, df: pd.DataFrame, date_col: str, 
                               value_col: str, title: str = "Sales Trend Over Time") -> go.Figure:
        """
        Create interactive sales trend visualization.
        
        Args:
            df (pd.DataFrame): Sales data
            date_col (str): Date column name
            value_col (str): Value column name
            title (str): Plot title
            
        Returns:
            go.Figure: Plotly figure object
        """
        try:
            fig = px.line(df, x=date_col, y=value_col, 
                         title=title,
                         template='plotly_white')
            
            fig.update_layout(
                xaxis_title='Date',
                yaxis_title='Sales',
                hovermode='x unified',
                height=500
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating sales trend plot: {str(e)}")
            return self._create_error_plot("Sales Trend Plot")
    
    def create_seasonality_plot(self, df: pd.DataFrame, date_col: str,
                               value_col: str) -> go.Figure:
        """
        Create seasonal decomposition plot.
        
        Args:
            df (pd.DataFrame): Sales data
            date_col (str): Date column name
            value_col (str): Value column name
            
        Returns:
            go.Figure: Seasonal decomposition plot
        """
        try:
            df_temp = df.copy()
            df_temp[date_col] = pd.to_datetime(df_temp[date_col])
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Monthly Seasonality', 'Weekly Seasonality', 
                              'Year-over-Year Comparison', 'Daily Patterns'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Monthly seasonality
            df_temp['month'] = df_temp[date_col].dt.month
            monthly_avg = df_temp.groupby('month')[value_col].mean().reset_index()
            fig.add_trace(
                go.Scatter(x=monthly_avg['month'], y=monthly_avg[value_col],
                          name='Monthly Avg', line=dict(color='blue')),
                row=1, col=1
            )
            
            # Weekly seasonality
            df_temp['day_of_week'] = df_temp[date_col].dt.dayofweek
            weekly_avg = df_temp.groupby('day_of_week')[value_col].mean().reset_index()
            fig.add_trace(
                go.Scatter(x=weekly_avg['day_of_week'], y=weekly_avg[value_col],
                          name='Weekly Avg', line=dict(color='green')),
                row=1, col=2
            )
            
            # Year-over-year comparison
            df_temp['year'] = df_temp[date_col].dt.year
            yearly_avg = df_temp.groupby('year')[value_col].mean().reset_index()
            fig.add_trace(
                go.Bar(x=yearly_avg['year'], y=yearly_avg[value_col],
                      name='Yearly Avg', marker_color='orange'),
                row=2, col=1
            )
            
            # Daily patterns (if hourly data available)
            if 'hour' in df_temp.columns:
                hourly_avg = df_temp.groupby('hour')[value_col].mean().reset_index()
                fig.add_trace(
                    go.Scatter(x=hourly_avg['hour'], y=hourly_avg[value_col],
                              name='Hourly Avg', line=dict(color='red')),
                    row=2, col=2
                )
            else:
                # Use day of month instead
                df_temp['day_of_month'] = df_temp[date_col].dt.day
                daily_avg = df_temp.groupby('day_of_month')[value_col].mean().reset_index()
                fig.add_trace(
                    go.Scatter(x=daily_avg['day_of_month'], y=daily_avg[value_col],
                              name='Daily Avg', line=dict(color='red')),
                    row=2, col=2
                )
            
            fig.update_layout(
                height=600,
                title_text="Seasonality Analysis",
                template='plotly_white',
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating seasonality plot: {str(e)}")
            return self._create_error_plot("Seasonality Plot")
    
    def create_correlation_heatmap(self, df: pd.DataFrame, title: str = "Feature Correlation Matrix") -> go.Figure:
        """
        Create correlation heatmap for numerical features.
        
        Args:
            df (pd.DataFrame): Input dataframe
            title (str): Plot title
            
        Returns:
            go.Figure: Correlation heatmap
        """
        try:
            # Select only numerical columns
            numeric_df = df.select_dtypes(include=[np.number])
            
            if numeric_df.empty:
                return self._create_message_plot("No numerical columns available for correlation analysis")
            
            # Calculate correlation matrix
            corr_matrix = numeric_df.corr()
            
            # Create heatmap
            fig = px.imshow(corr_matrix, 
                          title=title,
                          color_continuous_scale='RdBu_r',
                          aspect="auto",
                          template='plotly_white')
            
            fig.update_layout(height=600)
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating correlation heatmap: {str(e)}")
            return self._create_error_plot("Correlation Heatmap")
    
    def create_forecast_comparison_plot(self, actual_dates: np.ndarray, actual_values: np.ndarray,
                                       forecast_dates: np.ndarray, forecast_values: np.ndarray,
                                       model_name: str = "Model") -> go.Figure:
        """
        Create forecast vs actual comparison plot.
        
        Args:
            actual_dates (np.ndarray): Actual values dates
            actual_values (np.ndarray): Actual values
            forecast_dates (np.ndarray): Forecast dates
            forecast_values (np.ndarray): Forecast values
            model_name (str): Name of the forecasting model
            
        Returns:
            go.Figure: Forecast comparison plot
        """
        try:
            fig = go.Figure()
            
            # Add actual values
            fig.add_trace(go.Scatter(
                x=actual_dates,
                y=actual_values,
                name='Actual',
                line=dict(color='blue', width=2)
            ))
            
            # Add forecast values
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=forecast_values,
                name=f'{model_name} Forecast',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title=f'{model_name} - Forecast vs Actual',
                xaxis_title='Date',
                yaxis_title='Sales',
                template='plotly_white',
                hovermode='x unified',
                height=500
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating forecast comparison plot: {str(e)}")
            return self._create_error_plot("Forecast Comparison")
    
    def create_residual_analysis_plot(self, actual_values: np.ndarray, 
                                     predicted_values: np.ndarray) -> go.Figure:
        """
        Create residual analysis plots.
        
        Args:
            actual_values (np.ndarray): Actual values
            predicted_values (np.ndarray): Predicted values
            
        Returns:
            go.Figure: Residual analysis plots
        """
        try:
            residuals = predicted_values - actual_values
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Residuals vs Predicted', 'Residual Distribution',
                              'Q-Q Plot', 'Residuals Over Time'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Residuals vs Predicted
            fig.add_trace(
                go.Scatter(x=predicted_values, y=residuals,
                          mode='markers', name='Residuals'),
                row=1, col=1
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
            
            # Residual Distribution
            fig.add_trace(
                go.Histogram(x=residuals, name='Residual Distribution',
                           nbinsx=50),
                row=1, col=2
            )
            
            # Q-Q Plot (simplified)
            from scipy import stats
            sorted_residuals = np.sort(residuals)
            theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_residuals)))
            
            fig.add_trace(
                go.Scatter(x=theoretical_quantiles, y=sorted_residuals,
                          mode='markers', name='Q-Q Plot'),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=theoretical_quantiles, y=theoretical_quantiles,
                          mode='lines', name='Theoretical', line=dict(color='red', dash='dash')),
                row=2, col=1
            )
            
            # Residuals over time (using index as time proxy)
            fig.add_trace(
                go.Scatter(x=np.arange(len(residuals)), y=residuals,
                          mode='lines', name='Residuals Over Time'),
                row=2, col=2
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=2)
            
            fig.update_layout(
                height=600,
                title_text="Residual Analysis",
                template='plotly_white',
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating residual analysis plot: {str(e)}")
            return self._create_error_plot("Residual Analysis")
    
    def create_feature_importance_plot(self, importance_df: pd.DataFrame, 
                                      title: str = "Feature Importance") -> go.Figure:
        """
        Create feature importance bar chart.
        
        Args:
            importance_df (pd.DataFrame): DataFrame with feature importance scores
            title (str): Plot title
            
        Returns:
            go.Figure: Feature importance plot
        """
        try:
            if importance_df.empty:
                return self._create_message_plot("No feature importance data available")
            
            # Sort by importance
            importance_df = importance_df.sort_values('importance', ascending=True)
            
            fig = px.bar(importance_df, 
                        x='importance', 
                        y='feature',
                        title=title,
                        orientation='h',
                        template='plotly_white')
            
            fig.update_layout(
                height=max(400, len(importance_df) * 25),
                xaxis_title='Importance Score',
                yaxis_title='Features'
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating feature importance plot: {str(e)}")
            return self._create_error_plot("Feature Importance")
    
    def create_model_comparison_plot(self, comparison_df: pd.DataFrame, 
                                    metric: str = 'RMSE') -> go.Figure:
        """
        Create model comparison bar chart.
        
        Args:
            comparison_df (pd.DataFrame): Model comparison DataFrame
            metric (str): Metric to compare
            
        Returns:
            go.Figure: Model comparison plot
        """
        try:
            if comparison_df.empty:
                return self._create_message_plot("No model comparison data available")
            
            fig = px.bar(comparison_df,
                        x='Model',
                        y=metric,
                        title=f'Model Comparison - {metric}',
                        template='plotly_white',
                        color=metric,
                        color_continuous_scale='Viridis')
            
            fig.update_layout(
                height=500,
                xaxis_title='Models',
                yaxis_title=metric
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating model comparison plot: {str(e)}")
            return self._create_error_plot("Model Comparison")
    
    def _create_error_plot(self, title: str) -> go.Figure:
        """Create an error message plot."""
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating {title}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="red")
        )
        fig.update_layout(
            title=title,
            template='plotly_white',
            height=400
        )
        return fig
    
    def _create_message_plot(self, message: str) -> go.Figure:
        """Create a message plot."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=14, color="gray")
        )
        fig.update_layout(
            template='plotly_white',
            height=400
        )
        return fig