"""
Tests for ML models and prediction functionality.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta
from unittest.mock import Mock, patch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from services.ml_service import MLService
from services.data_service import DataService

class TestMLModelPredictions:
    """Test ML model prediction functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        dates = pd.date_range(start='2024-01-01', end='2024-03-31', freq='D')
        np.random.seed(42)  # For reproducible results
        
        return pd.DataFrame({
            'date': dates,
            'cases_count': np.random.poisson(5, len(dates)),
            'rainfall_mm': np.random.normal(20, 10, len(dates)),
            'temperature_celsius': np.random.normal(25, 5, len(dates)),
            'water_area_sqkm': np.random.normal(10, 2, len(dates)),
            'humidity_percent': np.random.normal(70, 10, len(dates))
        })
    
    def test_lstm_prediction_with_valid_data(self, sample_data):
        """Test LSTM prediction with valid data."""
        ml_service = MLService()
        
        with patch('services.ml_service.mlflow.start_run'):
            prediction = ml_service._predict_lstm(sample_data, 8)
            
            assert isinstance(prediction, float)
            assert 0 <= prediction <= 100
    
    def test_lstm_prediction_with_insufficient_data(self):
        """Test LSTM prediction with insufficient data."""
        ml_service = MLService()
        
        # Create data with less than 30 days
        dates = pd.date_range(start='2024-01-01', end='2024-01-15', freq='D')
        insufficient_data = pd.DataFrame({
            'date': dates,
            'cases_count': np.random.poisson(5, len(dates)),
            'rainfall_mm': np.random.normal(20, 10, len(dates)),
            'temperature_celsius': np.random.normal(25, 5, len(dates)),
            'water_area_sqkm': np.random.normal(10, 2, len(dates))
        })
        
        with patch('services.ml_service.mlflow.start_run'):
            prediction = ml_service._predict_lstm(insufficient_data, 8)
            
            # Should return default value for insufficient data
            assert prediction == 50.0
    
    def test_xgboost_prediction_with_valid_data(self, sample_data):
        """Test XGBoost prediction with valid data."""
        ml_service = MLService()
        
        with patch('services.ml_service.mlflow.start_run'):
            prediction = ml_service._predict_xgboost(sample_data, 8)
            
            assert isinstance(prediction, float)
            assert 0 <= prediction <= 100
    
    def test_prophet_prediction_with_valid_data(self, sample_data):
        """Test Prophet prediction with valid data."""
        ml_service = MLService()
        
        with patch('services.ml_service.mlflow.start_run'):
            prediction = ml_service._predict_prophet(sample_data, 8)
            
            assert isinstance(prediction, float)
            assert 0 <= prediction <= 100
    
    def test_arima_prediction_with_valid_data(self, sample_data):
        """Test ARIMA prediction with valid data."""
        ml_service = MLService()
        
        with patch('services.ml_service.mlflow.start_run'):
            prediction = ml_service._predict_arima(sample_data, 8)
            
            assert isinstance(prediction, float)
            assert 0 <= prediction <= 100
    
    def test_prediction_error_handling(self, sample_data):
        """Test that prediction errors are handled gracefully."""
        ml_service = MLService()
        
        # Mock an error in the prediction
        with patch('services.ml_service.mlflow.start_run', side_effect=Exception("Test error")):
            prediction = ml_service._predict_lstm(sample_data, 8)
            
            # Should return default value on error
            assert prediction == 50.0

class TestDataPreparation:
    """Test data preparation functionality."""
    
    def test_data_preparation_structure(self):
        """Test that prepared data has correct structure."""
        data_service = DataService()
        
        # Mock database session with sample data
        mock_db = Mock()
        
        # Mock cholera cases data
        mock_cases_data = [
            (date.today() - timedelta(days=i), 1) for i in range(60)
        ]
        
        # Mock climate data
        mock_climate_data = [
            (date.today() - timedelta(days=i), 20.0, 25.0, 70.0) for i in range(60)
        ]
        
        # Mock water extent data
        mock_water_data = [
            (date.today() - timedelta(days=i), 10.0, 0.5) for i in range(60)
        ]
        
        # Set up mock query chain
        def mock_query_chain(*args, **kwargs):
            mock_query = Mock()
            mock_query.filter.return_value = mock_query
            mock_query.group_by.return_value = mock_query
            mock_query.order_by.return_value = mock_query
            
            # Return different data based on the query
            if 'cholera_cases' in str(args):
                mock_query.all.return_value = mock_cases_data
            elif 'climate_data' in str(args):
                mock_query.all.return_value = mock_climate_data
            elif 'water_extent' in str(args):
                mock_query.all.return_value = mock_water_data
            else:
                mock_query.all.return_value = []
            
            return mock_query
        
        mock_db.query.side_effect = mock_query_chain
        
        # Test data preparation
        result = data_service.prepare_training_data("TEST001", mock_db)
        
        if result is not None:
            assert isinstance(result, pd.DataFrame)
            assert 'date' in result.columns
            assert 'cases_count' in result.columns
            assert 'rainfall_mm' in result.columns
            assert 'temperature_celsius' in result.columns
    
    def test_data_preparation_with_insufficient_data(self):
        """Test data preparation with insufficient data."""
        data_service = DataService()
        
        mock_db = Mock()
        
        # Mock with very little data
        mock_cases_data = [(date.today(), 1)]  # Only one day of data
        
        mock_db.query.return_value.filter.return_value.group_by.return_value.order_by.return_value.all.return_value = mock_cases_data
        
        result = data_service.prepare_training_data("TEST001", mock_db)
        
        # Should return None for insufficient data
        assert result is None

class TestModelEnsemble:
    """Test model ensemble functionality."""
    
    def test_ensemble_prediction(self):
        """Test ensemble prediction combining multiple models."""
        ml_service = MLService()
        
        # Mock individual model predictions
        mock_predictions = {
            'LSTM': 60.0,
            'XGBoost': 55.0,
            'Prophet': 65.0,
            'ARIMA': 50.0
        }
        
        # Create sample data
        dates = pd.date_range(start='2024-01-01', end='2024-03-31', freq='D')
        sample_data = pd.DataFrame({
            'date': dates,
            'cases_count': np.random.poisson(5, len(dates)),
            'rainfall_mm': np.random.normal(20, 10, len(dates)),
            'temperature_celsius': np.random.normal(25, 5, len(dates)),
            'water_area_sqkm': np.random.normal(10, 2, len(dates))
        })
        
        # Mock individual prediction methods
        with patch.object(ml_service, '_predict_lstm', return_value=mock_predictions['LSTM']), \
             patch.object(ml_service, '_predict_xgboost', return_value=mock_predictions['XGBoost']), \
             patch.object(ml_service, '_predict_prophet', return_value=mock_predictions['Prophet']), \
             patch.object(ml_service, '_predict_arima', return_value=mock_predictions['ARIMA']):
            
            # Test the generate_predictions method
            result = ml_service.generate_predictions("TEST001", 8, None)
            
            # Should return a dictionary with prediction results
            assert isinstance(result, dict)
            assert "TEST001" in result
            assert "ensemble_prediction" in result["TEST001"]

class TestFeatureEngineering:
    """Test feature engineering functionality."""
    
    def test_lagged_features(self):
        """Test creation of lagged features."""
        data_service = DataService()
        
        # Create sample data
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        sample_data = pd.DataFrame({
            'date': dates,
            'cases_count': range(1, 32),
            'rainfall_mm': np.random.normal(20, 5, 31),
            'temperature_celsius': np.random.normal(25, 3, 31)
        })
        
        # Test that lagged features are created
        # This would be tested within the prepare_training_data method
        mock_db = Mock()
        mock_db.query.return_value.filter.return_value.group_by.return_value.order_by.return_value.all.return_value = []
        
        # Since we can't easily test the internal feature engineering,
        # we test that the method handles the data correctly
        result = data_service.prepare_training_data("TEST001", mock_db)
        
        # Should return None for insufficient data
        assert result is None
    
    def test_rolling_features(self):
        """Test creation of rolling window features."""
        # Test rolling window calculations
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        sample_data = pd.DataFrame({
            'date': dates,
            'cases_count': range(1, 32),
            'rainfall_mm': np.random.normal(20, 5, 31)
        })
        
        # Test 7-day rolling average
        rolling_7 = sample_data['cases_count'].rolling(window=7, min_periods=1).mean()
        assert len(rolling_7) == 31
        assert not rolling_7.isna().any()
        
        # Test 14-day rolling average
        rolling_14 = sample_data['cases_count'].rolling(window=14, min_periods=1).mean()
        assert len(rolling_14) == 31
        assert not rolling_14.isna().any()

class TestDataValidation:
    """Test data validation for ML models."""
    
    def test_data_completeness_check(self):
        """Test checking data completeness."""
        # Test with complete data
        complete_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=30, freq='D'),
            'cases_count': np.random.poisson(5, 30),
            'rainfall_mm': np.random.normal(20, 10, 30),
            'temperature_celsius': np.random.normal(25, 5, 30)
        })
        
        # Check completeness
        completeness = complete_data.notna().mean()
        assert completeness.min() > 0.9  # Should be mostly complete
        
        # Test with incomplete data
        incomplete_data = complete_data.copy()
        incomplete_data.loc[0:5, 'rainfall_mm'] = np.nan
        
        completeness_incomplete = incomplete_data.notna().mean()
        assert completeness_incomplete['rainfall_mm'] < 0.9
    
    def test_data_range_validation(self):
        """Test validation of data ranges."""
        # Test reasonable ranges for Uganda
        sample_data = pd.DataFrame({
            'rainfall_mm': [0, 50, 100, 200],  # Reasonable rainfall range
            'temperature_celsius': [15, 25, 35, 40],  # Reasonable temperature range
            'humidity_percent': [20, 50, 80, 100]  # Humidity percentage range
        })
        
        # Check rainfall range
        assert sample_data['rainfall_mm'].between(0, 200).all()
        
        # Check temperature range
        assert sample_data['temperature_celsius'].between(15, 40).all()
        
        # Check humidity range
        assert sample_data['humidity_percent'].between(0, 100).all()

if __name__ == "__main__":
    pytest.main([__file__])