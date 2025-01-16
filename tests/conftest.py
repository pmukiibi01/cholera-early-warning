"""
Pytest configuration and shared fixtures.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

@pytest.fixture
def sample_districts():
    """Create sample districts for testing."""
    return [
        {
            "district_code": "TEST001",
            "district_name": "Test District 1",
            "region": "Central",
            "population": 100000
        },
        {
            "district_code": "TEST002",
            "district_name": "Test District 2",
            "region": "Eastern",
            "population": 150000
        }
    ]

@pytest.fixture
def sample_cholera_cases():
    """Create sample cholera cases for testing."""
    return [
        {
            "case_id": "CASE001",
            "district_code": "TEST001",
            "facility_code": "FAC001",
            "case_date": "2024-01-15",
            "age_group": "18-59",
            "gender": "M",
            "case_status": "confirmed",
            "lab_confirmed": True,
            "outcome": "recovered"
        },
        {
            "case_id": "CASE002",
            "district_code": "TEST001",
            "facility_code": "FAC002",
            "case_date": "2024-01-16",
            "age_group": "0-5",
            "gender": "F",
            "case_status": "suspected",
            "lab_confirmed": False,
            "outcome": "ongoing"
        }
    ]

@pytest.fixture
def sample_climate_data():
    """Create sample climate data for testing."""
    return [
        {
            "district_code": "TEST001",
            "date": "2024-01-15",
            "rainfall_mm": 25.5,
            "temperature_celsius": 28.0,
            "humidity_percent": 75.0,
            "data_source": "NASA"
        },
        {
            "district_code": "TEST001",
            "date": "2024-01-16",
            "rainfall_mm": 30.2,
            "temperature_celsius": 26.5,
            "humidity_percent": 80.0,
            "data_source": "NASA"
        }
    ]

@pytest.fixture
def sample_risk_predictions():
    """Create sample risk predictions for testing."""
    return [
        {
            "district_code": "TEST001",
            "prediction_date": "2024-01-15",
            "prediction_horizon_weeks": 8,
            "risk_score": 65.5,
            "confidence_interval_lower": 52.4,
            "confidence_interval_upper": 78.6,
            "model_name": "Ensemble",
            "model_version": "1.0"
        },
        {
            "district_code": "TEST002",
            "prediction_date": "2024-01-15",
            "prediction_horizon_weeks": 8,
            "risk_score": 45.2,
            "confidence_interval_lower": 36.2,
            "confidence_interval_upper": 54.2,
            "model_name": "Ensemble",
            "model_version": "1.0"
        }
    ]

@pytest.fixture
def sample_alerts():
    """Create sample alerts for testing."""
    return [
        {
            "alert_id": "ALERT001",
            "district_code": "TEST001",
            "alert_type": "cholera_risk",
            "severity": "high",
            "risk_score": 75.5,
            "prediction_horizon_weeks": 8,
            "triggered_at": "2024-01-15T10:00:00",
            "status": "active"
        },
        {
            "alert_id": "ALERT002",
            "district_code": "TEST002",
            "alert_type": "cholera_risk",
            "severity": "medium",
            "risk_score": 55.0,
            "prediction_horizon_weeks": 8,
            "triggered_at": "2024-01-15T11:00:00",
            "status": "acknowledged"
        }
    ]

@pytest.fixture
def sample_training_data():
    """Create sample training data for ML models."""
    dates = pd.date_range(start='2024-01-01', end='2024-03-31', freq='D')
    np.random.seed(42)  # For reproducible results
    
    return pd.DataFrame({
        'date': dates,
        'cases_count': np.random.poisson(5, len(dates)),
        'rainfall_mm': np.random.normal(20, 10, len(dates)),
        'temperature_celsius': np.random.normal(25, 5, len(dates)),
        'water_area_sqkm': np.random.normal(10, 2, len(dates)),
        'humidity_percent': np.random.normal(70, 10, len(dates)),
        'day_of_week': dates.dayofweek,
        'month': dates.month,
        'year': dates.year
    })

@pytest.fixture
def mock_database_session():
    """Create a mock database session for testing."""
    from unittest.mock import Mock
    
    mock_session = Mock()
    mock_session.query.return_value.filter.return_value.group_by.return_value.order_by.return_value.all.return_value = []
    mock_session.commit.return_value = None
    mock_session.add.return_value = None
    
    return mock_session

@pytest.fixture
def sample_time_series_data():
    """Create sample time series data for testing."""
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
    
    return pd.DataFrame({
        'date': dates,
        'cases_count': np.random.poisson(3, len(dates)),
        'rainfall_mm': np.random.normal(15, 8, len(dates)),
        'temperature_celsius': np.random.normal(26, 4, len(dates)),
        'humidity_percent': np.random.normal(72, 8, len(dates))
    })

@pytest.fixture
def sample_validation_results():
    """Create sample data validation results."""
    return {
        "district_code": "TEST001",
        "overall_validation_success": True,
        "validation_results": {
            "cholera_cases": {
                "district_code": "TEST001",
                "validation_success": True,
                "statistics": {
                    "total_cases": 150,
                    "lab_confirmed_cases": 120,
                    "date_range": {
                        "start": "2024-01-01",
                        "end": "2024-01-31"
                    }
                }
            },
            "climate_data": {
                "district_code": "TEST001",
                "validation_success": True,
                "statistics": {
                    "total_records": 31,
                    "avg_rainfall": 18.5,
                    "avg_temperature": 25.8,
                    "avg_humidity": 74.2
                }
            }
        },
        "timestamp": "2024-01-31T12:00:00"
    }

@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data."""
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Mark tests with 'integration' in the name as integration tests
        if "integration" in item.name:
            item.add_marker(pytest.mark.integration)
        
        # Mark tests with 'slow' in the name as slow tests
        if "slow" in item.name:
            item.add_marker(pytest.mark.slow)
        
        # Mark ML model tests as potentially slow
        if "ml" in item.name.lower() or "model" in item.name.lower():
            item.add_marker(pytest.mark.slow)