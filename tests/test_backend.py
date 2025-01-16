"""
Tests for the backend API endpoints and services.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from unittest.mock import Mock, patch

# Import the FastAPI app and dependencies
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from main import app
from database import get_db, Base
from models import District, CholeraCase, ClimateData, RiskPrediction, Alert
from services.ml_service import MLService
from services.data_service import DataService
from services.alert_service import AlertService

# Test database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

# Create test client
client = TestClient(app)

@pytest.fixture(scope="module")
def setup_database():
    """Set up test database."""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def sample_district():
    """Create a sample district for testing."""
    return District(
        district_code="TEST001",
        district_name="Test District",
        region="Central",
        population=100000
    )

@pytest.fixture
def sample_cholera_case():
    """Create a sample cholera case for testing."""
    return CholeraCase(
        case_id="CASE001",
        district_code="TEST001",
        facility_code="FAC001",
        case_date=date.today(),
        age_group="18-59",
        gender="M",
        case_status="confirmed",
        lab_confirmed=True,
        outcome="recovered"
    )

@pytest.fixture
def sample_climate_data():
    """Create sample climate data for testing."""
    return ClimateData(
        district_code="TEST001",
        date=date.today(),
        rainfall_mm=25.5,
        temperature_celsius=28.0,
        humidity_percent=75.0,
        data_source="NASA"
    )

class TestDistrictEndpoints:
    """Test district-related endpoints."""
    
    def test_create_district(self, setup_database):
        """Test creating a new district."""
        district_data = {
            "district_code": "TEST001",
            "district_name": "Test District",
            "region": "Central",
            "population": 100000
        }
        
        response = client.post("/districts/", json=district_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["district_code"] == "TEST001"
        assert data["district_name"] == "Test District"
        assert data["region"] == "Central"
        assert data["population"] == 100000
    
    def test_get_districts(self, setup_database):
        """Test getting all districts."""
        # First create a district
        district_data = {
            "district_code": "TEST001",
            "district_name": "Test District",
            "region": "Central",
            "population": 100000
        }
        client.post("/districts/", json=district_data)
        
        response = client.get("/districts/")
        assert response.status_code == 200
        
        data = response.json()
        assert len(data) >= 1
        assert any(d["district_code"] == "TEST001" for d in data)
    
    def test_get_district_by_code(self, setup_database):
        """Test getting a specific district by code."""
        # First create a district
        district_data = {
            "district_code": "TEST001",
            "district_name": "Test District",
            "region": "Central",
            "population": 100000
        }
        client.post("/districts/", json=district_data)
        
        response = client.get("/districts/TEST001")
        assert response.status_code == 200
        
        data = response.json()
        assert data["district_code"] == "TEST001"
        assert data["district_name"] == "Test District"
    
    def test_get_nonexistent_district(self, setup_database):
        """Test getting a district that doesn't exist."""
        response = client.get("/districts/NONEXISTENT")
        assert response.status_code == 404

class TestCholeraCaseEndpoints:
    """Test cholera case-related endpoints."""
    
    def test_create_cholera_case(self, setup_database):
        """Test creating a new cholera case."""
        # First create a district
        district_data = {
            "district_code": "TEST001",
            "district_name": "Test District",
            "region": "Central",
            "population": 100000
        }
        client.post("/districts/", json=district_data)
        
        case_data = {
            "case_id": "CASE001",
            "district_code": "TEST001",
            "facility_code": "FAC001",
            "case_date": "2024-01-15",
            "age_group": "18-59",
            "gender": "M",
            "case_status": "confirmed",
            "lab_confirmed": True,
            "outcome": "recovered"
        }
        
        response = client.post("/cholera-cases/", json=case_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["case_id"] == "CASE001"
        assert data["district_code"] == "TEST001"
        assert data["lab_confirmed"] == True
    
    def test_get_cholera_cases(self, setup_database):
        """Test getting cholera cases."""
        # First create a district and case
        district_data = {
            "district_code": "TEST001",
            "district_name": "Test District",
            "region": "Central",
            "population": 100000
        }
        client.post("/districts/", json=district_data)
        
        case_data = {
            "case_id": "CASE001",
            "district_code": "TEST001",
            "facility_code": "FAC001",
            "case_date": "2024-01-15",
            "age_group": "18-59",
            "gender": "M",
            "case_status": "confirmed",
            "lab_confirmed": True,
            "outcome": "recovered"
        }
        client.post("/cholera-cases/", json=case_data)
        
        response = client.get("/cholera-cases/")
        assert response.status_code == 200
        
        data = response.json()
        assert len(data) >= 1
        assert any(c["case_id"] == "CASE001" for c in data)
    
    def test_get_cholera_cases_by_district(self, setup_database):
        """Test getting cholera cases filtered by district."""
        # First create districts and cases
        district_data = {
            "district_code": "TEST001",
            "district_name": "Test District",
            "region": "Central",
            "population": 100000
        }
        client.post("/districts/", json=district_data)
        
        case_data = {
            "case_id": "CASE001",
            "district_code": "TEST001",
            "facility_code": "FAC001",
            "case_date": "2024-01-15",
            "age_group": "18-59",
            "gender": "M",
            "case_status": "confirmed",
            "lab_confirmed": True,
            "outcome": "recovered"
        }
        client.post("/cholera-cases/", json=case_data)
        
        response = client.get("/cholera-cases/?district_code=TEST001")
        assert response.status_code == 200
        
        data = response.json()
        assert len(data) >= 1
        assert all(c["district_code"] == "TEST001" for c in data)

class TestClimateDataEndpoints:
    """Test climate data-related endpoints."""
    
    def test_create_climate_data(self, setup_database):
        """Test creating climate data."""
        # First create a district
        district_data = {
            "district_code": "TEST001",
            "district_name": "Test District",
            "region": "Central",
            "population": 100000
        }
        client.post("/districts/", json=district_data)
        
        climate_data = {
            "district_code": "TEST001",
            "date": "2024-01-15",
            "rainfall_mm": 25.5,
            "temperature_celsius": 28.0,
            "humidity_percent": 75.0,
            "data_source": "NASA"
        }
        
        response = client.post("/climate-data/", json=climate_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["district_code"] == "TEST001"
        assert data["rainfall_mm"] == 25.5
        assert data["temperature_celsius"] == 28.0

class TestRiskPredictionEndpoints:
    """Test risk prediction-related endpoints."""
    
    def test_get_risk_predictions(self, setup_database):
        """Test getting risk predictions."""
        response = client.get("/risk-predictions/")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
    
    @patch('services.ml_service.MLService.generate_predictions')
    def test_generate_risk_predictions(self, mock_generate, setup_database):
        """Test generating risk predictions."""
        mock_generate.return_value = {"status": "success"}
        
        response = client.post("/risk-predictions/generate")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data

class TestAlertEndpoints:
    """Test alert-related endpoints."""
    
    def test_get_alerts(self, setup_database):
        """Test getting alerts."""
        response = client.get("/alerts/")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)

class TestDashboardEndpoints:
    """Test dashboard-related endpoints."""
    
    def test_get_dashboard_summary(self, setup_database):
        """Test getting dashboard summary."""
        response = client.get("/dashboard/summary")
        assert response.status_code == 200
        
        data = response.json()
        assert "total_districts" in data
        assert "total_cases" in data
        assert "active_alerts" in data

class TestMLService:
    """Test ML service functionality."""
    
    def test_ml_service_initialization(self):
        """Test ML service initialization."""
        ml_service = MLService()
        assert ml_service is not None
        assert hasattr(ml_service, 'data_service')
    
    @patch('services.ml_service.mlflow.start_run')
    def test_lstm_prediction(self, mock_mlflow):
        """Test LSTM prediction method."""
        ml_service = MLService()
        
        # Create sample data
        dates = pd.date_range(start='2024-01-01', end='2024-01-30', freq='D')
        data = pd.DataFrame({
            'date': dates,
            'cases_count': np.random.poisson(5, len(dates)),
            'rainfall_mm': np.random.normal(20, 10, len(dates)),
            'temperature_celsius': np.random.normal(25, 5, len(dates)),
            'water_area_sqkm': np.random.normal(10, 2, len(dates))
        })
        
        prediction = ml_service._predict_lstm(data, 8)
        assert isinstance(prediction, float)
        assert 0 <= prediction <= 100
    
    @patch('services.ml_service.mlflow.start_run')
    def test_xgboost_prediction(self, mock_mlflow):
        """Test XGBoost prediction method."""
        ml_service = MLService()
        
        # Create sample data
        dates = pd.date_range(start='2024-01-01', end='2024-01-30', freq='D')
        data = pd.DataFrame({
            'date': dates,
            'cases_count': np.random.poisson(5, len(dates)),
            'rainfall_mm': np.random.normal(20, 10, len(dates)),
            'temperature_celsius': np.random.normal(25, 5, len(dates)),
            'water_area_sqkm': np.random.normal(10, 2, len(dates))
        })
        
        prediction = ml_service._predict_xgboost(data, 8)
        assert isinstance(prediction, float)
        assert 0 <= prediction <= 100

class TestDataService:
    """Test data service functionality."""
    
    def test_data_service_initialization(self):
        """Test data service initialization."""
        data_service = DataService()
        assert data_service is not None
    
    def test_prepare_training_data_structure(self):
        """Test that prepare_training_data returns correct structure."""
        data_service = DataService()
        
        # Mock database session
        mock_db = Mock()
        
        # Mock queries to return sample data
        mock_cases = [
            (date.today() - timedelta(days=i), 1) for i in range(30)
        ]
        mock_climate = [
            (date.today() - timedelta(days=i), 20.0, 25.0, 70.0) for i in range(30)
        ]
        mock_water = [
            (date.today() - timedelta(days=i), 10.0, 0.5) for i in range(30)
        ]
        
        with patch.object(mock_db, 'query') as mock_query:
            mock_query.return_value.filter.return_value.group_by.return_value.order_by.return_value.all.return_value = mock_cases
            
            # Test with insufficient data
            result = data_service.prepare_training_data("TEST001", mock_db)
            assert result is None

class TestAlertService:
    """Test alert service functionality."""
    
    def test_alert_service_initialization(self):
        """Test alert service initialization."""
        alert_service = AlertService()
        assert alert_service is not None
        assert hasattr(alert_service, 'risk_thresholds')
    
    def test_get_severity_level(self):
        """Test severity level determination."""
        alert_service = AlertService()
        
        assert alert_service._get_severity_level(95) == "critical"
        assert alert_service._get_severity_level(80) == "high"
        assert alert_service._get_severity_level(60) == "medium"
        assert alert_service._get_severity_level(30) == "low"
    
    def test_update_risk_thresholds(self):
        """Test updating risk thresholds."""
        alert_service = AlertService()
        
        new_thresholds = {
            'critical': 95.0,
            'high': 80.0
        }
        
        alert_service.update_risk_thresholds(new_thresholds)
        
        assert alert_service.get_risk_thresholds()['critical'] == 95.0
        assert alert_service.get_risk_thresholds()['high'] == 80.0

class TestHealthCheck:
    """Test health check endpoints."""
    
    def test_root_endpoint(self):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "version" in data
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"

if __name__ == "__main__":
    pytest.main([__file__])