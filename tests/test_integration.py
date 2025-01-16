"""
Integration tests for the Cholera Early Warning System.
"""

import pytest
import requests
import time
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from unittest.mock import patch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from main import app
from database import get_db, Base
from services.ml_service import MLService
from services.data_service import DataService
from services.alert_service import AlertService

# Integration test configuration
BACKEND_URL = "http://localhost:8000"
FRONTEND_URL = "http://localhost:8050"

@pytest.mark.integration
class TestSystemIntegration:
    """Test end-to-end system integration."""
    
    def test_backend_health_check(self):
        """Test that backend is healthy."""
        try:
            response = requests.get(f"{BACKEND_URL}/health", timeout=5)
            assert response.status_code == 200
            assert response.json()["status"] == "healthy"
        except requests.exceptions.ConnectionError:
            pytest.skip("Backend not running")
    
    def test_frontend_health_check(self):
        """Test that frontend is accessible."""
        try:
            response = requests.get(f"{FRONTEND_URL}", timeout=5)
            assert response.status_code == 200
        except requests.exceptions.ConnectionError:
            pytest.skip("Frontend not running")
    
    def test_api_endpoints_accessible(self):
        """Test that all API endpoints are accessible."""
        try:
            endpoints = [
                "/",
                "/health",
                "/districts/",
                "/cholera-cases/",
                "/climate-data/",
                "/risk-predictions/",
                "/alerts/",
                "/dashboard/summary"
            ]
            
            for endpoint in endpoints:
                response = requests.get(f"{BACKEND_URL}{endpoint}", timeout=5)
                assert response.status_code == 200, f"Endpoint {endpoint} failed"
        except requests.exceptions.ConnectionError:
            pytest.skip("Backend not running")

@pytest.mark.integration
class TestDataFlowIntegration:
    """Test data flow through the system."""
    
    def test_district_creation_and_retrieval(self):
        """Test creating a district and retrieving it."""
        try:
            # Create a test district
            district_data = {
                "district_code": "INTEG001",
                "district_name": "Integration Test District",
                "region": "Test Region",
                "population": 200000
            }
            
            response = requests.post(f"{BACKEND_URL}/districts/", json=district_data)
            assert response.status_code == 200
            
            created_district = response.json()
            assert created_district["district_code"] == "INTEG001"
            
            # Retrieve the district
            response = requests.get(f"{BACKEND_URL}/districts/INTEG001")
            assert response.status_code == 200
            
            retrieved_district = response.json()
            assert retrieved_district["district_code"] == "INTEG001"
            assert retrieved_district["district_name"] == "Integration Test District"
            
        except requests.exceptions.ConnectionError:
            pytest.skip("Backend not running")
    
    def test_cholera_case_workflow(self):
        """Test the complete cholera case workflow."""
        try:
            # First create a district
            district_data = {
                "district_code": "INTEG002",
                "district_name": "Integration Test District 2",
                "region": "Test Region",
                "population": 150000
            }
            requests.post(f"{BACKEND_URL}/districts/", json=district_data)
            
            # Create a cholera case
            case_data = {
                "case_id": "INTEG_CASE001",
                "district_code": "INTEG002",
                "facility_code": "INTEG_FAC001",
                "case_date": date.today().isoformat(),
                "age_group": "18-59",
                "gender": "M",
                "case_status": "confirmed",
                "lab_confirmed": True,
                "outcome": "recovered"
            }
            
            response = requests.post(f"{BACKEND_URL}/cholera-cases/", json=case_data)
            assert response.status_code == 200
            
            created_case = response.json()
            assert created_case["case_id"] == "INTEG_CASE001"
            
            # Retrieve cases for the district
            response = requests.get(f"{BACKEND_URL}/cholera-cases/?district_code=INTEG002")
            assert response.status_code == 200
            
            cases = response.json()
            assert len(cases) >= 1
            assert any(case["case_id"] == "INTEG_CASE001" for case in cases)
            
        except requests.exceptions.ConnectionError:
            pytest.skip("Backend not running")
    
    def test_climate_data_workflow(self):
        """Test the climate data workflow."""
        try:
            # Create climate data
            climate_data = {
                "district_code": "INTEG002",
                "date": date.today().isoformat(),
                "rainfall_mm": 30.5,
                "temperature_celsius": 28.5,
                "humidity_percent": 78.0,
                "data_source": "Integration Test"
            }
            
            response = requests.post(f"{BACKEND_URL}/climate-data/", json=climate_data)
            assert response.status_code == 200
            
            created_data = response.json()
            assert created_data["district_code"] == "INTEG002"
            assert created_data["rainfall_mm"] == 30.5
            
            # Retrieve climate data
            response = requests.get(f"{BACKEND_URL}/climate-data/?district_code=INTEG002")
            assert response.status_code == 200
            
            data = response.json()
            assert len(data) >= 1
            
        except requests.exceptions.ConnectionError:
            pytest.skip("Backend not running")

@pytest.mark.integration
class TestMLPipelineIntegration:
    """Test ML pipeline integration."""
    
    def test_ml_service_integration(self):
        """Test ML service integration with database."""
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        
        # Create test database connection
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Create test data
        from models import District, CholeraCase, ClimateData
        
        district = District(
            district_code="ML_TEST001",
            district_name="ML Test District",
            region="Test Region",
            population=100000
        )
        session.add(district)
        
        # Add sample cases
        for i in range(30):
            case = CholeraCase(
                case_id=f"ML_CASE_{i:03d}",
                district_code="ML_TEST001",
                case_date=date.today() - timedelta(days=i),
                lab_confirmed=True,
                outcome="recovered"
            )
            session.add(case)
        
        # Add sample climate data
        for i in range(30):
            climate = ClimateData(
                district_code="ML_TEST001",
                date=date.today() - timedelta(days=i),
                rainfall_mm=20.0 + np.random.normal(0, 5),
                temperature_celsius=25.0 + np.random.normal(0, 3),
                humidity_percent=70.0 + np.random.normal(0, 10)
            )
            session.add(climate)
        
        session.commit()
        
        # Test data service
        data_service = DataService()
        training_data = data_service.prepare_training_data("ML_TEST001", session)
        
        assert training_data is not None
        assert len(training_data) >= 30
        assert 'cases_count' in training_data.columns
        assert 'rainfall_mm' in training_data.columns
        
        # Test ML service
        ml_service = MLService()
        
        with patch('services.ml_service.mlflow.start_run'):
            # Test individual model predictions
            lstm_pred = ml_service._predict_lstm(training_data, 8)
            xgb_pred = ml_service._predict_xgboost(training_data, 8)
            
            assert isinstance(lstm_pred, float)
            assert isinstance(xgb_pred, float)
            assert 0 <= lstm_pred <= 100
            assert 0 <= xgb_pred <= 100
        
        session.close()
    
    def test_alert_service_integration(self):
        """Test alert service integration."""
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        
        # Create test database connection
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Create test data
        from models import District, RiskPrediction
        
        district = District(
            district_code="ALERT_TEST001",
            district_name="Alert Test District",
            region="Test Region",
            population=100000
        )
        session.add(district)
        
        # Add high-risk prediction
        high_risk_prediction = RiskPrediction(
            district_code="ALERT_TEST001",
            prediction_date=date.today(),
            prediction_horizon_weeks=8,
            risk_score=80.0,  # High risk
            confidence_interval_lower=64.0,
            confidence_interval_upper=96.0,
            model_name="Ensemble",
            model_version="1.0"
        )
        session.add(high_risk_prediction)
        session.commit()
        
        # Test alert service
        alert_service = AlertService()
        result = alert_service.generate_alerts(session)
        
        assert result["status"] == "success"
        assert result["alerts_generated"] >= 0
        
        # Check if alert was created
        alerts = alert_service.get_active_alerts(session, "ALERT_TEST001")
        assert isinstance(alerts, list)
        
        session.close()

@pytest.mark.integration
class TestPrefectWorkflowIntegration:
    """Test Prefect workflow integration."""
    
    def test_prefect_flow_execution(self):
        """Test Prefect flow execution."""
        try:
            from prefect_flows import cholera_risk_prediction_pipeline
            
            # Mock the database and ML components
            with patch('prefect_flows.extract_districts', return_value=["TEST001"]), \
                 patch('prefect_flows.extract_data', return_value=pd.DataFrame({
                     'date': pd.date_range('2024-01-01', periods=60, freq='D'),
                     'cases_count': np.random.poisson(5, 60),
                     'rainfall_mm': np.random.normal(20, 10, 60),
                     'temperature_celsius': np.random.normal(25, 5, 60),
                     'water_area_sqkm': np.random.normal(10, 2, 60)
                 })), \
                 patch('prefect_flows.validate_data', return_value=True), \
                 patch('prefect_flows.generate_predictions', return_value={
                     'district_code': 'TEST001',
                     'predictions': {'Ensemble': 60.0},
                     'status': 'success'
                 }), \
                 patch('prefect_flows.generate_alerts', return_value={
                     'alerts_generated': 0,
                     'alerts': [],
                     'status': 'success'
                 }), \
                 patch('prefect_flows.cleanup_old_data', return_value={
                     'old_predictions_deleted': 0,
                     'old_alerts_deleted': 0
                 }):
                
                # Execute the pipeline
                result = cholera_risk_prediction_pipeline(horizon_weeks=8, districts=["TEST001"])
                
                assert result["status"] == "completed"
                assert result["total_districts"] == 1
                assert result["successful_predictions"] >= 0
                
        except ImportError:
            pytest.skip("Prefect not available")

@pytest.mark.integration
class TestDataValidationIntegration:
    """Test data validation integration."""
    
    def test_great_expectations_integration(self):
        """Test Great Expectations integration."""
        try:
            from services.data_validation import DataValidationService
            
            # Create validation service
            validation_service = DataValidationService()
            
            # Create sample data for validation
            sample_cases = pd.DataFrame({
                'case_id': [f'CASE_{i:03d}' for i in range(100)],
                'district_code': ['TEST001'] * 100,
                'facility_code': [f'FAC_{i:03d}' for i in range(100)],
                'case_date': pd.date_range('2024-01-01', periods=100, freq='D'),
                'age_group': np.random.choice(['0-5', '6-17', '18-59', '60+'], 100),
                'gender': np.random.choice(['M', 'F'], 100),
                'case_status': np.random.choice(['confirmed', 'suspected'], 100),
                'lab_confirmed': np.random.choice([True, False], 100),
                'outcome': np.random.choice(['recovered', 'died', 'ongoing'], 100)
            })
            
            # Test validation
            result = validation_service.validate_cholera_cases(sample_cases, "TEST001")
            
            assert "district_code" in result
            assert "validation_success" in result
            assert result["district_code"] == "TEST001"
            
        except ImportError:
            pytest.skip("Great Expectations not available")

@pytest.mark.integration
@pytest.mark.slow
class TestPerformanceIntegration:
    """Test system performance under load."""
    
    def test_bulk_data_insertion(self):
        """Test inserting large amounts of data."""
        try:
            # Create test district first
            district_data = {
                "district_code": "PERF001",
                "district_name": "Performance Test District",
                "region": "Test Region",
                "population": 500000
            }
            requests.post(f"{BACKEND_URL}/districts/", json=district_data)
            
            # Create multiple cases
            cases = []
            for i in range(100):
                case_data = {
                    "case_id": f"PERF_CASE_{i:03d}",
                    "district_code": "PERF001",
                    "facility_code": f"PERF_FAC_{i%10:03d}",
                    "case_date": (date.today() - timedelta(days=i)).isoformat(),
                    "age_group": np.random.choice(['0-5', '6-17', '18-59', '60+']),
                    "gender": np.random.choice(['M', 'F']),
                    "case_status": np.random.choice(['confirmed', 'suspected']),
                    "lab_confirmed": np.random.choice([True, False]),
                    "outcome": np.random.choice(['recovered', 'died', 'ongoing'])
                }
                cases.append(case_data)
            
            # Insert cases and measure time
            start_time = time.time()
            
            for case in cases:
                response = requests.post(f"{BACKEND_URL}/cholera-cases/", json=case)
                assert response.status_code == 200
            
            end_time = time.time()
            insertion_time = end_time - start_time
            
            # Should complete within reasonable time (adjust as needed)
            assert insertion_time < 30.0  # 30 seconds for 100 cases
            
            # Verify data was inserted
            response = requests.get(f"{BACKEND_URL}/cholera-cases/?district_code=PERF001")
            assert response.status_code == 200
            
            inserted_cases = response.json()
            assert len(inserted_cases) >= 100
            
        except requests.exceptions.ConnectionError:
            pytest.skip("Backend not running")

if __name__ == "__main__":
    pytest.main([__file__, "-m", "integration"])