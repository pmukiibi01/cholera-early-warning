"""
Prefect flows for orchestrating ML pipeline and data processing.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import List, Dict, Any, Optional
from prefect import flow, task, get_run_logger
from prefect.task_runners import SequentialTaskRunner
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
import mlflow
import mlflow.sklearn
import mlflow.tensorflow

from services.ml_service import MLService
from services.data_service import DataService
from services.alert_service import AlertService
from models import District, RiskPrediction, Alert

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Database connection
DATABASE_URL = "postgresql://cholera_user:cholera_password@localhost:5432/cholera_ew"
engine = create_engine(DATABASE_URL)


@task
def extract_districts() -> List[str]:
    """Extract list of all districts."""
    logger = get_run_logger()
    
    with Session(engine) as db:
        districts = db.query(District.district_code).all()
        district_codes = [d[0] for d in districts]
    
    logger.info(f"Extracted {len(district_codes)} districts")
    return district_codes


@task
def extract_data(district_code: str) -> Optional[pd.DataFrame]:
    """Extract and prepare data for a specific district."""
    logger = get_run_logger()
    
    try:
        data_service = DataService()
        with Session(engine) as db:
            data = data_service.prepare_training_data(district_code, db)
        
        if data is not None and len(data) > 30:
            logger.info(f"Extracted {len(data)} records for district {district_code}")
            return data
        else:
            logger.warning(f"Insufficient data for district {district_code}")
            return None
            
    except Exception as e:
        logger.error(f"Error extracting data for district {district_code}: {e}")
        return None


@task
def validate_data(data: pd.DataFrame, district_code: str) -> bool:
    """Validate extracted data quality."""
    logger = get_run_logger()
    
    try:
        # Check minimum data requirements
        if len(data) < 30:
            logger.warning(f"District {district_code}: Insufficient data points ({len(data)})")
            return False
        
        # Check for missing critical columns
        required_columns = ['date', 'cases_count', 'rainfall_mm', 'temperature_celsius']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logger.warning(f"District {district_code}: Missing columns {missing_columns}")
            return False
        
        # Check data completeness
        completeness = data[required_columns].notna().mean()
        if completeness.min() < 0.5:  # At least 50% completeness
            logger.warning(f"District {district_code}: Low data completeness ({completeness.min():.2f})")
            return False
        
        logger.info(f"District {district_code}: Data validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Error validating data for district {district_code}: {e}")
        return False


@task
def train_model(data: pd.DataFrame, district_code: str, model_name: str) -> Optional[Dict[str, Any]]:
    """Train a specific model for a district."""
    logger = get_run_logger()
    
    try:
        ml_service = MLService()
        
        # Train model based on type
        if model_name == "LSTM":
            model_info = ml_service._train_lstm_model(data, district_code)
        elif model_name == "XGBoost":
            model_info = ml_service._train_xgboost_model(data, district_code)
        elif model_name == "Prophet":
            model_info = ml_service._train_prophet_model(data, district_code)
        elif model_name == "ARIMA":
            model_info = ml_service._train_arima_model(data, district_code)
        else:
            logger.error(f"Unknown model type: {model_name}")
            return None
        
        logger.info(f"Successfully trained {model_name} for district {district_code}")
        return model_info
        
    except Exception as e:
        logger.error(f"Error training {model_name} for district {district_code}: {e}")
        return None


@task
def generate_predictions(district_code: str, horizon_weeks: int = 8) -> Optional[Dict[str, Any]]:
    """Generate predictions for a specific district."""
    logger = get_run_logger()
    
    try:
        ml_service = MLService()
        
        with Session(engine) as db:
            # Get data for the district
            data_service = DataService()
            data = data_service.prepare_training_data(district_code, db)
            
            if data is None or len(data) < 30:
                logger.warning(f"Insufficient data for predictions in district {district_code}")
                return None
            
            # Generate predictions using all models
            predictions = {}
            
            # LSTM prediction
            try:
                lstm_pred = ml_service._predict_lstm(data, horizon_weeks)
                predictions['LSTM'] = lstm_pred
            except Exception as e:
                logger.warning(f"LSTM prediction failed for {district_code}: {e}")
                predictions['LSTM'] = 50.0
            
            # XGBoost prediction
            try:
                xgb_pred = ml_service._predict_xgboost(data, horizon_weeks)
                predictions['XGBoost'] = xgb_pred
            except Exception as e:
                logger.warning(f"XGBoost prediction failed for {district_code}: {e}")
                predictions['XGBoost'] = 50.0
            
            # Prophet prediction
            try:
                prophet_pred = ml_service._predict_prophet(data, horizon_weeks)
                predictions['Prophet'] = prophet_pred
            except Exception as e:
                logger.warning(f"Prophet prediction failed for {district_code}: {e}")
                predictions['Prophet'] = 50.0
            
            # ARIMA prediction
            try:
                arima_pred = ml_service._predict_arima(data, horizon_weeks)
                predictions['ARIMA'] = arima_pred
            except Exception as e:
                logger.warning(f"ARIMA prediction failed for {district_code}: {e}")
                predictions['ARIMA'] = 50.0
            
            # Ensemble prediction
            ensemble_pred = np.mean(list(predictions.values()))
            predictions['Ensemble'] = ensemble_pred
            
            # Save predictions to database
            prediction_date = date.today()
            
            for model_name, pred_value in predictions.items():
                risk_prediction = RiskPrediction(
                    district_code=district_code,
                    prediction_date=prediction_date,
                    prediction_horizon_weeks=horizon_weeks,
                    risk_score=float(pred_value),
                    confidence_interval_lower=float(pred_value * 0.8),
                    confidence_interval_upper=float(pred_value * 1.2),
                    model_name=model_name,
                    model_version="1.0",
                    features_used={
                        "rainfall_lag": 7,
                        "temperature_lag": 14,
                        "cases_lag": 21,
                        "water_extent_lag": 7
                    }
                )
                db.add(risk_prediction)
            
            db.commit()
            
            logger.info(f"Generated predictions for district {district_code}: {predictions}")
            return {
                'district_code': district_code,
                'predictions': predictions,
                'status': 'success'
            }
            
    except Exception as e:
        logger.error(f"Error generating predictions for district {district_code}: {e}")
        return {
            'district_code': district_code,
            'predictions': {},
            'status': 'failed',
            'error': str(e)
        }


@task
def generate_alerts() -> Dict[str, Any]:
    """Generate alerts based on recent predictions."""
    logger = get_run_logger()
    
    try:
        alert_service = AlertService()
        
        with Session(engine) as db:
            result = alert_service.generate_alerts(db)
        
        logger.info(f"Generated {result['alerts_generated']} alerts")
        return result
        
    except Exception as e:
        logger.error(f"Error generating alerts: {e}")
        return {
            'alerts_generated': 0,
            'alerts': [],
            'status': 'error',
            'error': str(e)
        }


@task
def cleanup_old_data(days_to_keep: int = 365) -> Dict[str, int]:
    """Clean up old data to maintain database performance."""
    logger = get_run_logger()
    
    try:
        cutoff_date = date.today() - timedelta(days=days_to_keep)
        
        with Session(engine) as db:
            # Clean up old predictions (keep only recent ones)
            old_predictions = db.query(RiskPrediction).filter(
                RiskPrediction.prediction_date < cutoff_date
            ).count()
            
            db.query(RiskPrediction).filter(
                RiskPrediction.prediction_date < cutoff_date
            ).delete()
            
            # Clean up old alerts (keep resolved/already acknowledged)
            old_alerts = db.query(Alert).filter(
                and_(
                    Alert.triggered_at < cutoff_date,
                    Alert.status.in_(['active', 'acknowledged'])
                )
            ).count()
            
            db.query(Alert).filter(
                and_(
                    Alert.triggered_at < cutoff_date,
                    Alert.status.in_(['active', 'acknowledged'])
                )
            ).delete()
            
            db.commit()
        
        logger.info(f"Cleaned up {old_predictions} old predictions and {old_alerts} old alerts")
        return {
            'old_predictions_deleted': old_predictions,
            'old_alerts_deleted': old_alerts
        }
        
    except Exception as e:
        logger.error(f"Error cleaning up old data: {e}")
        return {'old_predictions_deleted': 0, 'old_alerts_deleted': 0}


@flow(
    name="cholera-risk-prediction-pipeline",
    task_runner=SequentialTaskRunner(),
    description="Complete pipeline for cholera risk prediction and alerting"
)
def cholera_risk_prediction_pipeline(
    horizon_weeks: int = 8,
    districts: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Main pipeline for cholera risk prediction."""
    logger = get_run_logger()
    
    try:
        # Step 1: Extract districts
        if districts is None:
            district_codes = extract_districts()
        else:
            district_codes = districts
        
        logger.info(f"Starting prediction pipeline for {len(district_codes)} districts")
        
        # Step 2: Process each district
        results = {}
        successful_predictions = 0
        failed_predictions = 0
        
        for district_code in district_codes:
            try:
                # Extract and validate data
                data = extract_data(district_code)
                if data is None:
                    failed_predictions += 1
                    continue
                
                if not validate_data(data, district_code):
                    failed_predictions += 1
                    continue
                
                # Generate predictions
                prediction_result = generate_predictions(district_code, horizon_weeks)
                
                if prediction_result and prediction_result['status'] == 'success':
                    results[district_code] = prediction_result
                    successful_predictions += 1
                else:
                    failed_predictions += 1
                    
            except Exception as e:
                logger.error(f"Error processing district {district_code}: {e}")
                failed_predictions += 1
        
        # Step 3: Generate alerts
        alert_result = generate_alerts()
        
        # Step 4: Cleanup old data
        cleanup_result = cleanup_old_data()
        
        pipeline_result = {
            'total_districts': len(district_codes),
            'successful_predictions': successful_predictions,
            'failed_predictions': failed_predictions,
            'alerts_generated': alert_result.get('alerts_generated', 0),
            'cleanup_result': cleanup_result,
            'status': 'completed'
        }
        
        logger.info(f"Pipeline completed: {pipeline_result}")
        return pipeline_result
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return {
            'total_districts': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'alerts_generated': 0,
            'status': 'failed',
            'error': str(e)
        }


@flow(
    name="daily-prediction-workflow",
    description="Daily workflow for generating predictions and alerts"
)
def daily_prediction_workflow() -> Dict[str, Any]:
    """Daily workflow that runs the prediction pipeline."""
    logger = get_run_logger()
    
    logger.info("Starting daily prediction workflow")
    
    # Run the main prediction pipeline
    result = cholera_risk_prediction_pipeline(horizon_weeks=8)
    
    logger.info(f"Daily workflow completed: {result}")
    return result


@flow(
    name="weekly-model-retraining",
    description="Weekly workflow for retraining models"
)
def weekly_model_retraining() -> Dict[str, Any]:
    """Weekly workflow for retraining ML models."""
    logger = get_run_logger()
    
    logger.info("Starting weekly model retraining")
    
    try:
        # Extract all districts
        district_codes = extract_districts()
        
        retraining_results = {}
        
        for district_code in district_codes:
            # Extract and validate data
            data = extract_data(district_code)
            if data is None:
                continue
            
            if not validate_data(data, district_code):
                continue
            
            # Retrain models
            models_to_retrain = ["XGBoost", "Prophet", "ARIMA"]
            
            for model_name in models_to_retrain:
                try:
                    model_info = train_model(data, district_code, model_name)
                    if model_info:
                        retraining_results[f"{district_code}_{model_name}"] = model_info
                except Exception as e:
                    logger.error(f"Failed to retrain {model_name} for {district_code}: {e}")
        
        logger.info(f"Weekly retraining completed: {len(retraining_results)} models retrained")
        return {
            'models_retrained': len(retraining_results),
            'results': retraining_results,
            'status': 'completed'
        }
        
    except Exception as e:
        logger.error(f"Weekly retraining failed: {e}")
        return {
            'models_retrained': 0,
            'results': {},
            'status': 'failed',
            'error': str(e)
        }


if __name__ == "__main__":
    # Run the daily workflow
    daily_prediction_workflow()