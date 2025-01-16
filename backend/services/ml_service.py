"""
Machine Learning service for cholera risk prediction.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from datetime import date, datetime, timedelta
from sqlalchemy.orm import Session
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ML libraries
import xgboost as xgb
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

from models import District, CholeraCase, ClimateData, WaterExtent, RiskPrediction
from services.data_service import DataService

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")


class MLService:
    """Service for machine learning operations."""
    
    def __init__(self):
        self.data_service = DataService()
        self.scaler = StandardScaler()
        
    def generate_predictions(
        self, 
        district_code: Optional[str] = None,
        horizon_weeks: int = 8,
        db: Session = None
    ) -> Dict[str, Any]:
        """Generate risk predictions for districts."""
        
        # Get districts to predict for
        if district_code:
            districts = db.query(District).filter(District.district_code == district_code).all()
        else:
            districts = db.query(District).all()
        
        results = {}
        
        for district in districts:
            try:
                # Get historical data for the district
                data = self.data_service.prepare_training_data(district.district_code, db)
                
                if data is None or len(data) < 30:  # Need at least 30 days of data
                    continue
                
                # Generate predictions using different models
                lstm_pred = self._predict_lstm(data, horizon_weeks)
                xgb_pred = self._predict_xgboost(data, horizon_weeks)
                prophet_pred = self._predict_prophet(data, horizon_weeks)
                arima_pred = self._predict_arima(data, horizon_weeks)
                
                # Ensemble prediction (simple average)
                ensemble_pred = np.mean([lstm_pred, xgb_pred, prophet_pred, arima_pred])
                
                # Save predictions to database
                prediction_date = date.today()
                
                for model_name, pred in [
                    ("LSTM", lstm_pred),
                    ("XGBoost", xgb_pred),
                    ("Prophet", prophet_pred),
                    ("ARIMA", arima_pred),
                    ("Ensemble", ensemble_pred)
                ]:
                    risk_prediction = RiskPrediction(
                        district_code=district.district_code,
                        prediction_date=prediction_date,
                        prediction_horizon_weeks=horizon_weeks,
                        risk_score=float(pred),
                        confidence_interval_lower=float(pred * 0.8),
                        confidence_interval_upper=float(pred * 1.2),
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
                
                results[district.district_code] = {
                    "ensemble_prediction": float(ensemble_pred),
                    "status": "success"
                }
                
            except Exception as e:
                results[district.district_code] = {
                    "error": str(e),
                    "status": "failed"
                }
        
        db.commit()
        return results
    
    def _predict_lstm(self, data: pd.DataFrame, horizon_weeks: int) -> float:
        """Predict using LSTM model."""
        try:
            with mlflow.start_run(run_name="lstm_prediction"):
                # Prepare data for LSTM
                features = ['rainfall_mm', 'temperature_celsius', 'cases_count', 'water_area_sqkm']
                X = data[features].fillna(0).values
                y = data['cases_count'].values
                
                # Create sequences
                seq_length = 14
                X_seq, y_seq = [], []
                for i in range(seq_length, len(X)):
                    X_seq.append(X[i-seq_length:i])
                    y_seq.append(y[i])
                
                X_seq, y_seq = np.array(X_seq), np.array(y_seq)
                
                # Split data
                split_idx = int(0.8 * len(X_seq))
                X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
                y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
                
                # Build LSTM model
                model = Sequential([
                    LSTM(50, return_sequences=True, input_shape=(seq_length, len(features))),
                    Dropout(0.2),
                    LSTM(50, return_sequences=False),
                    Dropout(0.2),
                    Dense(25),
                    Dense(1)
                ])
                
                model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
                
                # Train model
                model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=0)
                
                # Make prediction
                last_sequence = X_seq[-1:].reshape(1, seq_length, len(features))
                prediction = model.predict(last_sequence, verbose=0)[0][0]
                
                # Log metrics
                y_pred = model.predict(X_test, verbose=0).flatten()
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("mse", mse)
                mlflow.log_metric("r2", r2)
                mlflow.log_param("seq_length", seq_length)
                mlflow.log_param("horizon_weeks", horizon_weeks)
                
                # Log model
                mlflow.tensorflow.log_model(model, "lstm_model")
                
                return max(0, min(100, prediction))  # Clamp between 0 and 100
                
        except Exception as e:
            print(f"LSTM prediction error: {e}")
            return 50.0  # Default risk score
    
    def _predict_xgboost(self, data: pd.DataFrame, horizon_weeks: int) -> float:
        """Predict using XGBoost model."""
        try:
            with mlflow.start_run(run_name="xgboost_prediction"):
                # Prepare features
                features = ['rainfall_mm', 'temperature_celsius', 'cases_count', 'water_area_sqkm']
                
                # Create lagged features
                for feature in features:
                    for lag in [1, 7, 14, 21]:
                        data[f'{feature}_lag_{lag}'] = data[feature].shift(lag)
                
                # Create rolling features
                for feature in features:
                    data[f'{feature}_roll_7'] = data[feature].rolling(window=7).mean()
                    data[f'{feature}_roll_14'] = data[feature].rolling(window=14).mean()
                
                # Prepare training data
                feature_cols = [col for col in data.columns if col not in ['date', 'cases_count']]
                X = data[feature_cols].fillna(0)
                y = data['cases_count'].fillna(0)
                
                # Split data
                split_idx = int(0.8 * len(X))
                X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
                
                # Train XGBoost model
                model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                )
                
                model.fit(X_train, y_train)
                
                # Make prediction
                last_features = X.iloc[-1:].values
                prediction = model.predict(last_features)[0]
                
                # Log metrics
                y_pred = model.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("mse", mse)
                mlflow.log_metric("r2", r2)
                mlflow.log_param("n_estimators", 100)
                mlflow.log_param("max_depth", 6)
                mlflow.log_param("horizon_weeks", horizon_weeks)
                
                # Log model
                mlflow.sklearn.log_model(model, "xgboost_model")
                
                return max(0, min(100, prediction))  # Clamp between 0 and 100
                
        except Exception as e:
            print(f"XGBoost prediction error: {e}")
            return 50.0  # Default risk score
    
    def _predict_prophet(self, data: pd.DataFrame, horizon_weeks: int) -> float:
        """Predict using Prophet model."""
        try:
            with mlflow.start_run(run_name="prophet_prediction"):
                # Prepare data for Prophet
                prophet_data = data[['date', 'cases_count']].copy()
                prophet_data.columns = ['ds', 'y']
                prophet_data = prophet_data.dropna()
                
                if len(prophet_data) < 30:
                    return 50.0
                
                # Create and train Prophet model
                model = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=False,
                    seasonality_mode='multiplicative'
                )
                
                # Add external regressors
                for col in ['rainfall_mm', 'temperature_celsius', 'water_area_sqkm']:
                    if col in data.columns:
                        regressor_data = data[['date', col]].copy()
                        regressor_data.columns = ['ds', col]
                        regressor_data = regressor_data.dropna()
                        
                        # Merge with main data
                        prophet_data = prophet_data.merge(regressor_data, on='ds', how='left')
                        prophet_data[col] = prophet_data[col].fillna(prophet_data[col].mean())
                        
                        model.add_regressor(col)
                
                model.fit(prophet_data)
                
                # Create future dataframe
                future = model.make_future_dataframe(periods=horizon_weeks * 7)
                
                # Add external regressors to future
                for col in ['rainfall_mm', 'temperature_celsius', 'water_area_sqkm']:
                    if col in data.columns:
                        future[col] = data[col].mean()
                
                # Make prediction
                forecast = model.predict(future)
                prediction = forecast['yhat'].iloc[-1]
                
                # Calculate metrics
                train_forecast = forecast['yhat'].iloc[:len(prophet_data)]
                mae = mean_absolute_error(prophet_data['y'], train_forecast)
                mse = mean_squared_error(prophet_data['y'], train_forecast)
                
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("mse", mse)
                mlflow.log_param("horizon_weeks", horizon_weeks)
                
                return max(0, min(100, prediction))  # Clamp between 0 and 100
                
        except Exception as e:
            print(f"Prophet prediction error: {e}")
            return 50.0  # Default risk score
    
    def _predict_arima(self, data: pd.DataFrame, horizon_weeks: int) -> float:
        """Predict using ARIMA model."""
        try:
            with mlflow.start_run(run_name="arima_prediction"):
                # Prepare time series data
                ts_data = data['cases_count'].fillna(0)
                
                if len(ts_data) < 30:
                    return 50.0
                
                # Fit ARIMA model
                model = ARIMA(ts_data, order=(1, 1, 1))
                fitted_model = model.fit()
                
                # Make prediction
                forecast = fitted_model.forecast(steps=horizon_weeks * 7)
                prediction = forecast.iloc[-1]
                
                # Calculate metrics
                fitted_values = fitted_model.fittedvalues
                residuals = ts_data - fitted_values
                mae = np.mean(np.abs(residuals))
                mse = np.mean(residuals**2)
                
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("mse", mse)
                mlflow.log_param("order", "(1,1,1)")
                mlflow.log_param("horizon_weeks", horizon_weeks)
                
                return max(0, min(100, prediction))  # Clamp between 0 and 100
                
        except Exception as e:
            print(f"ARIMA prediction error: {e}")
            return 50.0  # Default risk score