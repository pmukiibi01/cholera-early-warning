"""
Pydantic schemas for API request/response models.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import date, datetime
from decimal import Decimal


# District schemas
class DistrictBase(BaseModel):
    district_code: str = Field(..., description="Unique district code")
    district_name: str = Field(..., description="District name")
    region: Optional[str] = Field(None, description="Region name")
    population: Optional[int] = Field(None, description="District population")


class DistrictCreate(DistrictBase):
    pass


class DistrictResponse(DistrictBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# Cholera case schemas
class CholeraCaseBase(BaseModel):
    case_id: str = Field(..., description="Unique case identifier")
    district_code: str = Field(..., description="District code")
    facility_code: Optional[str] = Field(None, description="Health facility code")
    case_date: date = Field(..., description="Date of case")
    age_group: Optional[str] = Field(None, description="Age group")
    gender: Optional[str] = Field(None, description="Gender")
    case_status: Optional[str] = Field(None, description="Case status")
    lab_confirmed: bool = Field(False, description="Laboratory confirmed")
    outcome: Optional[str] = Field(None, description="Case outcome")


class CholeraCaseCreate(CholeraCaseBase):
    pass


class CholeraCaseResponse(CholeraCaseBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# Climate data schemas
class ClimateDataBase(BaseModel):
    district_code: str = Field(..., description="District code")
    date: date = Field(..., description="Date of observation")
    rainfall_mm: Optional[Decimal] = Field(None, description="Rainfall in mm")
    temperature_celsius: Optional[Decimal] = Field(None, description="Temperature in Celsius")
    humidity_percent: Optional[Decimal] = Field(None, description="Humidity percentage")
    data_source: Optional[str] = Field(None, description="Data source")


class ClimateDataCreate(ClimateDataBase):
    pass


class ClimateDataResponse(ClimateDataBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# Water extent schemas
class WaterExtentBase(BaseModel):
    district_code: str = Field(..., description="District code")
    date: date = Field(..., description="Date of observation")
    water_area_sqkm: Optional[Decimal] = Field(None, description="Water area in square km")
    flood_risk_score: Optional[Decimal] = Field(None, description="Flood risk score")
    data_source: Optional[str] = Field(None, description="Data source")


class WaterExtentCreate(WaterExtentBase):
    pass


class WaterExtentResponse(WaterExtentBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# Risk prediction schemas
class RiskPredictionBase(BaseModel):
    district_code: str = Field(..., description="District code")
    prediction_date: date = Field(..., description="Date of prediction")
    prediction_horizon_weeks: int = Field(..., description="Prediction horizon in weeks")
    risk_score: Optional[Decimal] = Field(None, description="Risk score (0-100)")
    confidence_interval_lower: Optional[Decimal] = Field(None, description="Lower confidence interval")
    confidence_interval_upper: Optional[Decimal] = Field(None, description="Upper confidence interval")
    model_name: Optional[str] = Field(None, description="Model name used")
    model_version: Optional[str] = Field(None, description="Model version")
    features_used: Optional[Dict[str, Any]] = Field(None, description="Features used in prediction")


class RiskPredictionCreate(RiskPredictionBase):
    pass


class RiskPredictionResponse(RiskPredictionBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# Alert schemas
class AlertBase(BaseModel):
    alert_id: str = Field(..., description="Unique alert identifier")
    district_code: str = Field(..., description="District code")
    alert_type: Optional[str] = Field(None, description="Alert type")
    severity: Optional[str] = Field(None, description="Alert severity")
    risk_score: Optional[Decimal] = Field(None, description="Risk score")
    prediction_horizon_weeks: Optional[int] = Field(None, description="Prediction horizon")
    triggered_at: Optional[datetime] = Field(None, description="When alert was triggered")
    acknowledged_at: Optional[datetime] = Field(None, description="When alert was acknowledged")
    acknowledged_by: Optional[str] = Field(None, description="Who acknowledged the alert")
    status: Optional[str] = Field("active", description="Alert status")


class AlertCreate(AlertBase):
    pass


class AlertResponse(AlertBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# Dashboard schemas
class DashboardSummary(BaseModel):
    total_districts: int
    total_cases: int
    active_alerts: int
    recent_predictions: List[RiskPredictionResponse]


# Health facility schemas
class HealthFacilityBase(BaseModel):
    facility_code: str = Field(..., description="Unique facility code")
    facility_name: str = Field(..., description="Facility name")
    district_code: str = Field(..., description="District code")
    facility_type: Optional[str] = Field(None, description="Facility type")
    latitude: Optional[Decimal] = Field(None, description="Latitude")
    longitude: Optional[Decimal] = Field(None, description="Longitude")


class HealthFacilityCreate(HealthFacilityBase):
    pass


class HealthFacilityResponse(HealthFacilityBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# Model training schemas
class ModelTrainingRequest(BaseModel):
    model_name: str = Field(..., description="Name of the model to train")
    district_codes: Optional[List[str]] = Field(None, description="Specific districts to train on")
    start_date: Optional[date] = Field(None, description="Training start date")
    end_date: Optional[date] = Field(None, description="Training end date")


class ModelTrainingResponse(BaseModel):
    model_name: str
    status: str
    message: str
    run_id: Optional[str] = None