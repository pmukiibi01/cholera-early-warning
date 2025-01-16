"""
FastAPI application for Cholera Early Warning System.
"""

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional
import uvicorn
import os

from database import get_db, create_tables
from models import District, CholeraCase, ClimateData, RiskPrediction, Alert
from schemas import (
    DistrictCreate, DistrictResponse,
    CholeraCaseCreate, CholeraCaseResponse,
    ClimateDataCreate, ClimateDataResponse,
    RiskPredictionResponse,
    AlertResponse
)
from services.ml_service import MLService
from services.alert_service import AlertService

# Initialize FastAPI app
app = FastAPI(
    title="Cholera Early Warning System",
    description="Predict district-level cholera risk 8-12 weeks ahead",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
ml_service = MLService()
alert_service = AlertService()

@app.on_event("startup")
async def startup_event():
    """Initialize database tables on startup."""
    create_tables()

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Cholera Early Warning System API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

# District endpoints
@app.post("/districts/", response_model=DistrictResponse)
async def create_district(district: DistrictCreate, db: Session = Depends(get_db)):
    """Create a new district."""
    db_district = District(**district.dict())
    db.add(db_district)
    db.commit()
    db.refresh(db_district)
    return db_district

@app.get("/districts/", response_model=List[DistrictResponse])
async def get_districts(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """Get all districts."""
    districts = db.query(District).offset(skip).limit(limit).all()
    return districts

@app.get("/districts/{district_code}", response_model=DistrictResponse)
async def get_district(district_code: str, db: Session = Depends(get_db)):
    """Get a specific district by code."""
    district = db.query(District).filter(District.district_code == district_code).first()
    if not district:
        raise HTTPException(status_code=404, detail="District not found")
    return district

# Cholera cases endpoints
@app.post("/cholera-cases/", response_model=CholeraCaseResponse)
async def create_cholera_case(case: CholeraCaseCreate, db: Session = Depends(get_db)):
    """Create a new cholera case."""
    db_case = CholeraCase(**case.dict())
    db.add(db_case)
    db.commit()
    db.refresh(db_case)
    return db_case

@app.get("/cholera-cases/", response_model=List[CholeraCaseResponse])
async def get_cholera_cases(
    district_code: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get cholera cases with optional district filter."""
    query = db.query(CholeraCase)
    if district_code:
        query = query.filter(CholeraCase.district_code == district_code)
    cases = query.offset(skip).limit(limit).all()
    return cases

# Climate data endpoints
@app.post("/climate-data/", response_model=ClimateDataResponse)
async def create_climate_data(data: ClimateDataCreate, db: Session = Depends(get_db)):
    """Create new climate data."""
    db_data = ClimateData(**data.dict())
    db.add(db_data)
    db.commit()
    db.refresh(db_data)
    return db_data

@app.get("/climate-data/", response_model=List[ClimateDataResponse])
async def get_climate_data(
    district_code: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get climate data with optional district filter."""
    query = db.query(ClimateData)
    if district_code:
        query = query.filter(ClimateData.district_code == district_code)
    data = query.offset(skip).limit(limit).all()
    return data

# Risk prediction endpoints
@app.get("/risk-predictions/", response_model=List[RiskPredictionResponse])
async def get_risk_predictions(
    district_code: Optional[str] = None,
    model_name: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get risk predictions with optional filters."""
    query = db.query(RiskPrediction)
    if district_code:
        query = query.filter(RiskPrediction.district_code == district_code)
    if model_name:
        query = query.filter(RiskPrediction.model_name == model_name)
    predictions = query.offset(skip).limit(limit).all()
    return predictions

@app.post("/risk-predictions/generate")
async def generate_risk_predictions(
    background_tasks: BackgroundTasks,
    district_code: Optional[str] = None,
    horizon_weeks: int = 8,
    db: Session = Depends(get_db)
):
    """Generate new risk predictions for all districts or a specific district."""
    background_tasks.add_task(
        ml_service.generate_predictions,
        district_code=district_code,
        horizon_weeks=horizon_weeks,
        db=db
    )
    return {"message": "Risk prediction generation started"}

# Alert endpoints
@app.get("/alerts/", response_model=List[AlertResponse])
async def get_alerts(
    district_code: Optional[str] = None,
    status: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get alerts with optional filters."""
    query = db.query(Alert)
    if district_code:
        query = query.filter(Alert.district_code == district_code)
    if status:
        query = query.filter(Alert.status == status)
    alerts = query.offset(skip).limit(limit).all()
    return alerts

@app.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    acknowledged_by: str,
    db: Session = Depends(get_db)
):
    """Acknowledge an alert."""
    alert = db.query(Alert).filter(Alert.alert_id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    alert.acknowledged_at = datetime.utcnow()
    alert.acknowledged_by = acknowledged_by
    alert.status = "acknowledged"
    db.commit()
    
    return {"message": "Alert acknowledged successfully"}

# Dashboard endpoints
@app.get("/dashboard/summary")
async def get_dashboard_summary(db: Session = Depends(get_db)):
    """Get dashboard summary statistics."""
    total_districts = db.query(District).count()
    total_cases = db.query(CholeraCase).count()
    active_alerts = db.query(Alert).filter(Alert.status == "active").count()
    
    # Get recent predictions
    recent_predictions = db.query(RiskPrediction).order_by(
        RiskPrediction.prediction_date.desc()
    ).limit(10).all()
    
    return {
        "total_districts": total_districts,
        "total_cases": total_cases,
        "active_alerts": active_alerts,
        "recent_predictions": recent_predictions
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )