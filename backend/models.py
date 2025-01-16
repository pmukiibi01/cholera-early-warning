"""
SQLAlchemy models for Cholera Early Warning System.
"""

from sqlalchemy import Column, Integer, String, Date, DateTime, Boolean, Decimal, Text, ForeignKey, JSON
from sqlalchemy.orm import relationship
from geoalchemy2 import Geometry
from database import Base
from datetime import datetime


class District(Base):
    __tablename__ = "districts"
    __table_args__ = {'schema': 'cholera_ew'}

    id = Column(Integer, primary_key=True, index=True)
    district_code = Column(String(20), unique=True, index=True, nullable=False)
    district_name = Column(String(100), nullable=False)
    region = Column(String(50))
    population = Column(Integer)
    geometry = Column(Geometry('MULTIPOLYGON', srid=4326))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    health_facilities = relationship("HealthFacility", back_populates="district")
    cholera_cases = relationship("CholeraCase", back_populates="district")
    climate_data = relationship("ClimateData", back_populates="district")
    water_extent = relationship("WaterExtent", back_populates="district")
    risk_predictions = relationship("RiskPrediction", back_populates="district")
    alerts = relationship("Alert", back_populates="district")


class HealthFacility(Base):
    __tablename__ = "health_facilities"
    __table_args__ = {'schema': 'cholera_ew'}

    id = Column(Integer, primary_key=True, index=True)
    facility_code = Column(String(50), unique=True, index=True, nullable=False)
    facility_name = Column(String(200), nullable=False)
    district_code = Column(String(20), ForeignKey('cholera_ew.districts.district_code'))
    facility_type = Column(String(50))
    latitude = Column(Decimal(10, 8))
    longitude = Column(Decimal(11, 8))
    geometry = Column(Geometry('POINT', srid=4326))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    district = relationship("District", back_populates="health_facilities")
    cholera_cases = relationship("CholeraCase", back_populates="facility")


class CholeraCase(Base):
    __tablename__ = "cholera_cases"
    __table_args__ = {'schema': 'cholera_ew'}

    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(String(50), unique=True, index=True, nullable=False)
    district_code = Column(String(20), ForeignKey('cholera_ew.districts.district_code'))
    facility_code = Column(String(50), ForeignKey('cholera_ew.health_facilities.facility_code'))
    case_date = Column(Date, nullable=False, index=True)
    age_group = Column(String(20))
    gender = Column(String(10))
    case_status = Column(String(20))
    lab_confirmed = Column(Boolean, default=False)
    outcome = Column(String(20))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    district = relationship("District", back_populates="cholera_cases")
    facility = relationship("HealthFacility", back_populates="cholera_cases")


class ClimateData(Base):
    __tablename__ = "climate_data"
    __table_args__ = {'schema': 'cholera_ew'}

    id = Column(Integer, primary_key=True, index=True)
    district_code = Column(String(20), ForeignKey('cholera_ew.districts.district_code'))
    date = Column(Date, nullable=False, index=True)
    rainfall_mm = Column(Decimal(8, 2))
    temperature_celsius = Column(Decimal(5, 2))
    humidity_percent = Column(Decimal(5, 2))
    data_source = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    district = relationship("District", back_populates="climate_data")


class WaterExtent(Base):
    __tablename__ = "water_extent"
    __table_args__ = {'schema': 'cholera_ew'}

    id = Column(Integer, primary_key=True, index=True)
    district_code = Column(String(20), ForeignKey('cholera_ew.districts.district_code'))
    date = Column(Date, nullable=False, index=True)
    water_area_sqkm = Column(Decimal(12, 4))
    flood_risk_score = Column(Decimal(5, 2))
    data_source = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    district = relationship("District", back_populates="water_extent")


class RiskPrediction(Base):
    __tablename__ = "risk_predictions"
    __table_args__ = {'schema': 'cholera_ew'}

    id = Column(Integer, primary_key=True, index=True)
    district_code = Column(String(20), ForeignKey('cholera_ew.districts.district_code'))
    prediction_date = Column(Date, nullable=False, index=True)
    prediction_horizon_weeks = Column(Integer, nullable=False)
    risk_score = Column(Decimal(5, 2))
    confidence_interval_lower = Column(Decimal(5, 2))
    confidence_interval_upper = Column(Decimal(5, 2))
    model_name = Column(String(50))
    model_version = Column(String(20))
    features_used = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    district = relationship("District", back_populates="risk_predictions")


class Alert(Base):
    __tablename__ = "alerts"
    __table_args__ = {'schema': 'cholera_ew'}

    id = Column(Integer, primary_key=True, index=True)
    alert_id = Column(String(50), unique=True, index=True, nullable=False)
    district_code = Column(String(20), ForeignKey('cholera_ew.districts.district_code'))
    alert_type = Column(String(50))
    severity = Column(String(20))
    risk_score = Column(Decimal(5, 2))
    prediction_horizon_weeks = Column(Integer)
    triggered_at = Column(DateTime, default=datetime.utcnow)
    acknowledged_at = Column(DateTime)
    acknowledged_by = Column(String(100))
    status = Column(String(20), default='active')
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    district = relationship("District", back_populates="alerts")