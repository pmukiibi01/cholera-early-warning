"""
Data service for preparing and aggregating data for ML models.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from datetime import date, datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_

from models import District, CholeraCase, ClimateData, WaterExtent


class DataService:
    """Service for data preparation and aggregation."""
    
    def prepare_training_data(self, district_code: str, db: Session) -> Optional[pd.DataFrame]:
        """Prepare training data for a specific district."""
        try:
            # Get date range for data collection
            end_date = date.today()
            start_date = end_date - timedelta(days=365)  # Last year of data
            
            # Get cholera cases data
            cases_query = db.query(
                CholeraCase.case_date,
                func.count(CholeraCase.id).label('cases_count')
            ).filter(
                and_(
                    CholeraCase.district_code == district_code,
                    CholeraCase.case_date >= start_date,
                    CholeraCase.case_date <= end_date
                )
            ).group_by(CholeraCase.case_date).order_by(CholeraCase.case_date)
            
            cases_data = pd.DataFrame(cases_query.all(), columns=['date', 'cases_count'])
            
            # Get climate data
            climate_query = db.query(
                ClimateData.date,
                ClimateData.rainfall_mm,
                ClimateData.temperature_celsius,
                ClimateData.humidity_percent
            ).filter(
                and_(
                    ClimateData.district_code == district_code,
                    ClimateData.date >= start_date,
                    ClimateData.date <= end_date
                )
            ).order_by(ClimateData.date)
            
            climate_data = pd.DataFrame(climate_query.all())
            
            # Get water extent data
            water_query = db.query(
                WaterExtent.date,
                WaterExtent.water_area_sqkm,
                WaterExtent.flood_risk_score
            ).filter(
                and_(
                    WaterExtent.district_code == district_code,
                    WaterExtent.date >= start_date,
                    WaterExtent.date <= end_date
                )
            ).order_by(WaterExtent.date)
            
            water_data = pd.DataFrame(water_query.all())
            
            # Create complete date range
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            complete_df = pd.DataFrame({'date': date_range})
            
            # Merge all data
            complete_df = complete_df.merge(cases_data, on='date', how='left')
            complete_df = complete_df.merge(climate_data, on='date', how='left')
            complete_df = complete_df.merge(water_data, on='date', how='left')
            
            # Fill missing values
            complete_df['cases_count'] = complete_df['cases_count'].fillna(0)
            complete_df['rainfall_mm'] = complete_df['rainfall_mm'].fillna(
                complete_df['rainfall_mm'].mean()
            )
            complete_df['temperature_celsius'] = complete_df['temperature_celsius'].fillna(
                complete_df['temperature_celsius'].mean()
            )
            complete_df['humidity_percent'] = complete_df['humidity_percent'].fillna(
                complete_df['humidity_percent'].mean()
            )
            complete_df['water_area_sqkm'] = complete_df['water_area_sqkm'].fillna(
                complete_df['water_area_sqkm'].mean()
            )
            complete_df['flood_risk_score'] = complete_df['flood_risk_score'].fillna(0)
            
            # Add derived features
            complete_df['day_of_week'] = complete_df['date'].dt.dayofweek
            complete_df['month'] = complete_df['date'].dt.month
            complete_df['year'] = complete_df['date'].dt.year
            
            # Add lagged features
            for lag in [1, 7, 14, 21]:
                complete_df[f'cases_lag_{lag}'] = complete_df['cases_count'].shift(lag)
                complete_df[f'rainfall_lag_{lag}'] = complete_df['rainfall_mm'].shift(lag)
                complete_df[f'temperature_lag_{lag}'] = complete_df['temperature_celsius'].shift(lag)
            
            # Add rolling averages
            for window in [7, 14, 30]:
                complete_df[f'cases_roll_{window}'] = complete_df['cases_count'].rolling(
                    window=window, min_periods=1
                ).mean()
                complete_df[f'rainfall_roll_{window}'] = complete_df['rainfall_mm'].rolling(
                    window=window, min_periods=1
                ).mean()
                complete_df[f'temperature_roll_{window}'] = complete_df['temperature_celsius'].rolling(
                    window=window, min_periods=1
                ).mean()
            
            # Add seasonal features
            complete_df['is_rainy_season'] = complete_df['month'].isin([3, 4, 5, 9, 10, 11]).astype(int)
            complete_df['is_dry_season'] = complete_df['month'].isin([12, 1, 2, 6, 7, 8]).astype(int)
            
            return complete_df
            
        except Exception as e:
            print(f"Error preparing training data for {district_code}: {e}")
            return None
    
    def get_district_statistics(self, district_code: str, db: Session) -> Dict[str, Any]:
        """Get statistics for a specific district."""
        try:
            # Get district info
            district = db.query(District).filter(District.district_code == district_code).first()
            if not district:
                return {}
            
            # Get case statistics
            total_cases = db.query(CholeraCase).filter(
                CholeraCase.district_code == district_code
            ).count()
            
            recent_cases = db.query(CholeraCase).filter(
                and_(
                    CholeraCase.district_code == district_code,
                    CholeraCase.case_date >= date.today() - timedelta(days=30)
                )
            ).count()
            
            # Get climate statistics
            recent_climate = db.query(
                func.avg(ClimateData.rainfall_mm).label('avg_rainfall'),
                func.avg(ClimateData.temperature_celsius).label('avg_temperature'),
                func.avg(ClimateData.humidity_percent).label('avg_humidity')
            ).filter(
                and_(
                    ClimateData.district_code == district_code,
                    ClimateData.date >= date.today() - timedelta(days=30)
                )
            ).first()
            
            return {
                'district_name': district.district_name,
                'region': district.region,
                'population': district.population,
                'total_cases': total_cases,
                'recent_cases_30d': recent_cases,
                'avg_rainfall_30d': float(recent_climate.avg_rainfall) if recent_climate.avg_rainfall else 0,
                'avg_temperature_30d': float(recent_climate.avg_temperature) if recent_climate.avg_temperature else 0,
                'avg_humidity_30d': float(recent_climate.avg_humidity) if recent_climate.avg_humidity else 0
            }
            
        except Exception as e:
            print(f"Error getting district statistics for {district_code}: {e}")
            return {}
    
    def get_national_statistics(self, db: Session) -> Dict[str, Any]:
        """Get national-level statistics."""
        try:
            # Get total districts
            total_districts = db.query(District).count()
            
            # Get total cases
            total_cases = db.query(CholeraCase).count()
            
            # Get recent cases (last 30 days)
            recent_cases = db.query(CholeraCase).filter(
                CholeraCase.case_date >= date.today() - timedelta(days=30)
            ).count()
            
            # Get cases by region
            cases_by_region = db.query(
                District.region,
                func.count(CholeraCase.id).label('case_count')
            ).join(CholeraCase).filter(
                CholeraCase.case_date >= date.today() - timedelta(days=30)
            ).group_by(District.region).all()
            
            # Get top affected districts
            top_districts = db.query(
                District.district_name,
                func.count(CholeraCase.id).label('case_count')
            ).join(CholeraCase).filter(
                CholeraCase.case_date >= date.today() - timedelta(days=30)
            ).group_by(District.district_name).order_by(
                func.count(CholeraCase.id).desc()
            ).limit(10).all()
            
            return {
                'total_districts': total_districts,
                'total_cases': total_cases,
                'recent_cases_30d': recent_cases,
                'cases_by_region': [{'region': r[0], 'cases': r[1]} for r in cases_by_region],
                'top_affected_districts': [{'district': d[0], 'cases': d[1]} for d in top_districts]
            }
            
        except Exception as e:
            print(f"Error getting national statistics: {e}")
            return {}
    
    def get_time_series_data(self, district_code: str, days: int = 90, db: Session = None) -> pd.DataFrame:
        """Get time series data for a specific district."""
        try:
            end_date = date.today()
            start_date = end_date - timedelta(days=days)
            
            # Get cases data
            cases_query = db.query(
                CholeraCase.case_date,
                func.count(CholeraCase.id).label('cases_count')
            ).filter(
                and_(
                    CholeraCase.district_code == district_code,
                    CholeraCase.case_date >= start_date,
                    CholeraCase.case_date <= end_date
                )
            ).group_by(CholeraCase.case_date).order_by(CholeraCase.case_date)
            
            cases_data = pd.DataFrame(cases_query.all(), columns=['date', 'cases_count'])
            
            # Get climate data
            climate_query = db.query(
                ClimateData.date,
                ClimateData.rainfall_mm,
                ClimateData.temperature_celsius,
                ClimateData.humidity_percent
            ).filter(
                and_(
                    ClimateData.district_code == district_code,
                    ClimateData.date >= start_date,
                    ClimateData.date <= end_date
                )
            ).order_by(ClimateData.date)
            
            climate_data = pd.DataFrame(climate_query.all())
            
            # Create complete date range
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            complete_df = pd.DataFrame({'date': date_range})
            
            # Merge data
            complete_df = complete_df.merge(cases_data, on='date', how='left')
            complete_df = complete_df.merge(climate_data, on='date', how='left')
            
            # Fill missing values
            complete_df['cases_count'] = complete_df['cases_count'].fillna(0)
            complete_df['rainfall_mm'] = complete_df['rainfall_mm'].fillna(0)
            complete_df['temperature_celsius'] = complete_df['temperature_celsius'].fillna(0)
            complete_df['humidity_percent'] = complete_df['humidity_percent'].fillna(0)
            
            return complete_df
            
        except Exception as e:
            print(f"Error getting time series data for {district_code}: {e}")
            return pd.DataFrame()