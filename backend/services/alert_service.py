"""
Alert service for generating and managing cholera risk alerts.
"""

import uuid
from typing import List, Optional, Dict, Any
from datetime import date, datetime, timedelta
from decimal import Decimal
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

from models import Alert, RiskPrediction, District


class AlertService:
    """Service for managing cholera risk alerts."""
    
    def __init__(self):
        self.risk_thresholds = {
            'low': 25.0,
            'medium': 50.0,
            'high': 75.0,
            'critical': 90.0
        }
    
    def generate_alerts(self, db: Session) -> Dict[str, Any]:
        """Generate new alerts based on recent risk predictions."""
        try:
            # Get recent predictions (last 7 days)
            recent_date = date.today() - timedelta(days=7)
            
            recent_predictions = db.query(RiskPrediction).filter(
                and_(
                    RiskPrediction.prediction_date >= recent_date,
                    RiskPrediction.model_name == 'Ensemble'  # Use ensemble predictions
                )
            ).all()
            
            alerts_generated = []
            
            for prediction in recent_predictions:
                # Check if alert already exists for this prediction
                existing_alert = db.query(Alert).filter(
                    and_(
                        Alert.district_code == prediction.district_code,
                        Alert.prediction_horizon_weeks == prediction.prediction_horizon_weeks,
                        Alert.triggered_at >= recent_date,
                        Alert.status == 'active'
                    )
                ).first()
                
                if existing_alert:
                    continue
                
                # Determine alert severity based on risk score
                risk_score = float(prediction.risk_score) if prediction.risk_score else 0
                severity = self._get_severity_level(risk_score)
                
                # Only generate alerts for medium, high, or critical risk
                if severity in ['medium', 'high', 'critical']:
                    alert = Alert(
                        alert_id=str(uuid.uuid4()),
                        district_code=prediction.district_code,
                        alert_type='cholera_risk',
                        severity=severity,
                        risk_score=prediction.risk_score,
                        prediction_horizon_weeks=prediction.prediction_horizon_weeks,
                        triggered_at=datetime.utcnow(),
                        status='active'
                    )
                    
                    db.add(alert)
                    alerts_generated.append({
                        'alert_id': alert.alert_id,
                        'district_code': alert.district_code,
                        'severity': alert.severity,
                        'risk_score': float(alert.risk_score),
                        'prediction_horizon_weeks': alert.prediction_horizon_weeks
                    })
            
            db.commit()
            
            return {
                'alerts_generated': len(alerts_generated),
                'alerts': alerts_generated,
                'status': 'success'
            }
            
        except Exception as e:
            print(f"Error generating alerts: {e}")
            return {
                'alerts_generated': 0,
                'alerts': [],
                'status': 'error',
                'error': str(e)
            }
    
    def get_active_alerts(self, db: Session, district_code: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get active alerts, optionally filtered by district."""
        try:
            query = db.query(Alert).filter(Alert.status == 'active')
            
            if district_code:
                query = query.filter(Alert.district_code == district_code)
            
            alerts = query.order_by(Alert.triggered_at.desc()).all()
            
            return [
                {
                    'alert_id': alert.alert_id,
                    'district_code': alert.district_code,
                    'alert_type': alert.alert_type,
                    'severity': alert.severity,
                    'risk_score': float(alert.risk_score) if alert.risk_score else 0,
                    'prediction_horizon_weeks': alert.prediction_horizon_weeks,
                    'triggered_at': alert.triggered_at,
                    'status': alert.status
                }
                for alert in alerts
            ]
            
        except Exception as e:
            print(f"Error getting active alerts: {e}")
            return []
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str, db: Session) -> bool:
        """Acknowledge an alert."""
        try:
            alert = db.query(Alert).filter(Alert.alert_id == alert_id).first()
            
            if not alert:
                return False
            
            alert.acknowledged_at = datetime.utcnow()
            alert.acknowledged_by = acknowledged_by
            alert.status = 'acknowledged'
            
            db.commit()
            return True
            
        except Exception as e:
            print(f"Error acknowledging alert {alert_id}: {e}")
            return False
    
    def resolve_alert(self, alert_id: str, db: Session) -> bool:
        """Resolve an alert."""
        try:
            alert = db.query(Alert).filter(Alert.alert_id == alert_id).first()
            
            if not alert:
                return False
            
            alert.status = 'resolved'
            
            db.commit()
            return True
            
        except Exception as e:
            print(f"Error resolving alert {alert_id}: {e}")
            return False
    
    def get_alert_statistics(self, db: Session) -> Dict[str, Any]:
        """Get alert statistics."""
        try:
            total_alerts = db.query(Alert).count()
            active_alerts = db.query(Alert).filter(Alert.status == 'active').count()
            acknowledged_alerts = db.query(Alert).filter(Alert.status == 'acknowledged').count()
            resolved_alerts = db.query(Alert).filter(Alert.status == 'resolved').count()
            
            # Alerts by severity
            severity_counts = {}
            for severity in ['low', 'medium', 'high', 'critical']:
                count = db.query(Alert).filter(
                    and_(Alert.severity == severity, Alert.status == 'active')
                ).count()
                severity_counts[severity] = count
            
            # Recent alerts (last 30 days)
            recent_date = datetime.utcnow() - timedelta(days=30)
            recent_alerts = db.query(Alert).filter(
                Alert.triggered_at >= recent_date
            ).count()
            
            return {
                'total_alerts': total_alerts,
                'active_alerts': active_alerts,
                'acknowledged_alerts': acknowledged_alerts,
                'resolved_alerts': resolved_alerts,
                'severity_counts': severity_counts,
                'recent_alerts_30d': recent_alerts
            }
            
        except Exception as e:
            print(f"Error getting alert statistics: {e}")
            return {}
    
    def get_district_alerts(self, district_code: str, days: int = 30, db: Session = None) -> List[Dict[str, Any]]:
        """Get alerts for a specific district."""
        try:
            start_date = datetime.utcnow() - timedelta(days=days)
            
            alerts = db.query(Alert).filter(
                and_(
                    Alert.district_code == district_code,
                    Alert.triggered_at >= start_date
                )
            ).order_by(Alert.triggered_at.desc()).all()
            
            return [
                {
                    'alert_id': alert.alert_id,
                    'alert_type': alert.alert_type,
                    'severity': alert.severity,
                    'risk_score': float(alert.risk_score) if alert.risk_score else 0,
                    'prediction_horizon_weeks': alert.prediction_horizon_weeks,
                    'triggered_at': alert.triggered_at,
                    'acknowledged_at': alert.acknowledged_at,
                    'acknowledged_by': alert.acknowledged_by,
                    'status': alert.status
                }
                for alert in alerts
            ]
            
        except Exception as e:
            print(f"Error getting district alerts for {district_code}: {e}")
            return []
    
    def _get_severity_level(self, risk_score: float) -> str:
        """Determine severity level based on risk score."""
        if risk_score >= self.risk_thresholds['critical']:
            return 'critical'
        elif risk_score >= self.risk_thresholds['high']:
            return 'high'
        elif risk_score >= self.risk_thresholds['medium']:
            return 'medium'
        else:
            return 'low'
    
    def update_risk_thresholds(self, thresholds: Dict[str, float]):
        """Update risk thresholds for alert generation."""
        self.risk_thresholds.update(thresholds)
    
    def get_risk_thresholds(self) -> Dict[str, float]:
        """Get current risk thresholds."""
        return self.risk_thresholds.copy()