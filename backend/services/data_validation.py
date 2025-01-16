"""
Data validation service using Great Expectations.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, date
import great_expectations as ge
from great_expectations.core.batch import RuntimeBatchRequest
from great_expectations.core.yaml_handler import YAMLHandler
from great_expectations.data_context import BaseDataContext
from great_expectations.data_context.types.base import DataContextConfig, FilesystemStoreBackendDefaults
from sqlalchemy.orm import Session

from models import District, CholeraCase, ClimateData, WaterExtent, RiskPrediction

yaml = YAMLHandler()


class DataValidationService:
    """Service for data validation using Great Expectations."""
    
    def __init__(self, data_context_path: str = "./great_expectations"):
        self.data_context_path = data_context_path
        self.context = self._setup_data_context()
    
    def _setup_data_context(self) -> BaseDataContext:
        """Set up Great Expectations data context."""
        try:
            # Create data context configuration
            data_context_config = DataContextConfig(
                config_version=3.0,
                datasources={
                    "cholera_ew_datasource": {
                        "class_name": "Datasource",
                        "execution_engine": {
                            "class_name": "PandasExecutionEngine",
                        },
                        "data_connectors": {
                            "default_runtime_data_connector": {
                                "class_name": "RuntimeDataConnector",
                                "batch_identifiers": ["default_identifier_name"],
                            },
                        },
                    }
                },
                stores={
                    "expectations_store": {
                        "class_name": "ExpectationsStore",
                        "store_backend": {
                            "class_name": "TupleFilesystemStoreBackend",
                            "base_directory": f"{self.data_context_path}/expectations",
                        },
                    },
                    "validations_store": {
                        "class_name": "ValidationsStore",
                        "store_backend": {
                            "class_name": "TupleFilesystemStoreBackend",
                            "base_directory": f"{self.data_context_path}/validations",
                        },
                    },
                    "evaluation_parameter_store": {"class_name": "EvaluationParameterStore"},
                    "checkpoint_store": {
                        "class_name": "CheckpointStore",
                        "store_backend": {
                            "class_name": "TupleFilesystemStoreBackend",
                            "base_directory": f"{self.data_context_path}/checkpoints",
                        },
                    },
                },
                expectations_store_name="expectations_store",
                validations_store_name="validations_store",
                evaluation_parameter_store_name="evaluation_parameter_store",
                checkpoint_store_name="checkpoint_store",
                data_docs_sites={
                    "local_site": {
                        "class_name": "SiteBuilder",
                        "show_how_to_buttons": True,
                        "store_backend": {
                            "class_name": "TupleFilesystemStoreBackend",
                            "base_directory": f"{self.data_context_path}/data_docs/local_site",
                        },
                        "site_index_builder": {
                            "class_name": "DefaultSiteIndexBuilder",
                        },
                    }
                },
                validation_operators={
                    "action_list_operator": {
                        "class_name": "ActionListValidationOperator",
                        "action_list": [
                            {
                                "name": "store_validation_result",
                                "action": {
                                    "class_name": "StoreValidationResultAction",
                                },
                            },
                            {
                                "name": "update_data_docs",
                                "action": {
                                    "class_name": "UpdateDataDocsAction",
                                },
                            },
                        ],
                    },
                },
                anonymous_usage_statistics={
                    "enabled": False
                },
            )
            
            # Create data context
            context = BaseDataContext(project_config=data_context_config)
            return context
            
        except Exception as e:
            print(f"Error setting up Great Expectations context: {e}")
            raise
    
    def validate_cholera_cases(self, data: pd.DataFrame, district_code: str) -> Dict[str, Any]:
        """Validate cholera cases data."""
        try:
            # Create batch request
            batch_request = RuntimeBatchRequest(
                datasource_name="cholera_ew_datasource",
                data_connector_name="default_runtime_data_connector",
                data_asset_name="cholera_cases",
                runtime_parameters={"batch_data": data},
                batch_identifiers={"default_identifier_name": f"cholera_cases_{district_code}"},
            )
            
            # Create or get validator
            validator = self.context.get_validator(
                batch_request=batch_request,
                expectation_suite_name="cholera_cases_suite"
            )
            
            # Add expectations
            validator.expect_table_row_count_to_be_between(min_value=1, max_value=10000)
            validator.expect_table_columns_to_match_ordered_list(
                column_list=["id", "case_id", "district_code", "facility_code", "case_date", 
                           "age_group", "gender", "case_status", "lab_confirmed", "outcome"]
            )
            
            # Validate case_id uniqueness
            validator.expect_column_values_to_be_unique(column="case_id")
            
            # Validate district_code
            validator.expect_column_values_to_not_be_null(column="district_code")
            validator.expect_column_values_to_equal(column="district_code", value=district_code)
            
            # Validate case_date
            validator.expect_column_values_to_not_be_null(column="case_date")
            validator.expect_column_values_to_be_of_type(column="case_date", type_="datetime64[ns]")
            
            # Validate lab_confirmed
            validator.expect_column_values_to_be_of_type(column="lab_confirmed", type_="bool")
            
            # Validate age_group
            validator.expect_column_values_to_be_in_set(
                column="age_group", 
                value_set=["0-5", "6-17", "18-59", "60+", None]
            )
            
            # Validate gender
            validator.expect_column_values_to_be_in_set(
                column="gender", 
                value_set=["M", "F", None]
            )
            
            # Validate outcome
            validator.expect_column_values_to_be_in_set(
                column="outcome", 
                value_set=["recovered", "died", "ongoing", None]
            )
            
            # Run validation
            validation_result = validator.validate()
            
            return {
                "district_code": district_code,
                "validation_success": validation_result.success,
                "validation_results": validation_result.to_json_dict(),
                "statistics": {
                    "total_cases": len(data),
                    "lab_confirmed_cases": data["lab_confirmed"].sum() if "lab_confirmed" in data.columns else 0,
                    "date_range": {
                        "start": str(data["case_date"].min()) if "case_date" in data.columns else None,
                        "end": str(data["case_date"].max()) if "case_date" in data.columns else None
                    }
                }
            }
            
        except Exception as e:
            return {
                "district_code": district_code,
                "validation_success": False,
                "error": str(e),
                "statistics": {}
            }
    
    def validate_climate_data(self, data: pd.DataFrame, district_code: str) -> Dict[str, Any]:
        """Validate climate data."""
        try:
            # Create batch request
            batch_request = RuntimeBatchRequest(
                datasource_name="cholera_ew_datasource",
                data_connector_name="default_runtime_data_connector",
                data_asset_name="climate_data",
                runtime_parameters={"batch_data": data},
                batch_identifiers={"default_identifier_name": f"climate_data_{district_code}"},
            )
            
            # Create or get validator
            validator = self.context.get_validator(
                batch_request=batch_request,
                expectation_suite_name="climate_data_suite"
            )
            
            # Add expectations
            validator.expect_table_columns_to_match_ordered_list(
                column_list=["id", "district_code", "date", "rainfall_mm", 
                           "temperature_celsius", "humidity_percent", "data_source"]
            )
            
            # Validate district_code
            validator.expect_column_values_to_not_be_null(column="district_code")
            validator.expect_column_values_to_equal(column="district_code", value=district_code)
            
            # Validate date
            validator.expect_column_values_to_not_be_null(column="date")
            validator.expect_column_values_to_be_of_type(column="date", type_="datetime64[ns]")
            
            # Validate rainfall (reasonable range for Uganda)
            validator.expect_column_values_to_be_between(
                column="rainfall_mm", 
                min_value=0, 
                max_value=200, 
                mostly=0.95  # Allow 5% outliers
            )
            
            # Validate temperature (reasonable range for Uganda)
            validator.expect_column_values_to_be_between(
                column="temperature_celsius", 
                min_value=15, 
                max_value=40, 
                mostly=0.95
            )
            
            # Validate humidity (percentage range)
            validator.expect_column_values_to_be_between(
                column="humidity_percent", 
                min_value=0, 
                max_value=100, 
                mostly=0.95
            )
            
            # Run validation
            validation_result = validator.validate()
            
            return {
                "district_code": district_code,
                "validation_success": validation_result.success,
                "validation_results": validation_result.to_json_dict(),
                "statistics": {
                    "total_records": len(data),
                    "avg_rainfall": float(data["rainfall_mm"].mean()) if "rainfall_mm" in data.columns else None,
                    "avg_temperature": float(data["temperature_celsius"].mean()) if "temperature_celsius" in data.columns else None,
                    "avg_humidity": float(data["humidity_percent"].mean()) if "humidity_percent" in data.columns else None,
                    "date_range": {
                        "start": str(data["date"].min()) if "date" in data.columns else None,
                        "end": str(data["date"].max()) if "date" in data.columns else None
                    }
                }
            }
            
        except Exception as e:
            return {
                "district_code": district_code,
                "validation_success": False,
                "error": str(e),
                "statistics": {}
            }
    
    def validate_risk_predictions(self, data: pd.DataFrame, district_code: str) -> Dict[str, Any]:
        """Validate risk predictions data."""
        try:
            # Create batch request
            batch_request = RuntimeBatchRequest(
                datasource_name="cholera_ew_datasource",
                data_connector_name="default_runtime_data_connector",
                data_asset_name="risk_predictions",
                runtime_parameters={"batch_data": data},
                batch_identifiers={"default_identifier_name": f"risk_predictions_{district_code}"},
            )
            
            # Create or get validator
            validator = self.context.get_validator(
                batch_request=batch_request,
                expectation_suite_name="risk_predictions_suite"
            )
            
            # Add expectations
            validator.expect_table_columns_to_match_ordered_list(
                column_list=["id", "district_code", "prediction_date", "prediction_horizon_weeks",
                           "risk_score", "confidence_interval_lower", "confidence_interval_upper",
                           "model_name", "model_version", "features_used"]
            )
            
            # Validate district_code
            validator.expect_column_values_to_not_be_null(column="district_code")
            validator.expect_column_values_to_equal(column="district_code", value=district_code)
            
            # Validate prediction_date
            validator.expect_column_values_to_not_be_null(column="prediction_date")
            validator.expect_column_values_to_be_of_type(column="prediction_date", type_="datetime64[ns]")
            
            # Validate prediction_horizon_weeks
            validator.expect_column_values_to_not_be_null(column="prediction_horizon_weeks")
            validator.expect_column_values_to_be_between(
                column="prediction_horizon_weeks", 
                min_value=1, 
                max_value=24
            )
            
            # Validate risk_score (0-100 range)
            validator.expect_column_values_to_not_be_null(column="risk_score")
            validator.expect_column_values_to_be_between(
                column="risk_score", 
                min_value=0, 
                max_value=100
            )
            
            # Validate confidence intervals
            validator.expect_column_values_to_be_between(
                column="confidence_interval_lower", 
                min_value=0, 
                max_value=100
            )
            validator.expect_column_values_to_be_between(
                column="confidence_interval_upper", 
                min_value=0, 
                max_value=100
            )
            
            # Validate model_name
            validator.expect_column_values_to_not_be_null(column="model_name")
            validator.expect_column_values_to_be_in_set(
                column="model_name", 
                value_set=["LSTM", "XGBoost", "Prophet", "ARIMA", "Ensemble"]
            )
            
            # Run validation
            validation_result = validator.validate()
            
            return {
                "district_code": district_code,
                "validation_success": validation_result.success,
                "validation_results": validation_result.to_json_dict(),
                "statistics": {
                    "total_predictions": len(data),
                    "avg_risk_score": float(data["risk_score"].mean()) if "risk_score" in data.columns else None,
                    "model_distribution": data["model_name"].value_counts().to_dict() if "model_name" in data.columns else {},
                    "prediction_date_range": {
                        "start": str(data["prediction_date"].min()) if "prediction_date" in data.columns else None,
                        "end": str(data["prediction_date"].max()) if "prediction_date" in data.columns else None
                    }
                }
            }
            
        except Exception as e:
            return {
                "district_code": district_code,
                "validation_success": False,
                "error": str(e),
                "statistics": {}
            }
    
    def validate_district_data(self, district_code: str, db: Session) -> Dict[str, Any]:
        """Validate all data for a specific district."""
        try:
            # Get data for the district
            cholera_cases = pd.read_sql(
                f"SELECT * FROM cholera_ew.cholera_cases WHERE district_code = '{district_code}'",
                db.bind
            )
            
            climate_data = pd.read_sql(
                f"SELECT * FROM cholera_ew.climate_data WHERE district_code = '{district_code}'",
                db.bind
            )
            
            risk_predictions = pd.read_sql(
                f"SELECT * FROM cholera_ew.risk_predictions WHERE district_code = '{district_code}'",
                db.bind
            )
            
            # Validate each dataset
            validation_results = {}
            
            if not cholera_cases.empty:
                validation_results["cholera_cases"] = self.validate_cholera_cases(cholera_cases, district_code)
            
            if not climate_data.empty:
                validation_results["climate_data"] = self.validate_climate_data(climate_data, district_code)
            
            if not risk_predictions.empty:
                validation_results["risk_predictions"] = self.validate_risk_predictions(risk_predictions, district_code)
            
            # Overall validation status
            all_valid = all(
                result["validation_success"] 
                for result in validation_results.values()
            )
            
            return {
                "district_code": district_code,
                "overall_validation_success": all_valid,
                "validation_results": validation_results,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                "district_code": district_code,
                "overall_validation_success": False,
                "error": str(e),
                "validation_results": {},
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def run_data_quality_report(self, db: Session) -> Dict[str, Any]:
        """Run comprehensive data quality report for all districts."""
        try:
            # Get all districts
            districts = db.query(District.district_code).all()
            district_codes = [d[0] for d in districts]
            
            report = {
                "report_timestamp": datetime.utcnow().isoformat(),
                "total_districts": len(district_codes),
                "district_validations": {},
                "summary": {
                    "total_validated": 0,
                    "successful_validations": 0,
                    "failed_validations": 0
                }
            }
            
            # Validate each district
            for district_code in district_codes:
                validation_result = self.validate_district_data(district_code, db)
                report["district_validations"][district_code] = validation_result
                
                report["summary"]["total_validated"] += 1
                if validation_result["overall_validation_success"]:
                    report["summary"]["successful_validations"] += 1
                else:
                    report["summary"]["failed_validations"] += 1
            
            return report
            
        except Exception as e:
            return {
                "report_timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "district_validations": {},
                "summary": {
                    "total_validated": 0,
                    "successful_validations": 0,
                    "failed_validations": 0
                }
            }