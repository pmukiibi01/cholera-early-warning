-- Initialize database for Cholera Early Warning System

-- Create extensions
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;
CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;
CREATE EXTENSION IF NOT EXISTS postgis_tiger_geocoder;

-- Create schemas
CREATE SCHEMA IF NOT EXISTS cholera_ew;
CREATE SCHEMA IF NOT EXISTS mlflow;

-- Set search path
SET search_path TO cholera_ew, public;

-- Create districts table (Uganda administrative boundaries)
CREATE TABLE IF NOT EXISTS districts (
    id SERIAL PRIMARY KEY,
    district_code VARCHAR(20) UNIQUE NOT NULL,
    district_name VARCHAR(100) NOT NULL,
    region VARCHAR(50),
    population INTEGER,
    geometry GEOMETRY(MULTIPOLYGON, 4326),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create health facilities table
CREATE TABLE IF NOT EXISTS health_facilities (
    id SERIAL PRIMARY KEY,
    facility_code VARCHAR(50) UNIQUE NOT NULL,
    facility_name VARCHAR(200) NOT NULL,
    district_code VARCHAR(20) REFERENCES districts(district_code),
    facility_type VARCHAR(50),
    latitude DECIMAL(10, 8),
    longitude DECIMAL(11, 8),
    geometry GEOMETRY(POINT, 4326),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create cholera cases table
CREATE TABLE IF NOT EXISTS cholera_cases (
    id SERIAL PRIMARY KEY,
    case_id VARCHAR(50) UNIQUE NOT NULL,
    district_code VARCHAR(20) REFERENCES districts(district_code),
    facility_code VARCHAR(50) REFERENCES health_facilities(facility_code),
    case_date DATE NOT NULL,
    age_group VARCHAR(20),
    gender VARCHAR(10),
    case_status VARCHAR(20),
    lab_confirmed BOOLEAN DEFAULT FALSE,
    outcome VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create climate data table
CREATE TABLE IF NOT EXISTS climate_data (
    id SERIAL PRIMARY KEY,
    district_code VARCHAR(20) REFERENCES districts(district_code),
    date DATE NOT NULL,
    rainfall_mm DECIMAL(8, 2),
    temperature_celsius DECIMAL(5, 2),
    humidity_percent DECIMAL(5, 2),
    data_source VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(district_code, date)
);

-- Create water extent data table
CREATE TABLE IF NOT EXISTS water_extent (
    id SERIAL PRIMARY KEY,
    district_code VARCHAR(20) REFERENCES districts(district_code),
    date DATE NOT NULL,
    water_area_sqkm DECIMAL(12, 4),
    flood_risk_score DECIMAL(5, 2),
    data_source VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(district_code, date)
);

-- Create risk predictions table
CREATE TABLE IF NOT EXISTS risk_predictions (
    id SERIAL PRIMARY KEY,
    district_code VARCHAR(20) REFERENCES districts(district_code),
    prediction_date DATE NOT NULL,
    prediction_horizon_weeks INTEGER NOT NULL,
    risk_score DECIMAL(5, 2),
    confidence_interval_lower DECIMAL(5, 2),
    confidence_interval_upper DECIMAL(5, 2),
    model_name VARCHAR(50),
    model_version VARCHAR(20),
    features_used JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(district_code, prediction_date, prediction_horizon_weeks, model_name)
);

-- Create alerts table
CREATE TABLE IF NOT EXISTS alerts (
    id SERIAL PRIMARY KEY,
    alert_id VARCHAR(50) UNIQUE NOT NULL,
    district_code VARCHAR(20) REFERENCES districts(district_code),
    alert_type VARCHAR(50),
    severity VARCHAR(20),
    risk_score DECIMAL(5, 2),
    prediction_horizon_weeks INTEGER,
    triggered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    acknowledged_at TIMESTAMP,
    acknowledged_by VARCHAR(100),
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create spatial indexes
CREATE INDEX IF NOT EXISTS idx_districts_geometry ON districts USING GIST (geometry);
CREATE INDEX IF NOT EXISTS idx_health_facilities_geometry ON health_facilities USING GIST (geometry);
CREATE INDEX IF NOT EXISTS idx_districts_district_code ON districts (district_code);
CREATE INDEX IF NOT EXISTS idx_cholera_cases_district_date ON cholera_cases (district_code, case_date);
CREATE INDEX IF NOT EXISTS idx_climate_data_district_date ON climate_data (district_code, date);
CREATE INDEX IF NOT EXISTS idx_risk_predictions_district_date ON risk_predictions (district_code, prediction_date);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_districts_updated_at BEFORE UPDATE ON districts FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_health_facilities_updated_at BEFORE UPDATE ON health_facilities FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_cholera_cases_updated_at BEFORE UPDATE ON cholera_cases FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_climate_data_updated_at BEFORE UPDATE ON climate_data FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_water_extent_updated_at BEFORE UPDATE ON water_extent FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_risk_predictions_updated_at BEFORE UPDATE ON risk_predictions FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_alerts_updated_at BEFORE UPDATE ON alerts FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();