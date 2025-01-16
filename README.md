# Cholera Early Warning System

A comprehensive machine learning-based early warning system for predicting district-level cholera risk 8-12 weeks ahead in Uganda. The system integrates multiple data sources including DHIS2 cases, NASA climate data, UNHCR settlement maps, UBOS population data, and remote-sensing water extent data.

## 🌟 Features

- **Multi-Model Predictions**: LSTM, XGBoost, Prophet, and ARIMA models with ensemble predictions
- **Spatial Analytics**: PostGIS integration for geographic data processing
- **Real-time Dashboard**: Interactive web interface with risk maps and alerts
- **Data Quality Assurance**: Great Expectations for comprehensive data validation
- **Workflow Orchestration**: Prefect for automated ML pipeline management
- **Model Tracking**: MLflow for experiment tracking and model versioning
- **Alert System**: Automated risk-based alerting with severity levels
- **API-First Design**: RESTful API for easy integration with existing systems

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │    Backend      │    │   Database      │
│   (Dash/Leaflet)│◄──►│   (FastAPI)     │◄──►│  (PostgreSQL    │
│                 │    │                 │    │   + PostGIS)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐             │
         │              │  ML Pipeline    │             │
         │              │  (Prefect)      │             │
         │              └─────────────────┘             │
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   MLflow        │    │  Great          │    │   Redis         │
│  (Tracking)     │    │ Expectations    │    │  (Caching)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.11+ (for local development)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd cholera-ew-system
   ```

2. **Start the system with Docker Compose**
   ```bash
   docker-compose up -d
   ```

3. **Wait for services to initialize** (approximately 2-3 minutes)

4. **Access the application**
   - Frontend Dashboard: http://localhost:8050
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - MLflow UI: http://localhost:5000
   - Prefect UI: http://localhost:4200

### Verify Installation

Check that all services are running:
```bash
docker-compose ps
```

All services should show "Up" status.

## 📊 Data Structure

### Required Data Files

The system expects the following data structure:

```
data/
├── districts/
│   ├── districts.geojson          # District boundaries
│   └── districts.csv              # District metadata
├── health_facilities/
│   └── facilities.csv             # Health facility locations
├── cholera_cases/
│   └── cases.csv                  # Cholera case reports
├── climate/
│   ├── rainfall.csv               # NASA rainfall data
│   ├── temperature.csv            # NASA temperature data
│   └── humidity.csv               # NASA humidity data
└── water_extent/
    └── water_data.csv             # Remote sensing water data
```

### Sample Data Format

#### Districts CSV
```csv
district_code,district_name,region,population,latitude,longitude
KMP001,Kampala,Central,1500000,0.3476,32.5825
MBL001,Mbarara,Western,500000,-0.6100,30.6500
```

#### Cholera Cases CSV
```csv
case_id,district_code,facility_code,case_date,age_group,gender,case_status,lab_confirmed,outcome
CASE001,KMP001,FAC001,2024-01-15,18-59,M,confirmed,true,recovered
CASE002,MBL001,FAC002,2024-01-16,0-5,F,suspected,false,ongoing
```

#### Climate Data CSV
```csv
district_code,date,rainfall_mm,temperature_celsius,humidity_percent,data_source
KMP001,2024-01-15,25.5,28.0,75.0,NASA
MBL001,2024-01-15,30.2,26.5,80.0,NASA
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
# Database Configuration
DATABASE_URL=postgresql://cholera_user:cholera_password@postgres:5432/cholera_ew
POSTGRES_DB=cholera_ew
POSTGRES_USER=cholera_user
POSTGRES_PASSWORD=cholera_password

# MLflow Configuration
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_BACKEND_STORE_URI=postgresql://cholera_user:cholera_password@postgres:5432/cholera_ew

# Redis Configuration
REDIS_URL=redis://redis:6379

# Prefect Configuration
PREFECT_API_URL=http://prefect:4200/api

# Backend URL (for frontend)
BACKEND_URL=http://backend:8000
```

### Model Configuration

Edit `backend/config/model_config.py` to adjust model parameters:

```python
MODEL_CONFIG = {
    "LSTM": {
        "sequence_length": 14,
        "epochs": 50,
        "batch_size": 32,
        "learning_rate": 0.001
    },
    "XGBoost": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1
    },
    "Prophet": {
        "yearly_seasonality": True,
        "weekly_seasonality": True,
        "daily_seasonality": False
    },
    "ARIMA": {
        "order": (1, 1, 1)
    }
}
```

## 📈 Usage

### API Endpoints

#### Districts
- `GET /districts/` - List all districts
- `POST /districts/` - Create new district
- `GET /districts/{district_code}` - Get specific district

#### Cholera Cases
- `GET /cholera-cases/` - List cholera cases
- `POST /cholera-cases/` - Create new case
- `GET /cholera-cases/?district_code={code}` - Filter by district

#### Risk Predictions
- `GET /risk-predictions/` - List predictions
- `POST /risk-predictions/generate` - Generate new predictions

#### Alerts
- `GET /alerts/` - List active alerts
- `POST /alerts/{alert_id}/acknowledge` - Acknowledge alert

#### Dashboard
- `GET /dashboard/summary` - Get dashboard statistics

### Generating Predictions

1. **Via API**
   ```bash
   curl -X POST "http://localhost:8000/risk-predictions/generate" \
        -H "Content-Type: application/json" \
        -d '{"district_code": "KMP001", "horizon_weeks": 8}'
   ```

2. **Via Prefect Flow**
   ```python
   from prefect_flows import cholera_risk_prediction_pipeline
   
   result = cholera_risk_prediction_pipeline(
       horizon_weeks=8,
       districts=["KMP001", "MBL001"]
   )
   ```

3. **Scheduled Execution**
   ```bash
   # Run daily predictions
   docker-compose exec backend python -m prefect_flows daily_prediction_workflow
   
   # Run weekly model retraining
   docker-compose exec backend python -m prefect_flows weekly_model_retraining
   ```

### Data Upload

1. **Via API**
   ```bash
   # Upload cholera cases
   curl -X POST "http://localhost:8000/cholera-cases/" \
        -H "Content-Type: application/json" \
        -d @data/cholera_cases/cases.csv
   
   # Upload climate data
   curl -X POST "http://localhost:8000/climate-data/" \
        -H "Content-Type: application/json" \
        -d @data/climate/rainfall.csv
   ```

2. **Via Database**
   ```bash
   # Connect to database
   docker-compose exec postgres psql -U cholera_user -d cholera_ew
   
   # Import CSV data
   \copy cholera_ew.cholera_cases FROM '/app/data/cholera_cases/cases.csv' CSV HEADER;
   ```

## 🧪 Testing

### Run Tests

```bash
# Run all tests
docker-compose exec backend python -m pytest tests/

# Run specific test categories
docker-compose exec backend python -m pytest tests/ -m unit
docker-compose exec backend python -m pytest tests/ -m integration
docker-compose exec backend python -m pytest tests/ -m "not slow"

# Run with coverage
docker-compose exec backend python -m pytest tests/ --cov=backend --cov-report=html
```

### Test Categories

- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end system testing
- **Performance Tests**: Load and stress testing

### Manual Testing

1. **Health Check**
   ```bash
   curl http://localhost:8000/health
   ```

2. **Create Test Data**
   ```bash
   # Create district
   curl -X POST "http://localhost:8000/districts/" \
        -H "Content-Type: application/json" \
        -d '{"district_code": "TEST001", "district_name": "Test District", "region": "Test", "population": 100000}'
   
   # Create cholera case
   curl -X POST "http://localhost:8000/cholera-cases/" \
        -H "Content-Type: application/json" \
        -d '{"case_id": "TEST_CASE001", "district_code": "TEST001", "case_date": "2024-01-15", "lab_confirmed": true}'
   ```

## 📊 Monitoring and Logging

### Application Logs

```bash
# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f backend
docker-compose logs -f postgres
docker-compose logs -f mlflow
```

### MLflow Monitoring

Access MLflow UI at http://localhost:5000 to:
- View experiment runs
- Compare model performance
- Track model versions
- Monitor prediction accuracy

### Prefect Monitoring

Access Prefect UI at http://localhost:4200 to:
- Monitor workflow execution
- View flow run history
- Debug failed runs
- Schedule recurring tasks

### Data Quality Monitoring

Great Expectations validation results are available in:
- `backend/great_expectations/validations/`
- `backend/great_expectations/data_docs/`

## 🔧 Development

### Local Development Setup

1. **Clone and setup**
   ```bash
   git clone <repository-url>
   cd cholera-ew-system
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Start only database services**
   ```bash
   docker-compose up -d postgres redis mlflow
   ```

3. **Run backend locally**
   ```bash
   cd backend
   alembic upgrade head
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

4. **Run frontend locally**
   ```bash
   cd frontend
   python app.py
   ```

### Code Structure

```
cholera-ew-system/
├── backend/
│   ├── alembic/              # Database migrations
│   ├── services/             # Business logic
│   │   ├── ml_service.py     # ML model operations
│   │   ├── data_service.py   # Data processing
│   │   ├── alert_service.py  # Alert management
│   │   └── data_validation.py # Data quality
│   ├── models.py             # Database models
│   ├── schemas.py            # API schemas
│   ├── main.py               # FastAPI application
│   └── prefect_flows.py      # Workflow definitions
├── frontend/
│   ├── app.py                # Dash application
│   ├── assets/               # Static assets
│   └── requirements.txt      # Frontend dependencies
├── tests/                    # Test suites
├── config/                   # Configuration files
├── data/                     # Sample data
├── docker-compose.yml        # Container orchestration
└── requirements.txt          # Backend dependencies
```

### Adding New Models

1. **Implement model in MLService**
   ```python
   def _predict_new_model(self, data: pd.DataFrame, horizon_weeks: int) -> float:
       # Implementation here
       pass
   ```

2. **Add to ensemble prediction**
   ```python
   def generate_predictions(self, district_code: str, horizon_weeks: int):
       # Add new model to predictions
       new_model_pred = self._predict_new_model(data, horizon_weeks)
   ```

3. **Update tests**
   ```python
   def test_new_model_prediction(self):
       # Test implementation
       pass
   ```

### Database Migrations

```bash
# Create new migration
docker-compose exec backend alembic revision --autogenerate -m "Description"

# Apply migrations
docker-compose exec backend alembic upgrade head

# Rollback migration
docker-compose exec backend alembic downgrade -1
```

## 🚀 Deployment

### Production Deployment

1. **Environment Setup**
   ```bash
   # Set production environment variables
   export ENV=production
   export DATABASE_URL=postgresql://user:pass@prod-db:5432/cholera_ew
   export REDIS_URL=redis://prod-redis:6379
   ```

2. **Security Configuration**
   ```bash
   # Update docker-compose.prod.yml
   # - Use secrets management
   # - Enable SSL/TLS
   # - Configure firewalls
   # - Set up monitoring
   ```

3. **Deploy**
   ```bash
   docker-compose -f docker-compose.prod.yml up -d
   ```

### Scaling

- **Horizontal Scaling**: Add more backend instances behind load balancer
- **Database Scaling**: Use read replicas for analytics queries
- **Cache Scaling**: Use Redis Cluster for distributed caching
- **ML Scaling**: Use distributed training with Ray or Dask

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Write comprehensive tests
- Update documentation
- Use type hints
- Add docstrings to functions and classes

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Uganda Ministry of Health for data access and domain expertise
- NASA for climate data
- UNHCR for settlement data
- UBOS for population data
- Open source community for the excellent tools and libraries

## 📞 Support

For support and questions:
- Create an issue in the repository
- Contact the development team
- Check the documentation wiki

## 🔄 Version History

- **v1.0.0** - Initial release with core ML models and dashboard
- **v1.1.0** - Added Prefect orchestration and Great Expectations validation
- **v1.2.0** - Enhanced frontend with interactive maps and improved UX

---

**Note**: This system is designed for research and operational use in public health settings. Ensure proper data governance and privacy protection when handling health data.