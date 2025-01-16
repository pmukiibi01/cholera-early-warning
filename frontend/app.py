"""
Dash frontend application for Cholera Early Warning System.
"""

import dash
from dash import dcc, html, Input, Output, callback, dash_table
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import requests
import os
from typing import Dict, Any, List, Optional

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
app.title = "Cholera Early Warning System"

# Backend URL
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# App layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("Cholera Early Warning System", className="header-title"),
        html.P("District-level cholera risk prediction and alerting system", className="header-subtitle"),
        html.Div(id="last-updated", className="last-updated")
    ], className="header"),
    
    # Navigation tabs
    dcc.Tabs(id="main-tabs", value="dashboard", children=[
        dcc.Tab(label="Dashboard", value="dashboard"),
        dcc.Tab(label="Risk Map", value="risk-map"),
        dcc.Tab(label="Predictions", value="predictions"),
        dcc.Tab(label="Alerts", value="alerts"),
        dcc.Tab(label="Data Quality", value="data-quality"),
    ]),
    
    # Content area
    html.Div(id="tab-content", className="content"),
    
    # Hidden div for storing data
    html.Div(id="data-store", style={"display": "none"}),
    
    # Auto-refresh interval
    dcc.Interval(
        id="interval-component",
        interval=5*60*1000,  # Update every 5 minutes
        n_intervals=0
    )
])

def get_dashboard_content():
    """Get dashboard tab content."""
    return html.Div([
        # Summary cards
        html.Div([
            html.Div([
                html.H3(id="total-districts", children="0"),
                html.P("Total Districts")
            ], className="summary-card"),
            html.Div([
                html.H3(id="total-cases", children="0"),
                html.P("Total Cases")
            ], className="summary-card"),
            html.Div([
                html.H3(id="active-alerts", children="0"),
                html.P("Active Alerts")
            ], className="summary-card"),
            html.Div([
                html.H3(id="avg-risk-score", children="0"),
                html.P("Avg Risk Score")
            ], className="summary-card"),
        ], className="summary-cards"),
        
        # Charts row
        html.Div([
            # Cases over time
            html.Div([
                html.H4("Cases Over Time (Last 30 Days)"),
                dcc.Graph(id="cases-timeline")
            ], className="chart-container"),
            
            # Risk distribution
            html.Div([
                html.H4("Risk Score Distribution"),
                dcc.Graph(id="risk-distribution")
            ], className="chart-container"),
        ], className="charts-row"),
        
        # Recent predictions table
        html.Div([
            html.H4("Recent Risk Predictions"),
            html.Div(id="predictions-table")
        ], className="table-container")
    ])

def get_risk_map_content():
    """Get risk map tab content."""
    return html.Div([
        html.Div([
            html.H4("Cholera Risk Map"),
            html.P("Interactive map showing current cholera risk predictions by district"),
            
            # Map controls
            html.Div([
                html.Label("Prediction Horizon:"),
                dcc.Dropdown(
                    id="horizon-selector",
                    options=[
                        {"label": "4 weeks", "value": 4},
                        {"label": "8 weeks", "value": 8},
                        {"label": "12 weeks", "value": 12}
                    ],
                    value=8,
                    style={"width": "200px"}
                ),
                html.Label("Model:"),
                dcc.Dropdown(
                    id="model-selector",
                    options=[
                        {"label": "Ensemble", "value": "Ensemble"},
                        {"label": "LSTM", "value": "LSTM"},
                        {"label": "XGBoost", "value": "XGBoost"},
                        {"label": "Prophet", "value": "Prophet"},
                        {"label": "ARIMA", "value": "ARIMA"}
                    ],
                    value="Ensemble",
                    style={"width": "200px"}
                )
            ], className="map-controls"),
            
            # Map container
            html.Div(id="risk-map", className="map-container")
        ])
    ])

def get_predictions_content():
    """Get predictions tab content."""
    return html.Div([
        html.Div([
            html.H4("Risk Predictions"),
            html.P("Detailed view of cholera risk predictions by district and model"),
            
            # Filters
            html.Div([
                html.Label("District:"),
                dcc.Dropdown(
                    id="district-filter",
                    placeholder="Select district...",
                    style={"width": "300px"}
                ),
                html.Label("Model:"),
                dcc.Dropdown(
                    id="model-filter",
                    options=[
                        {"label": "All Models", "value": "all"},
                        {"label": "Ensemble", "value": "Ensemble"},
                        {"label": "LSTM", "value": "LSTM"},
                        {"label": "XGBoost", "value": "XGBoost"},
                        {"label": "Prophet", "value": "Prophet"},
                        {"label": "ARIMA", "value": "ARIMA"}
                    ],
                    value="all",
                    style={"width": "200px"}
                ),
                html.Label("Horizon:"),
                dcc.Dropdown(
                    id="horizon-filter",
                    options=[
                        {"label": "All Horizons", "value": "all"},
                        {"label": "4 weeks", "value": 4},
                        {"label": "8 weeks", "value": 8},
                        {"label": "12 weeks", "value": 12}
                    ],
                    value="all",
                    style={"width": "200px"}
                )
            ], className="filters"),
            
            # Predictions table
            html.Div(id="predictions-detailed-table")
        ])
    ])

def get_alerts_content():
    """Get alerts tab content."""
    return html.Div([
        html.Div([
            html.H4("Active Alerts"),
            html.P("Current cholera risk alerts by severity and district"),
            
            # Alert filters
            html.Div([
                html.Label("Severity:"),
                dcc.Dropdown(
                    id="severity-filter",
                    options=[
                        {"label": "All Severities", "value": "all"},
                        {"label": "Critical", "value": "critical"},
                        {"label": "High", "value": "high"},
                        {"label": "Medium", "value": "medium"},
                        {"label": "Low", "value": "low"}
                    ],
                    value="all",
                    style={"width": "200px"}
                ),
                html.Label("District:"),
                dcc.Dropdown(
                    id="alert-district-filter",
                    placeholder="Select district...",
                    style={"width": "300px"}
                )
            ], className="filters"),
            
            # Alerts table
            html.Div(id="alerts-table")
        ])
    ])

def get_data_quality_content():
    """Get data quality tab content."""
    return html.Div([
        html.Div([
            html.H4("Data Quality Report"),
            html.P("Comprehensive data validation results across all districts"),
            
            # Quality summary
            html.Div([
                html.Div([
                    html.H3(id="total-validated", children="0"),
                    html.P("Districts Validated")
                ], className="quality-card"),
                html.Div([
                    html.H3(id="successful-validations", children="0"),
                    html.P("Successful Validations")
                ], className="quality-card"),
                html.Div([
                    html.H3(id="failed-validations", children="0"),
                    html.P("Failed Validations")
                ], className="quality-card"),
            ], className="quality-summary"),
            
            # Quality details
            html.Div(id="quality-details")
        ])
    ])

@app.callback(
    Output("tab-content", "children"),
    [Input("main-tabs", "value")]
)
def render_tab_content(active_tab):
    """Render content based on active tab."""
    if active_tab == "dashboard":
        return get_dashboard_content()
    elif active_tab == "risk-map":
        return get_risk_map_content()
    elif active_tab == "predictions":
        return get_predictions_content()
    elif active_tab == "alerts":
        return get_alerts_content()
    elif active_tab == "data-quality":
        return get_data_quality_content()
    else:
        return html.Div("Select a tab")

@app.callback(
    [Output("total-districts", "children"),
     Output("total-cases", "children"),
     Output("active-alerts", "children"),
     Output("avg-risk-score", "children"),
     Output("last-updated", "children")],
    [Input("interval-component", "n_intervals")]
)
def update_summary_cards(n):
    """Update summary cards with latest data."""
    try:
        # Get dashboard summary from backend
        response = requests.get(f"{BACKEND_URL}/dashboard/summary")
        if response.status_code == 200:
            data = response.json()
            
            total_districts = data.get("total_districts", 0)
            total_cases = data.get("total_cases", 0)
            active_alerts = data.get("active_alerts", 0)
            
            # Calculate average risk score from recent predictions
            recent_predictions = data.get("recent_predictions", [])
            avg_risk_score = 0
            if recent_predictions:
                risk_scores = [p.get("risk_score", 0) for p in recent_predictions if p.get("risk_score")]
                avg_risk_score = sum(risk_scores) / len(risk_scores) if risk_scores else 0
            
            last_updated = f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            return (
                f"{total_districts:,}",
                f"{total_cases:,}",
                f"{active_alerts:,}",
                f"{avg_risk_score:.1f}",
                last_updated
            )
        else:
            return "Error", "Error", "Error", "Error", "Error fetching data"
    except Exception as e:
        return "Error", "Error", "Error", "Error", f"Error: {str(e)}"

@app.callback(
    Output("cases-timeline", "figure"),
    [Input("interval-component", "n_intervals")]
)
def update_cases_timeline(n):
    """Update cases timeline chart."""
    try:
        # Get cholera cases data from backend
        response = requests.get(f"{BACKEND_URL}/cholera-cases/")
        if response.status_code == 200:
            cases_data = response.json()
            
            # Convert to DataFrame and process
            df = pd.DataFrame(cases_data)
            if not df.empty:
                df['case_date'] = pd.to_datetime(df['case_date'])
                
                # Group by date and count cases
                daily_cases = df.groupby(df['case_date'].dt.date).size().reset_index()
                daily_cases.columns = ['date', 'cases']
                
                # Filter last 30 days
                cutoff_date = (datetime.now() - timedelta(days=30)).date()
                daily_cases = daily_cases[daily_cases['date'] >= cutoff_date]
                
                fig = px.line(
                    daily_cases, 
                    x='date', 
                    y='cases',
                    title="Daily Cholera Cases (Last 30 Days)",
                    labels={'cases': 'Number of Cases', 'date': 'Date'}
                )
                fig.update_layout(height=400)
                return fig
            else:
                return go.Figure().add_annotation(
                    text="No data available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
        else:
            return go.Figure().add_annotation(
                text="Error loading data",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
    except Exception as e:
        return go.Figure().add_annotation(
            text=f"Error: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )

@app.callback(
    Output("risk-distribution", "figure"),
    [Input("interval-component", "n_intervals")]
)
def update_risk_distribution(n):
    """Update risk distribution chart."""
    try:
        # Get risk predictions data from backend
        response = requests.get(f"{BACKEND_URL}/risk-predictions/")
        if response.status_code == 200:
            predictions_data = response.json()
            
            if predictions_data:
                df = pd.DataFrame(predictions_data)
                
                # Filter for ensemble predictions only
                ensemble_df = df[df['model_name'] == 'Ensemble']
                
                if not ensemble_df.empty:
                    fig = px.histogram(
                        ensemble_df,
                        x='risk_score',
                        nbins=20,
                        title="Risk Score Distribution (Ensemble Model)",
                        labels={'risk_score': 'Risk Score', 'count': 'Number of Districts'}
                    )
                    fig.update_layout(height=400)
                    return fig
                else:
                    return go.Figure().add_annotation(
                        text="No ensemble predictions available",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5, showarrow=False
                    )
            else:
                return go.Figure().add_annotation(
                    text="No predictions available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
        else:
            return go.Figure().add_annotation(
                text="Error loading data",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
    except Exception as e:
        return go.Figure().add_annotation(
            text=f"Error: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )

@app.callback(
    Output("predictions-table", "children"),
    [Input("interval-component", "n_intervals")]
)
def update_predictions_table(n):
    """Update recent predictions table."""
    try:
        # Get recent predictions from backend
        response = requests.get(f"{BACKEND_URL}/risk-predictions/")
        if response.status_code == 200:
            predictions_data = response.json()
            
            if predictions_data:
                df = pd.DataFrame(predictions_data)
                
                # Filter for recent predictions (last 7 days)
                df['prediction_date'] = pd.to_datetime(df['prediction_date'])
                recent_date = datetime.now() - timedelta(days=7)
                recent_predictions = df[df['prediction_date'] >= recent_date]
                
                # Sort by prediction date and risk score
                recent_predictions = recent_predictions.sort_values(
                    ['prediction_date', 'risk_score'], 
                    ascending=[False, False]
                ).head(20)  # Show top 20
                
                # Format the table
                table_data = recent_predictions[[
                    'district_code', 'prediction_date', 'prediction_horizon_weeks',
                    'risk_score', 'model_name', 'confidence_interval_lower', 'confidence_interval_upper'
                ]].to_dict('records')
                
                return dash_table.DataTable(
                    data=table_data,
                    columns=[
                        {"name": "District", "id": "district_code"},
                        {"name": "Date", "id": "prediction_date", "type": "datetime"},
                        {"name": "Horizon (weeks)", "id": "prediction_horizon_weeks"},
                        {"name": "Risk Score", "id": "risk_score", "type": "numeric", "format": {"specifier": ".1f"}},
                        {"name": "Model", "id": "model_name"},
                        {"name": "CI Lower", "id": "confidence_interval_lower", "type": "numeric", "format": {"specifier": ".1f"}},
                        {"name": "CI Upper", "id": "confidence_interval_upper", "type": "numeric", "format": {"specifier": ".1f"}}
                    ],
                    style_cell={'textAlign': 'left', 'fontSize': '12px'},
                    style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                    style_data_conditional=[
                        {
                            'if': {'filter_query': '{risk_score} > 75'},
                            'backgroundColor': '#ffebee',
                            'color': 'black',
                        },
                        {
                            'if': {'filter_query': '{risk_score} > 50 && {risk_score} <= 75'},
                            'backgroundColor': '#fff3e0',
                            'color': 'black',
                        },
                        {
                            'if': {'filter_query': '{risk_score} <= 50'},
                            'backgroundColor': '#e8f5e8',
                            'color': 'black',
                        }
                    ]
                )
            else:
                return html.P("No predictions available")
        else:
            return html.P("Error loading predictions data")
    except Exception as e:
        return html.P(f"Error: {str(e)}")

if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8050)