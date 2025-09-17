#!/usr/bin/env python3
"""
Asteroid Tracking Visualization Dashboard
Interactive web dashboard for visualizing asteroid trajectories and collision predictions
"""

import json
import logging
import math
import os
import threading
import time
from datetime import datetime, timedelta

import dash
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, callback_context, dcc, html
from kafka import KafkaConsumer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AsteroidVisualizationDashboard:
    def __init__(self, data_path="/tmp/asteroid_data", kafka_servers=['kafka:9092']):
        """Initialize the visualization dashboard"""
        self.data_path = data_path
        self.kafka_servers = kafka_servers
        
        # Data storage
        self.asteroid_data = []
        self.planet_data = []
        self.alert_data = []
        
        # Physical constants
        self.AU = 149597870.7  # 1 AU in km
        
        # Initialize Dash app
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()
        
        # Start data updating thread
        self.data_thread = threading.Thread(target=self.update_data_continuously, daemon=True)
        self.data_thread.start()
        
        logger.info("Visualization dashboard initialized")
    
    def setup_layout(self):
        """Setup the dashboard layout"""
        self.app.layout = html.Div([
            html.H1("Asteroid Collision Prediction System", 
                   style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 30}),
            
            # Control panel
            html.Div([
                html.Div([
                    html.H3("Control Panel", style={'color': '#34495e'}),
                    html.Label("View Type:"),
                    dcc.Dropdown(
                        id='view-type',
                        options=[
                            {'label': '3D Solar System', 'value': '3d'},
                            {'label': '2D Orbital View', 'value': '2d'},
                            {'label': 'Earth-Centric View', 'value': 'earth'}
                        ],
                        value='3d'
                    ),
                    html.Br(),
                    html.Label("Time Range (days):"),
                    dcc.Slider(
                        id='time-range',
                        min=1, max=365, value=30,
                        marks={1: '1', 30: '30', 90: '90', 180: '180', 365: '365'},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    html.Br(),
                    html.Label("Risk Level Filter:"),
                    dcc.Checklist(
                        id='risk-filter',
                        options=[
                            {'label': 'High Risk', 'value': 'HIGH'},
                            {'label': 'Medium Risk', 'value': 'MEDIUM'},
                            {'label': 'Low Risk', 'value': 'LOW'}
                        ],
                        value=['HIGH', 'MEDIUM', 'LOW']
                    ),
                    html.Br(),
                    html.Button('Refresh Data', id='refresh-button', 
                              style={'backgroundColor': '#3498db', 'color': 'white'})
                ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top', 
                         'padding': '20px', 'backgroundColor': '#ecf0f1', 'borderRadius': '10px'})
            ], style={'marginBottom': 20}),
            
            # Main visualization area
            html.Div([
                # 3D Plot
                html.Div([
                    dcc.Graph(id='asteroid-3d-plot', style={'height': '600px'})
                ], style={'width': '75%', 'display': 'inline-block'}),
                
                # Statistics panel
                html.Div([
                    html.H4("System Statistics", style={'color': '#2c3e50'}),
                    html.Div(id='statistics-panel')
                ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top',
                         'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px'})
            ]),
            
            # Secondary plots
            html.Div([
                html.Div([
                    dcc.Graph(id='collision-probability-plot')
                ], style={'width': '50%', 'display': 'inline-block'}),
                
                html.Div([
                    dcc.Graph(id='asteroid-size-distribution')
                ], style={'width': '50%', 'display': 'inline-block'})
            ], style={'marginTop': 20}),
            
            # Alerts panel
            html.Div([
                html.H3("Collision Alerts", style={'color': '#e74c3c'}),
                html.Div(id='alerts-panel')
            ], style={'marginTop': 20, 'padding': '20px', 'backgroundColor': '#fdf2f2', 
                     'borderRadius': '10px', 'border': '1px solid #e74c3c'}),
            
            # Data table
            html.Div([
                html.H3("Asteroid Data Table"),
                html.Div(id='data-table')
            ], style={'marginTop': 20}),
            
            # Auto-refresh component
            dcc.Interval(
                id='interval-component',
                interval=30*1000,  # Update every 30 seconds
                n_intervals=0
            )
        ])
    
    def setup_callbacks(self):
        """Setup Dash callbacks"""
        
        @self.app.callback(
            [Output('asteroid-3d-plot', 'figure'),
             Output('statistics-panel', 'children'),
             Output('collision-probability-plot', 'figure'),
             Output('asteroid-size-distribution', 'figure'),
             Output('alerts-panel', 'children'),
             Output('data-table', 'children')],
            [Input('interval-component', 'n_intervals'),
             Input('refresh-button', 'n_clicks'),
             Input('view-type', 'value'),
             Input('time-range', 'value'),
             Input('risk-filter', 'value')]
        )
        def update_dashboard(n_intervals, n_clicks, view_type, time_range, risk_filter):
            """Update all dashboard components"""
            
            # Load fresh data
            self.load_latest_data()
            
            # Filter data based on time range and risk level
            filtered_asteroids = self.filter_asteroid_data(time_range, risk_filter)
            
            # Create main plot
            main_figure = self.create_main_plot(filtered_asteroids, view_type)
            
            # Create statistics
            stats = self.create_statistics_panel(filtered_asteroids)
            
            # Create secondary plots
            prob_figure = self.create_collision_probability_plot(filtered_asteroids)
            size_figure = self.create_size_distribution_plot(filtered_asteroids)
            
            # Create alerts
            alerts = self.create_alerts_panel()
            
            # Create data table
            data_table = self.create_data_table(filtered_asteroids)
            
            return main_figure, stats, prob_figure, size_figure, alerts, data_table
    
    def update_data_continuously(self):
        """Continuously update data from Kafka"""
        try:
            # Initialize Kafka consumers
            asteroid_consumer = KafkaConsumer(
                'asteroid-data',
                bootstrap_servers=self.kafka_servers,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='latest',
                group_id='dashboard-consumer'
            )
            
            planet_consumer = KafkaConsumer(
                'planet-data',
                bootstrap_servers=self.kafka_servers,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='latest',
                group_id='dashboard-consumer'
            )
            
            alert_consumer = KafkaConsumer(
                'collision-alerts',
                bootstrap_servers=self.kafka_servers,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='latest',
                group_id='dashboard-consumer'
            )
            
            while True:
                # Poll for new messages
                asteroid_messages = asteroid_consumer.poll(timeout_ms=1000)
                planet_messages = planet_consumer.poll(timeout_ms=1000)
                alert_messages = alert_consumer.poll(timeout_ms=1000)
                
                # Process asteroid data
                for message_batch in asteroid_messages.values():
                    for message in message_batch:
                        self.asteroid_data.append(message.value)
                        # Keep only recent data (last 1000 records)
                        if len(self.asteroid_data) > 1000:
                            self.asteroid_data.pop(0)
                
                # Process planet data
                for message_batch in planet_messages.values():
                    for message in message_batch:
                        # Update planet position
                        planet_name = message.value['planet']
                        # Remove old data for this planet
                        self.planet_data = [p for p in self.planet_data if p['planet'] != planet_name]
                        self.planet_data.append(message.value)
                
                # Process alert data
                for message_batch in alert_messages.values():
                    for message in message_batch:
                        self.alert_data.append(message.value)
                        # Keep only recent alerts (last 100)
                        if len(self.alert_data) > 100:
                            self.alert_data.pop(0)
                
                time.sleep(5)  # Sleep for 5 seconds
                
        except Exception as e:
            logger.error(f"Error in data update thread: {e}")
            # Fallback to loading static data
            self.load_static_data()
    
    def load_latest_data(self):
        """Load latest data from files if Kafka is not available"""
        try:
            # Try to load from parquet files
            asteroid_files = [f for f in os.listdir(f"{self.data_path}/asteroid_data/raw") 
                            if f.endswith('.parquet')]
            
            if asteroid_files:
                latest_file = sorted(asteroid_files)[-1]
                df = pd.read_parquet(f"{self.data_path}/asteroid_data/raw/{latest_file}")
                self.asteroid_data = df.to_dict('records')
            
        except Exception as e:
            logger.warning(f"Could not load latest data: {e}")
            # Generate sample data if no data available
            if not self.asteroid_data:
                self.generate_sample_data()
    
    def load_static_data(self):
        """Load static data for demonstration"""
        self.generate_sample_data()
    
    def generate_sample_data(self):
        """Generate enhanced sample data for demonstration"""
        logger.info("Generating enhanced sample data for visualization...")
        
        np.random.seed(42)
        n_asteroids = 100
        
        # Generate sample asteroids with enhanced variety
        for i in range(n_asteroids):
            # Create different types of asteroids with varied positions
            asteroid_type = np.random.choice(
                ["near_earth", "main_belt", "trojan", "outer"],
                p=[0.25, 0.60, 0.10, 0.05]  # 25% near-Earth for more exciting data
            )
            
            if asteroid_type == "near_earth":
                distance = np.random.uniform(0.8 * self.AU, 1.3 * self.AU)
                inclination = np.random.normal(0, 0.15)
            elif asteroid_type == "main_belt":
                distance = np.random.uniform(2.0 * self.AU, 3.5 * self.AU)
                inclination = np.random.normal(0, 0.1)
            elif asteroid_type == "trojan":
                distance = np.random.uniform(4.8 * self.AU, 5.5 * self.AU)
                inclination = np.random.normal(0, 0.05)
            else:  # outer
                distance = np.random.uniform(5.0 * self.AU, 8.0 * self.AU)
                inclination = np.random.normal(0, 0.2)
            
            angle = np.random.uniform(0, 2 * np.pi)
            
            position = {
                "x": distance * np.cos(angle) * np.cos(inclination),
                "y": distance * np.sin(angle) * np.cos(inclination),
                "z": distance * np.sin(inclination)
            }
            
            # Enhanced velocity calculation
            orbital_speed = np.sqrt(1.327e11 / distance)
            velocity_perturbation = np.random.uniform(0.7, 1.4)
            
            velocity = {
                "vx": -orbital_speed * np.sin(angle) * velocity_perturbation + np.random.normal(0, 10),
                "vy": orbital_speed * np.cos(angle) * velocity_perturbation + np.random.normal(0, 10),
                "vz": np.random.normal(0, 5)
            }
            
            # More varied size distribution
            size_category = np.random.choice(
                ["small", "medium", "large", "very_large"],
                p=[0.50, 0.30, 0.15, 0.05]
            )
            
            if size_category == "small":
                size = np.random.uniform(0.1, 1.0)
            elif size_category == "medium":
                size = np.random.uniform(1.0, 5.0)
            elif size_category == "large":
                size = np.random.uniform(5.0, 20.0)
            else:  # very_large
                size = np.random.uniform(20.0, 100.0)
            
            earth_distance = np.sqrt((position["x"] - self.AU)**2 + position["y"]**2 + position["z"]**2)
            
            # Enhanced collision probability calculation
            distance_factor = max(0, 1 - (earth_distance / (0.05 * self.AU)))
            size_factor = min(size / 5.0, 2.0)
            velocity_magnitude = np.sqrt(velocity["vx"]**2 + velocity["vy"]**2 + velocity["vz"]**2)
            velocity_factor = min(velocity_magnitude / 50000, 1.5)
            
            base_probability = distance_factor * size_factor * velocity_factor
            random_factor = np.random.uniform(0.1, 2.0)
            collision_prob = base_probability * random_factor * 0.01
            
            # Ensure some asteroids have higher probabilities for variety
            if np.random.random() < 0.15:  # 15% high risk
                collision_prob = max(collision_prob, np.random.uniform(0.001, 0.01))
            elif np.random.random() < 0.25:  # 25% medium risk
                collision_prob = max(collision_prob, np.random.uniform(0.0001, 0.001))
            elif np.random.random() < 0.35:  # 35% low risk
                collision_prob = max(collision_prob, np.random.uniform(0.00001, 0.0001))
            
            # Determine risk level and color
            if collision_prob > 0.001:
                risk_level, color = "HIGH", "red"
            elif collision_prob > 0.0001:
                risk_level, color = "MEDIUM", "orange"
            elif collision_prob > 0.00001:
                risk_level, color = "LOW", "yellow"
            elif collision_prob > 0.000001:
                risk_level, color = "VERY_LOW", "blue"
            else:
                risk_level, color = "SAFE", "green"
            
            asteroid = {
                "id": f"asteroid_{i:06d}",
                "timestamp": datetime.now().isoformat(),
                "position": position,
                "velocity": velocity,
                "size": size,
                "mass": size ** 3 * np.random.uniform(2000, 5000),
                "enhanced_collision_probability": collision_prob,
                "collision_probability": collision_prob,
                "distance_from_earth": earth_distance,
                "risk_level": risk_level,
                "color": color,
                "asteroid_type": asteroid_type,
                "size_category": size_category
            }
            
            self.asteroid_data.append(asteroid)
        
        # Generate sample planets
        planets = {
            "Mercury": {"distance": 0.39 * self.AU},
            "Venus": {"distance": 0.72 * self.AU},
            "Earth": {"distance": 1.0 * self.AU},
            "Mars": {"distance": 1.52 * self.AU},
            "Jupiter": {"distance": 5.20 * self.AU}
        }
        
        for name, info in planets.items():
            angle = np.random.uniform(0, 2 * np.pi)
            self.planet_data.append({
                "planet": name,
                "timestamp": datetime.now().isoformat(),
                "position": {
                    "x": info["distance"] * np.cos(angle),
                    "y": info["distance"] * np.sin(angle),
                    "z": 0
                }
            })
    
    def filter_asteroid_data(self, time_range, risk_filter):
        """Filter asteroid data based on criteria"""
        filtered = []
        cutoff_time = datetime.now() - timedelta(days=time_range)
        
        for asteroid in self.asteroid_data:
            # Time filter
            asteroid_time = datetime.fromisoformat(asteroid['timestamp'].replace('Z', '+00:00').replace('+00:00', ''))
            if asteroid_time < cutoff_time:
                continue
            
            # Risk filter
            risk_level = asteroid.get('risk_level', 'LOW')
            if risk_level not in risk_filter:
                continue
            
            filtered.append(asteroid)
        
        return filtered
    
    def create_main_plot(self, asteroids, view_type):
        """Create the main 3D/2D plot"""
        fig = go.Figure()
        
        if view_type == '3d':
            # 3D Solar System View
            fig = go.Figure()
            
            # Add Sun
            fig.add_trace(go.Scatter3d(
                x=[0], y=[0], z=[0],
                mode='markers',
                marker=dict(size=20, color='yellow'),
                name='Sun',
                hovertemplate='Sun<extra></extra>'
            ))
            
            # Add planets
            for planet in self.planet_data:
                pos = planet['position']
                fig.add_trace(go.Scatter3d(
                    x=[pos['x']], y=[pos['y']], z=[pos['z']],
                    mode='markers',
                    marker=dict(size=8, color='blue'),
                    name=planet['planet'],
                    hovertemplate=f"{planet['planet']}<br>" +
                                f"X: {pos['x']:.0f} km<br>" +
                                f"Y: {pos['y']:.0f} km<br>" +
                                f"Z: {pos['z']:.0f} km<extra></extra>"
                ))
            
            # Add asteroids
            x_coords = []
            y_coords = []
            z_coords = []
            colors = []
            sizes = []
            hover_texts = []
            
            for asteroid in asteroids:
                pos = asteroid['position']
                x_coords.append(pos['x'])
                y_coords.append(pos['y'])
                z_coords.append(pos['z'])
                
                risk_level = asteroid.get('risk_level', 'LOW')
                if risk_level == 'HIGH':
                    colors.append('red')
                    sizes.append(8)
                elif risk_level == 'MEDIUM':
                    colors.append('orange')
                    sizes.append(6)
                else:
                    colors.append('green')
                    sizes.append(4)
                
                hover_texts.append(
                    f"ID: {asteroid['id']}<br>" +
                    f"Size: {asteroid['size']:.2f} km<br>" +
                    f"Risk: {risk_level}<br>" +
                    f"Collision Prob: {asteroid.get('enhanced_collision_probability', 0):.8f}"
                )
            
            fig.add_trace(go.Scatter3d(
                x=x_coords, y=y_coords, z=z_coords,
                mode='markers',
                marker=dict(size=sizes, color=colors),
                name='Asteroids',
                hovertemplate='%{hovertext}<extra></extra>',
                hovertext=hover_texts
            ))
            
            # Add Earth orbit
            theta = np.linspace(0, 2*np.pi, 100)
            earth_orbit_x = self.AU * np.cos(theta)
            earth_orbit_y = self.AU * np.sin(theta)
            earth_orbit_z = np.zeros_like(theta)
            
            fig.add_trace(go.Scatter3d(
                x=earth_orbit_x, y=earth_orbit_y, z=earth_orbit_z,
                mode='lines',
                line=dict(color='lightblue', width=2),
                name='Earth Orbit',
                hoverinfo='none'
            ))
            
            fig.update_layout(
                title="3D Solar System View",
                scene=dict(
                    xaxis_title="X (km)",
                    yaxis_title="Y (km)",
                    zaxis_title="Z (km)",
                    aspectmode='cube'
                ),
                height=600
            )
        
        elif view_type == '2d':
            # 2D Orbital View
            fig = go.Figure()
            
            # Add Sun
            fig.add_trace(go.Scatter(
                x=[0], y=[0],
                mode='markers',
                marker=dict(size=15, color='yellow'),
                name='Sun'
            ))
            
            # Add planets
            for planet in self.planet_data:
                pos = planet['position']
                fig.add_trace(go.Scatter(
                    x=[pos['x']], y=[pos['y']],
                    mode='markers',
                    marker=dict(size=8, color='blue'),
                    name=planet['planet']
                ))
            
            # Add asteroids
            for risk_level in ['HIGH', 'MEDIUM', 'LOW']:
                risk_asteroids = [a for a in asteroids if a.get('risk_level') == risk_level]
                if not risk_asteroids:
                    continue
                
                x_coords = [a['position']['x'] for a in risk_asteroids]
                y_coords = [a['position']['y'] for a in risk_asteroids]
                
                color = 'red' if risk_level == 'HIGH' else 'orange' if risk_level == 'MEDIUM' else 'green'
                
                fig.add_trace(go.Scatter(
                    x=x_coords, y=y_coords,
                    mode='markers',
                    marker=dict(color=color, size=6),
                    name=f'{risk_level} Risk Asteroids'
                ))
            
            # Add Earth orbit
            theta = np.linspace(0, 2*np.pi, 100)
            fig.add_trace(go.Scatter(
                x=self.AU * np.cos(theta),
                y=self.AU * np.sin(theta),
                mode='lines',
                line=dict(color='lightblue', dash='dash'),
                name='Earth Orbit'
            ))
            
            fig.update_layout(
                title="2D Orbital View",
                xaxis_title="X (km)",
                yaxis_title="Y (km)",
                height=600
            )
        
        elif view_type == 'earth':
            # Earth-Centric View
            fig = go.Figure()
            
            earth_pos = next((p['position'] for p in self.planet_data if p['planet'] == 'Earth'), 
                           {'x': self.AU, 'y': 0, 'z': 0})
            
            # Add Earth
            fig.add_trace(go.Scatter3d(
                x=[earth_pos['x']], y=[earth_pos['y']], z=[earth_pos['z']],
                mode='markers',
                marker=dict(size=15, color='blue'),
                name='Earth'
            ))
            
            # Add nearby asteroids (within 0.1 AU)
            nearby_asteroids = [a for a in asteroids 
                              if a.get('distance_from_earth', float('inf')) < 0.1 * self.AU]
            
            if nearby_asteroids:
                x_coords = [a['position']['x'] for a in nearby_asteroids]
                y_coords = [a['position']['y'] for a in nearby_asteroids]
                z_coords = [a['position']['z'] for a in nearby_asteroids]
                colors = ['red' if a.get('risk_level') == 'HIGH' else 'orange' 
                         if a.get('risk_level') == 'MEDIUM' else 'green' for a in nearby_asteroids]
                
                fig.add_trace(go.Scatter3d(
                    x=x_coords, y=y_coords, z=z_coords,
                    mode='markers',
                    marker=dict(color=colors, size=8),
                    name='Nearby Asteroids'
                ))
            
            fig.update_layout(
                title="Earth-Centric View (Nearby Asteroids)",
                scene=dict(
                    xaxis_title="X (km)",
                    yaxis_title="Y (km)",
                    zaxis_title="Z (km)"
                ),
                height=600
            )
        
        return fig
    
    def create_statistics_panel(self, asteroids):
        """Create statistics panel"""
        total_asteroids = len(asteroids)
        high_risk = len([a for a in asteroids if a.get('risk_level') == 'HIGH'])
        medium_risk = len([a for a in asteroids if a.get('risk_level') == 'MEDIUM'])
        low_risk = len([a for a in asteroids if a.get('risk_level') == 'LOW'])
        
        avg_size = np.mean([a['size'] for a in asteroids]) if asteroids else 0
        max_collision_prob = max([a.get('enhanced_collision_probability', 0) for a in asteroids]) if asteroids else 0
        
        return html.Div([
            html.P(f"Total Asteroids: {total_asteroids}", style={'fontSize': '16px', 'margin': '5px'}),
            html.P(f"High Risk: {high_risk}", style={'fontSize': '14px', 'color': 'red', 'margin': '5px'}),
            html.P(f"Medium Risk: {medium_risk}", style={'fontSize': '14px', 'color': 'orange', 'margin': '5px'}),
            html.P(f"Low Risk: {low_risk}", style={'fontSize': '14px', 'color': 'green', 'margin': '5px'}),
            html.Hr(),
            html.P(f"Avg Size: {avg_size:.2f} km", style={'fontSize': '14px', 'margin': '5px'}),
            html.P(f"Max Collision Prob: {max_collision_prob:.8f}", style={'fontSize': '14px', 'margin': '5px'}),
            html.Hr(),
            html.P(f"Last Updated: {datetime.now().strftime('%H:%M:%S')}", 
                  style={'fontSize': '12px', 'color': 'gray', 'margin': '5px'})
        ])
    
    def create_collision_probability_plot(self, asteroids):
        """Create collision probability histogram"""
        if not asteroids:
            return go.Figure().update_layout(title="No data available")
        
        probs = [a.get('enhanced_collision_probability', 0) for a in asteroids]
        
        fig = px.histogram(
            x=probs,
            nbins=20,
            title="Collision Probability Distribution",
            labels={'x': 'Collision Probability', 'y': 'Count'}
        )
        
        return fig
    
    def create_size_distribution_plot(self, asteroids):
        """Create size distribution plot"""
        if not asteroids:
            return go.Figure().update_layout(title="No data available")
        
        sizes = [a['size'] for a in asteroids]
        risk_levels = [a.get('risk_level', 'LOW') for a in asteroids]
        
        fig = px.box(
            x=risk_levels,
            y=sizes,
            title="Asteroid Size Distribution by Risk Level",
            labels={'x': 'Risk Level', 'y': 'Size (km)'}
        )
        
        return fig
    
    def create_alerts_panel(self):
        """Create alerts panel"""
        if not self.alert_data:
            return html.P("No active alerts", style={'color': 'green'})
        
        recent_alerts = sorted(self.alert_data, key=lambda x: x['timestamp'], reverse=True)[:5]
        
        alert_elements = []
        for alert in recent_alerts:
            alert_elements.append(
                html.Div([
                    html.Strong(f"Alert: {alert['alert_id']}"),
                    html.Br(),
                    html.Span(f"Asteroid: {alert['asteroid_id']}"),
                    html.Br(),
                    html.Span(f"Probability: {alert['collision_probability']:.8f}"),
                    html.Br(),
                    html.Span(f"Level: {alert['alert_level']}"),
                    html.Br(),
                    html.Small(f"Time: {alert['timestamp']}")
                ], style={'border': '1px solid #ccc', 'padding': '10px', 'margin': '5px', 
                         'backgroundColor': '#ffe6e6' if alert['alert_level'] == 'HIGH' else '#fff3e6'})
            )
        
        return html.Div(alert_elements)
    
    def create_data_table(self, asteroids):
        """Create data table"""
        if not asteroids:
            return html.P("No data available")
        
        # Limit to top 20 by collision probability
        top_asteroids = sorted(asteroids, 
                             key=lambda x: x.get('enhanced_collision_probability', 0), 
                             reverse=True)[:20]
        
        table_data = []
        for asteroid in top_asteroids:
            table_data.append(html.Tr([
                html.Td(asteroid['id']),
                html.Td(f"{asteroid['size']:.2f}"),
                html.Td(f"{asteroid.get('distance_from_earth', 0):.0f}"),
                html.Td(f"{asteroid.get('enhanced_collision_probability', 0):.8f}"),
                html.Td(asteroid.get('risk_level', 'LOW'))
            ]))
        
        return html.Table([
            html.Thead([
                html.Tr([
                    html.Th("ID"),
                    html.Th("Size (km)"),
                    html.Th("Earth Distance (km)"),
                    html.Th("Collision Probability"),
                    html.Th("Risk Level")
                ])
            ]),
            html.Tbody(table_data)
        ], style={'width': '100%', 'border': '1px solid #ccc'})
    
    def run_dashboard(self, host='0.0.0.0', port=8050, debug=False):
        """Run the dashboard"""
        logger.info(f"Starting dashboard on http://{host}:{port}")
        self.app.run_server(host=host, port=port, debug=debug)

def main():
    """Main function to run the visualization dashboard"""
    dashboard = AsteroidVisualizationDashboard()
    dashboard.run_dashboard(debug=True)

if __name__ == "__main__":
    main()