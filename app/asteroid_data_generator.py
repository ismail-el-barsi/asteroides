#!/usr/bin/env python3
"""
Asteroid Data Generator
Generates continuous simulated asteroid and planet data for collision prediction system
"""

import json
import math
import random
import time
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
from kafka import KafkaProducer


class AsteroidDataGenerator:
    def __init__(self, kafka_bootstrap_servers: List[str] = ['kafka:9092']):
        """Initialize the asteroid data generator"""
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None
        )
        
        # Astronomical units in kilometers
        self.AU = 149597870.7  # 1 AU in km
        
        # Known planets data (simplified positions)
        self.planets = {
            "Mercury": {"distance": 0.39 * self.AU, "period": 88, "mass": 3.3011e23},
            "Venus": {"distance": 0.72 * self.AU, "period": 225, "mass": 4.8675e24},
            "Earth": {"distance": 1.0 * self.AU, "period": 365, "mass": 5.972e24},
            "Mars": {"distance": 1.52 * self.AU, "period": 687, "mass": 6.4171e23},
            "Jupiter": {"distance": 5.20 * self.AU, "period": 4333, "mass": 1.8982e27},
            "Saturn": {"distance": 9.54 * self.AU, "period": 10759, "mass": 5.6834e26}
        }
        
        # Asteroid tracking
        self.active_asteroids = {}
        self.asteroid_counter = 0
        
    def generate_asteroid_data(self) -> Dict:
        """Generate realistic asteroid data"""
        self.asteroid_counter += 1
        asteroid_id = f"asteroid_{self.asteroid_counter:06d}"
        
        # Generate initial position (between Mars and Jupiter orbit mostly)
        distance = random.uniform(1.5 * self.AU, 5.0 * self.AU)
        angle = random.uniform(0, 2 * math.pi)
        inclination = random.gauss(0, 0.1)  # Most asteroids have low inclination
        
        position = {
            "x": distance * math.cos(angle) * math.cos(inclination),
            "y": distance * math.sin(angle) * math.cos(inclination),
            "z": distance * math.sin(inclination)
        }
        
        # Generate velocity (orbital mechanics approximation)
        orbital_speed = math.sqrt(1.327e11 / distance)  # Simplified orbital velocity
        velocity_perturbation = random.uniform(0.8, 1.2)
        
        velocity = {
            "vx": -orbital_speed * math.sin(angle) * velocity_perturbation + random.gauss(0, 5),
            "vy": orbital_speed * math.cos(angle) * velocity_perturbation + random.gauss(0, 5),
            "vz": random.gauss(0, 2)
        }
        
        # Asteroid physical properties
        size = random.lognormvariate(0, 1)  # Size in kilometers
        density = random.uniform(2000, 5000)  # kg/m³
        volume = (4/3) * math.pi * (size * 500) ** 3  # Convert to meters
        mass = density * volume
        
        asteroid_data = {
            "id": asteroid_id,
            "timestamp": datetime.now().isoformat(),
            "position": position,
            "velocity": velocity,
            "size": round(size, 3),
            "mass": mass,
            "density": density,
            "classification": self._classify_asteroid(distance),
            "discovery_date": datetime.now().isoformat(),
            "orbital_period": self._calculate_orbital_period(distance),
            "eccentricity": random.uniform(0.05, 0.3),
            "potentially_hazardous": self._is_potentially_hazardous(position, velocity, size)
        }
        
        self.active_asteroids[asteroid_id] = asteroid_data
        return asteroid_data
    
    def update_asteroid_position(self, asteroid_id: str, time_delta: float = 3600) -> Dict:
        """Update asteroid position based on simple orbital mechanics"""
        if asteroid_id not in self.active_asteroids:
            return None
            
        asteroid = self.active_asteroids[asteroid_id].copy()
        
        # Simple numerical integration (Euler method)
        # In reality, this would use more sophisticated orbital mechanics
        pos = asteroid["position"]
        vel = asteroid["velocity"]
        
        # Gravitational acceleration from Sun (simplified)
        r = math.sqrt(pos["x"]**2 + pos["y"]**2 + pos["z"]**2)
        if r > 0:
            g_sun = 1.327e11  # GM_sun in km³/s²
            acc_magnitude = g_sun / (r**2)
            
            acc = {
                "ax": -acc_magnitude * pos["x"] / r,
                "ay": -acc_magnitude * pos["y"] / r,
                "az": -acc_magnitude * pos["z"] / r
            }
            
            # Update velocity
            vel["vx"] += acc["ax"] * time_delta
            vel["vy"] += acc["ay"] * time_delta
            vel["vz"] += acc["az"] * time_delta
            
            # Update position
            pos["x"] += vel["vx"] * time_delta
            pos["y"] += vel["vy"] * time_delta
            pos["z"] += vel["vz"] * time_delta
        
        asteroid["timestamp"] = datetime.now().isoformat()
        asteroid["potentially_hazardous"] = self._is_potentially_hazardous(pos, vel, asteroid["size"])
        
        self.active_asteroids[asteroid_id] = asteroid
        return asteroid
    
    def generate_planet_data(self, planet_name: str) -> Dict:
        """Generate current planet position data"""
        if planet_name not in self.planets:
            return None
            
        planet_info = self.planets[planet_name]
        
        # Simple circular orbit approximation
        current_time = time.time()
        angle = (current_time / (planet_info["period"] * 24 * 3600)) * 2 * math.pi
        
        position = {
            "x": planet_info["distance"] * math.cos(angle),
            "y": planet_info["distance"] * math.sin(angle),
            "z": 0  # Simplified to ecliptic plane
        }
        
        # Orbital velocity
        orbital_speed = 2 * math.pi * planet_info["distance"] / (planet_info["period"] * 24 * 3600)
        velocity = {
            "vx": -orbital_speed * math.sin(angle),
            "vy": orbital_speed * math.cos(angle),
            "vz": 0
        }
        
        return {
            "planet": planet_name,
            "timestamp": datetime.now().isoformat(),
            "position": position,
            "velocity": velocity,
            "mass": planet_info["mass"],
            "orbital_period": planet_info["period"]
        }
    
    def _classify_asteroid(self, distance: float) -> str:
        """Classify asteroid based on orbital distance"""
        if distance < 1.3 * self.AU:
            return "NEA"  # Near-Earth Asteroid
        elif distance < 2.5 * self.AU:
            return "MBA_Inner"  # Main Belt Asteroid - Inner
        elif distance < 3.3 * self.AU:
            return "MBA_Outer"  # Main Belt Asteroid - Outer
        else:
            return "Outer"  # Outer asteroid
    
    def _calculate_orbital_period(self, distance: float) -> float:
        """Calculate orbital period using Kepler's third law"""
        # T² ∝ a³ (in years for distance in AU)
        distance_au = distance / self.AU
        return math.sqrt(distance_au ** 3) * 365.25  # days
    
    def _is_potentially_hazardous(self, position: Dict, velocity: Dict, size: float) -> bool:
        """Determine if asteroid is potentially hazardous"""
        earth_distance = math.sqrt(
            (position["x"] - self.AU)**2 + 
            position["y"]**2 + 
            position["z"]**2
        )
        
        # PHO criteria: within 0.05 AU (7.5 million km) and diameter > 140m
        return earth_distance < 0.05 * self.AU and size > 0.14
    
    def calculate_collision_probability(self, asteroid_data: Dict) -> float:
        """Calculate rough collision probability with Earth"""
        pos = asteroid_data["position"]
        vel = asteroid_data["velocity"]
        
        # Distance to Earth
        earth_pos = self.generate_planet_data("Earth")["position"]
        distance_to_earth = math.sqrt(
            (pos["x"] - earth_pos["x"])**2 +
            (pos["y"] - earth_pos["y"])**2 +
            (pos["z"] - earth_pos["z"])**2
        )
        
        # Simple collision probability calculation
        # Real calculations would involve complex orbital mechanics
        earth_radius = 6371  # km
        earth_soi = 0.01 * self.AU  # Sphere of influence (simplified)
        
        if distance_to_earth > earth_soi:
            return 0.0
        
        # Probability increases as distance decreases and size increases
        size_factor = min(asteroid_data["size"] / 10.0, 1.0)
        distance_factor = max(0, 1 - (distance_to_earth / earth_soi))
        
        probability = size_factor * distance_factor * 0.001  # Very low base probability
        return min(probability, 1.0)
    
    def publish_asteroid_data(self, asteroid_data: Dict):
        """Publish asteroid data to Kafka topic"""
        try:
            # Add collision probability
            asteroid_data["collision_probability"] = self.calculate_collision_probability(asteroid_data)
            
            self.producer.send(
                'asteroid-data',
                key=asteroid_data["id"],
                value=asteroid_data
            )
            print(f"Published asteroid data: {asteroid_data['id']}")
        except Exception as e:
            print(f"Error publishing asteroid data: {e}")
    
    def publish_planet_data(self, planet_data: Dict):
        """Publish planet data to Kafka topic"""
        try:
            self.producer.send(
                'planet-data',
                key=planet_data["planet"],
                value=planet_data
            )
            print(f"Published planet data: {planet_data['planet']}")
        except Exception as e:
            print(f"Error publishing planet data: {e}")
    
    def run_continuous_generation(self, asteroid_interval: float = 30, planet_interval: float = 300):
        """Run continuous data generation"""
        print("Starting asteroid data generation...")
        last_planet_update = 0
        
        try:
            while True:
                current_time = time.time()
                
                # Generate new asteroid data
                asteroid_data = self.generate_asteroid_data()
                self.publish_asteroid_data(asteroid_data)
                
                # Update existing asteroid positions
                for asteroid_id in list(self.active_asteroids.keys()):
                    updated_asteroid = self.update_asteroid_position(asteroid_id)
                    if updated_asteroid:
                        self.publish_asteroid_data(updated_asteroid)
                
                # Update planet positions periodically
                if current_time - last_planet_update > planet_interval:
                    for planet_name in self.planets.keys():
                        planet_data = self.generate_planet_data(planet_name)
                        if planet_data:
                            self.publish_planet_data(planet_data)
                    last_planet_update = current_time
                
                # Check for collision alerts
                self._check_collision_alerts()
                
                time.sleep(asteroid_interval)
                
        except KeyboardInterrupt:
            print("Stopping data generation...")
        finally:
            self.producer.close()
    
    def _check_collision_alerts(self):
        """Check for high-probability collision scenarios"""
        for asteroid_id, asteroid in self.active_asteroids.items():
            if asteroid.get("collision_probability", 0) > 0.0001:  # 0.01% threshold
                alert = {
                    "alert_id": f"alert_{asteroid_id}_{int(time.time())}",
                    "timestamp": datetime.now().isoformat(),
                    "asteroid_id": asteroid_id,
                    "collision_probability": asteroid["collision_probability"],
                    "alert_level": "HIGH" if asteroid["collision_probability"] > 0.001 else "MEDIUM",
                    "estimated_impact_time": self._estimate_impact_time(asteroid),
                    "asteroid_data": asteroid
                }
                
                try:
                    self.producer.send('collision-alerts', value=alert)
                    print(f"COLLISION ALERT: {asteroid_id} - Probability: {asteroid['collision_probability']:.6f}")
                except Exception as e:
                    print(f"Error sending collision alert: {e}")
    
    def _estimate_impact_time(self, asteroid: Dict) -> str:
        """Estimate potential impact time (simplified calculation)"""
        # This is a very simplified calculation
        # Real impact prediction requires sophisticated orbital mechanics
        pos = asteroid["position"]
        vel = asteroid["velocity"]
        
        earth_distance = math.sqrt(pos["x"]**2 + pos["y"]**2 + pos["z"]**2)
        velocity_magnitude = math.sqrt(vel["vx"]**2 + vel["vy"]**2 + vel["vz"]**2)
        
        if velocity_magnitude > 0:
            # Very rough time estimate
            time_to_impact = earth_distance / velocity_magnitude  # seconds
            impact_time = datetime.fromtimestamp(time.time() + time_to_impact)
            return impact_time.isoformat()
        
        return "Unknown"

def main():
    """Main function to run the asteroid data generator"""
    print("Initializing Asteroid Data Generator...")
    
    # Wait for Kafka to be ready
    print("Waiting for Kafka to be ready...")
    time.sleep(30)
    
    generator = AsteroidDataGenerator()
    
    # Generate some initial data
    print("Generating initial planet positions...")
    for planet_name in generator.planets.keys():
        planet_data = generator.generate_planet_data(planet_name)
        if planet_data:
            generator.publish_planet_data(planet_data)
    
    # Start continuous generation
    print("Starting continuous asteroid tracking...")
    generator.run_continuous_generation(asteroid_interval=10, planet_interval=120)

if __name__ == "__main__":
    main()if __name__ == "__main__":
    main()