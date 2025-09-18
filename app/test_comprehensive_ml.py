#!/usr/bin/env python3
"""
Comprehensive PyTorch Model Test
Tests the asteroid collision model with various asteroid data scenarios
"""

import json
import math
import random
import time
from datetime import datetime
from typing import Dict, List

# Import du prédicteur ML
try:
    from ml_predictor import predict_collision_probability
    ML_AVAILABLE = True
    print("✅ Modèle ML PyTorch chargé")
except ImportError as e:
    ML_AVAILABLE = False
    print(f"❌ Modèle ML non disponible: {e}")
    exit(1)

class ComprehensiveMLTester:
    def __init__(self):
        """Initialize the comprehensive ML tester"""
        self.AU = 149597870.7  # 1 AU in km
        self.test_results = []
        
    def create_test_asteroid(self, scenario: str, **kwargs) -> Dict:
        """Create test asteroid data for different scenarios"""
        
        base_data = {
            "id": f"test_asteroid_{scenario}_{int(time.time())}",
            "timestamp": datetime.now().isoformat(),
            "discovery_date": datetime.now().isoformat(),
            "asteroid_type": "near_earth",
            "size_category": "medium",
            "classification": "NEA",
            "orbital_period": 365.25,
            "eccentricity": 0.15,
            "potentially_hazardous": False
        }
        
        if scenario == "high_risk_near_earth":
            # Très proche de la Terre, grande taille
            position = {
                "x": 0.95 * self.AU,  # Très proche de l'orbite terrestre
                "y": 0.1 * self.AU,
                "z": 0.05 * self.AU
            }
            velocity = {
                "vx": -25000,  # Vitesse élevée
                "vy": 15000,
                "vz": 5000
            }
            size = 2.5  # Grande taille
            density = 3500
            potentially_hazardous = True
            
        elif scenario == "medium_risk_crossing":
            # Trajectoire de croisement avec la Terre
            position = {
                "x": 1.1 * self.AU,
                "y": 0.3 * self.AU,
                "z": 0.1 * self.AU
            }
            velocity = {
                "vx": -20000,
                "vy": 10000,
                "vz": 2000
            }
            size = 1.2
            density = 2800
            potentially_hazardous = True
            
        elif scenario == "low_risk_distant":
            # Loin de la Terre, petite taille
            position = {
                "x": 2.5 * self.AU,  # Ceinture d'astéroïdes
                "y": 1.8 * self.AU,
                "z": 0.2 * self.AU
            }
            velocity = {
                "vx": -15000,
                "vy": 12000,
                "vz": 1000
            }
            size = 0.3
            density = 2000
            potentially_hazardous = False
            
        elif scenario == "very_high_risk_impact":
            # Trajectoire d'impact direct
            position = {
                "x": 1.0001 * self.AU,  # Presque sur l'orbite terrestre
                "y": 0.001 * self.AU,   # Très proche
                "z": 0.0001 * self.AU
            }
            velocity = {
                "vx": -30000,  # Vitesse très élevée vers la Terre
                "vy": -5000,
                "vz": -1000
            }
            size = 5.0  # Très grande taille
            density = 5000
            potentially_hazardous = True
            
        elif scenario == "safe_outer_belt":
            # Ceinture extérieure, très sûr
            position = {
                "x": 4.2 * self.AU,
                "y": 3.1 * self.AU,
                "z": 0.5 * self.AU
            }
            velocity = {
                "vx": -8000,
                "vy": 6000,
                "vz": 500
            }
            size = 0.8
            density = 1800
            potentially_hazardous = False
            
        elif scenario == "jupiter_trojan":
            # Astéroïde troyen de Jupiter
            position = {
                "x": 5.2 * self.AU,
                "y": 0.8 * self.AU,
                "z": 0.1 * self.AU
            }
            velocity = {
                "vx": -10000,
                "vy": 8000,
                "vz": 200
            }
            size = 1.5
            density = 2200
            potentially_hazardous = False
            
        elif scenario == "fast_small_neo":
            # Petit NEO rapide
            position = {
                "x": 0.98 * self.AU,
                "y": 0.2 * self.AU,
                "z": 0.03 * self.AU
            }
            velocity = {
                "vx": -35000,  # Très rapide
                "vy": 20000,
                "vz": 8000
            }
            size = 0.15  # Petit
            density = 4000
            potentially_hazardous = True
            
        elif scenario == "slow_large_mba":
            # Grand astéroïde de la ceinture principale, lent
            position = {
                "x": 2.8 * self.AU,
                "y": 1.5 * self.AU,
                "z": 0.3 * self.AU
            }
            velocity = {
                "vx": -12000,
                "vy": 8000,
                "vz": 1500
            }
            size = 8.0  # Très grand
            density = 2500
            potentially_hazardous = False
            
        else:
            # Scénario aléatoire
            distance = random.uniform(0.8 * self.AU, 5.0 * self.AU)
            angle = random.uniform(0, 2 * math.pi)
            inclination = random.gauss(0, 0.1)
            
            position = {
                "x": distance * math.cos(angle) * math.cos(inclination),
                "y": distance * math.sin(angle) * math.cos(inclination),
                "z": distance * math.sin(inclination)
            }
            
            orbital_speed = math.sqrt(1.327e11 / distance)
            velocity = {
                "vx": -orbital_speed * math.sin(angle) + random.gauss(0, 5000),
                "vy": orbital_speed * math.cos(angle) + random.gauss(0, 5000),
                "vz": random.gauss(0, 2000)
            }
            
            size = random.uniform(0.1, 10.0)
            density = random.uniform(1500, 6000)
            potentially_hazardous = distance < 1.3 * self.AU and size > 0.14
        
        # Calculate mass and volume
        volume = (4/3) * math.pi * (size * 500) ** 3  # Convert km to m
        mass = density * volume
        
        # Update base data with scenario-specific values
        base_data.update({
            "position": position,
            "velocity": velocity,
            "size": round(size, 3),
            "mass": mass,
            "density": density,
            "potentially_hazardous": potentially_hazardous
        })
        
        # Override with any custom kwargs
        base_data.update(kwargs)
        
        return base_data
    
    def test_scenario(self, scenario_name: str, asteroid_data: Dict) -> Dict:
        """Test a single scenario and return results"""
        print(f"\n🧪 Test Scénario: {scenario_name}")
        print("=" * 50)
        
        # Display asteroid info
        pos = asteroid_data["position"]
        vel = asteroid_data["velocity"]
        
        distance_from_earth = math.sqrt(
            (pos["x"] - self.AU)**2 + pos["y"]**2 + pos["z"]**2
        )
        distance_au = distance_from_earth / self.AU
        
        velocity_magnitude = math.sqrt(vel["vx"]**2 + vel["vy"]**2 + vel["vz"]**2)
        
        print(f"📍 Astéroïde: {asteroid_data['id']}")
        print(f"📏 Taille: {asteroid_data['size']} km")
        print(f"⚖️  Masse: {asteroid_data['mass']:.2e} kg")
        print(f"🌍 Distance Terre: {distance_au:.4f} AU ({distance_from_earth/1e6:.2f} millions km)")
        print(f"🚀 Vitesse: {velocity_magnitude:.0f} km/s")
        print(f"⚠️  Potentiellement dangereux: {asteroid_data['potentially_hazardous']}")
        
        # Test ML prediction
        start_time = time.time()
        try:
            probability, risk_level = predict_collision_probability(asteroid_data)
            prediction_time = time.time() - start_time
            
            print(f"\n🤖 Prédiction ML:")
            print(f"   Probabilité: {probability:.8f} ({probability*100:.6f}%)")
            print(f"   Niveau de risque: {risk_level}")
            print(f"   Temps de calcul: {prediction_time*1000:.2f} ms")
            
            # Risk level validation
            if probability > 0.001:
                expected_risk = "HIGH"
            elif probability > 0.0001:
                expected_risk = "MEDIUM"
            elif probability > 0.00001:
                expected_risk = "LOW"
            else:
                expected_risk = "SAFE"
            
            risk_match = risk_level == expected_risk
            if not risk_match:
                print(f"⚠️  Attention: Niveau de risque attendu {expected_risk}, obtenu {risk_level}")
            
            result = {
                "scenario": scenario_name,
                "asteroid_id": asteroid_data["id"],
                "size": asteroid_data["size"],
                "distance_au": distance_au,
                "velocity_magnitude": velocity_magnitude,
                "potentially_hazardous": asteroid_data["potentially_hazardous"],
                "ml_probability": probability,
                "ml_risk_level": risk_level,
                "prediction_time_ms": prediction_time * 1000,
                "risk_level_match": risk_match,
                "success": True
            }
            
            if probability > 0.0001:
                print(f"🚨 ALERTE: Probabilité élevée détectée!")
            elif probability > 0.00001:
                print(f"⚡ Attention: Risque modéré détecté")
            else:
                print(f"✅ Risque faible - Situation normale")
                
        except Exception as e:
            print(f"❌ Erreur lors de la prédiction: {e}")
            result = {
                "scenario": scenario_name,
                "asteroid_id": asteroid_data["id"],
                "error": str(e),
                "success": False
            }
        
        return result
    
    def run_comprehensive_tests(self) -> List[Dict]:
        """Run comprehensive tests with multiple scenarios"""
        print("🚀 DÉBUT DES TESTS COMPRÉHENSIFS DU MODÈLE PYTORCH")
        print("=" * 70)
        
        # Define test scenarios
        scenarios = [
            "high_risk_near_earth",
            "medium_risk_crossing", 
            "low_risk_distant",
            "very_high_risk_impact",
            "safe_outer_belt",
            "jupiter_trojan",
            "fast_small_neo",
            "slow_large_mba"
        ]
        
        results = []
        
        # Test each scenario
        for scenario in scenarios:
            asteroid_data = self.create_test_asteroid(scenario)
            result = self.test_scenario(scenario, asteroid_data)
            results.append(result)
            self.test_results.append(result)
            time.sleep(1)  # Small delay between tests
        
        # Test with random asteroids
        print(f"\n🎲 Tests avec astéroïdes aléatoires...")
        for i in range(5):
            asteroid_data = self.create_test_asteroid("random")
            result = self.test_scenario(f"random_{i+1}", asteroid_data)
            results.append(result)
            self.test_results.append(result)
            time.sleep(0.5)
        
        return results
    
    def analyze_results(self):
        """Analyze and display test results summary"""
        print(f"\n📊 ANALYSE DES RÉSULTATS")
        print("=" * 50)
        
        successful_tests = [r for r in self.test_results if r.get("success", False)]
        failed_tests = [r for r in self.test_results if not r.get("success", False)]
        
        print(f"✅ Tests réussis: {len(successful_tests)}/{len(self.test_results)}")
        print(f"❌ Tests échoués: {len(failed_tests)}")
        
        if successful_tests:
            probabilities = [r["ml_probability"] for r in successful_tests]
            prediction_times = [r["prediction_time_ms"] for r in successful_tests]
            
            print(f"\n📈 Statistiques des prédictions:")
            print(f"   Probabilité min: {min(probabilities):.8f}")
            print(f"   Probabilité max: {max(probabilities):.8f}")
            print(f"   Probabilité moyenne: {sum(probabilities)/len(probabilities):.8f}")
            print(f"   Temps de prédiction moyen: {sum(prediction_times)/len(prediction_times):.2f} ms")
            
            # Analyze risk levels
            risk_levels = {}
            for result in successful_tests:
                risk = result["ml_risk_level"]
                risk_levels[risk] = risk_levels.get(risk, 0) + 1
            
            print(f"\n📊 Distribution des niveaux de risque:")
            for risk, count in sorted(risk_levels.items()):
                percentage = (count / len(successful_tests)) * 100
                print(f"   {risk}: {count} ({percentage:.1f}%)")
            
            # High risk asteroids
            high_risk = [r for r in successful_tests if r["ml_probability"] > 0.0001]
            if high_risk:
                print(f"\n🚨 Astéroïdes à risque élevé détectés:")
                for r in high_risk:
                    print(f"   {r['scenario']}: {r['ml_probability']:.6f} ({r['ml_risk_level']})")
        
        if failed_tests:
            print(f"\n❌ Tests échoués:")
            for test in failed_tests:
                print(f"   {test['scenario']}: {test.get('error', 'Erreur inconnue')}")
        
        print(f"\n🏆 RÉSUMÉ FINAL:")
        success_rate = (len(successful_tests) / len(self.test_results)) * 100
        print(f"   Taux de réussite: {success_rate:.1f}%")
        
        if success_rate >= 90:
            print(f"   🎉 EXCELLENT! Le modèle fonctionne parfaitement!")
        elif success_rate >= 70:
            print(f"   ✅ BIEN! Le modèle fonctionne correctement")
        else:
            print(f"   ⚠️  ATTENTION! Le modèle nécessite des améliorations")

def main():
    """Main function to run comprehensive ML tests"""
    if not ML_AVAILABLE:
        print("❌ Modèle ML non disponible. Tests annulés.")
        return
    
    print("🧪 TESTS COMPRÉHENSIFS DU MODÈLE PYTORCH")
    print("Asteroid Collision Prediction Model")
    print("=" * 70)
    
    tester = ComprehensiveMLTester()
    
    # Run all tests
    results = tester.run_comprehensive_tests()
    
    # Analyze results
    tester.analyze_results()
    
    # Save results to JSON
    try:
        with open('/tmp/asteroid_data/ml_test_results.json', 'w') as f:
            json.dump(tester.test_results, f, indent=2)
        print(f"\n💾 Résultats sauvegardés dans: /tmp/asteroid_data/ml_test_results.json")
    except Exception as e:
        print(f"⚠️  Impossible de sauvegarder les résultats: {e}")
    
    print(f"\n🎯 Tests terminés! Le modèle PyTorch est opérationnel.")

if __name__ == "__main__":
    main()