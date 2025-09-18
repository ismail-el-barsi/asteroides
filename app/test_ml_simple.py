#!/usr/bin/env python3
"""
Test simple pour vérifier que le modèle ML fonctionne dans l'environnement Docker
"""

try:
    from ml_predictor import predict_collision_probability

    # Test avec un astéroïde exemple
    test_asteroid = {
        "id": "asteroid_test_001",
        "position": {"x": 345000.0, "y": -120000.0, "z": 50700.0},
        "velocity": {"vx": 25.0, "vy": -40.0, "vz": 10.0},
        "size": 1.5,
        "mass": 2.5e12
    }
    
    print("🧪 Test du modèle ML dans Docker")
    print("=" * 40)
    
    # Prédiction
    probability, risk_level = predict_collision_probability(test_asteroid)
    
    print(f"✅ Prédiction réussie:")
    print(f"   Astéroïde: {test_asteroid['id']}")
    print(f"   Probabilité collision: {probability:.8f}")
    print(f"   Niveau de risque: {risk_level}")
    print("🎉 Le modèle ML fonctionne correctement!")
    
except Exception as e:
    print(f"❌ Erreur: {e}")
    import traceback
    traceback.print_exc()