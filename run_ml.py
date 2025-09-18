#!/usr/bin/env python3
"""
Script de lancement pour l'application avec modèle ML PyTorch
"""

import subprocess
import sys


def main():
    """Lance l'application avec le modèle ML dans Docker"""
    print("🚀 Lancement de l'application avec modèle ML PyTorch...")
    print("-" * 60)
    
    try:
        # Tester si le modèle ML fonctionne dans le conteneur
        print("🧪 Test du modèle ML...")
        result = subprocess.run([
            'docker', 'exec', 'asteroid-app', 
            'python', 'test_ml_simple.py'
        ], capture_output=False, text=True)
        
        if result.returncode == 0:
            print("\n✅ Modèle ML testé avec succès!")
            print("\n🌐 Dashboard disponible sur: http://localhost:8050")
            print("   Le système utilise maintenant de vraies prédictions ML!")
        else:
            print(f"\n⚠️  Test ML avec code de retour: {result.returncode}")
            print("   L'application fonctionne avec le mode fallback")
            
    except Exception as e:
        print(f"❌ Erreur: {e}")
        print("💡 Assurez-vous que le conteneur 'asteroid-app' est démarré")
        print("   Commande: docker-compose up -d")

if __name__ == "__main__":
    main()
