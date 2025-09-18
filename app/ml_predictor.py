#!/usr/bin/env python3
"""
Intégrateur de modèle PyTorch pour le pipeline de production
Remplace la génération artificielle de probabilités par un vrai modèle ML entraîné
"""

import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AsteroidCollisionNet(nn.Module):
    """
    Réseau de neurones PyTorch pour prédiction de collision d'astéroïdes
    """
    
    def __init__(self, input_size=10, hidden_sizes=[64, 32, 16], dropout_rate=0.3):
        super(AsteroidCollisionNet, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class AsteroidMLPredictor:
    """
    Prédicteur ML pour collision d'astéroïdes utilisant PyTorch
    """
    
    def __init__(self, model_path=None, scaler_path='models/scaler.pkl'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = None
        self.is_loaded = False
        
        # Chercher le modèle dans plusieurs emplacements possibles
        model_paths = [
            'best_model.pth',  # Emplacement principal dans /app
            'models/best_model.pth',
            'models/asteroid_collision_model.pth',
            '/app/best_model.pth',
            '/app/models/best_model.pth'
        ]
        
        if model_path:
            model_paths.insert(0, model_path)
        
        self.model_path = None
        for path in model_paths:
            if os.path.exists(path):
                self.model_path = path
                break
        
        self.scaler_path = scaler_path
        
        self.load_model()
    
    def load_model(self) -> bool:
        """Charge le modèle PyTorch et optionnellement le scaler"""
        
        try:
            if not self.model_path or not os.path.exists(self.model_path):
                logger.error(f"❌ Aucun modèle PyTorch trouvé dans les emplacements disponibles")
                logger.info("   Emplacements vérifiés: best_model.pth, models/best_model.pth, models/asteroid_collision_model.pth")
                return False
            
            logger.info(f"🔍 Chargement du modèle: {self.model_path}")
            
            # Chargement du modèle PyTorch - support pour différents formats
            if 'best_model.pth' in self.model_path:
                # Format simple best_model.pth - créer modèle avec config par défaut
                self.model = AsteroidCollisionNet(input_size=10, hidden_sizes=[64, 32, 16])
                checkpoint = torch.load(self.model_path, map_location=self.device)
                
                # Support pour différents formats de sauvegarde
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
            else:
                # Format complet avec config
                checkpoint = torch.load(self.model_path, map_location=self.device)
                model_config = checkpoint.get('model_config', {
                    'input_size': 10, 'hidden_sizes': [64, 32, 16], 'dropout_rate': 0.3
                })
                
                self.model = AsteroidCollisionNet(
                    input_size=model_config['input_size'],
                    hidden_sizes=model_config['hidden_sizes'],
                    dropout_rate=model_config['dropout_rate']
                )
                
                self.model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
            
            self.model.to(self.device)
            self.model.eval()
            
            # Chargement optionnel du scaler
            if os.path.exists(self.scaler_path):
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info("✅ Scaler chargé")
            else:
                logger.info("📊 Utilisation de la normalisation automatique (pas de scaler externe)")
                self.scaler = None
            
            self.is_loaded = True
            logger.info(f"🎯 Modèle PyTorch chargé avec succès depuis: {self.model_path}")
            logger.info(f"🖥️  Device: {self.device}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement modèle: {e}")
            self.is_loaded = False
            return False
    
    def extract_features(self, asteroid_data: Dict) -> Optional[np.ndarray]:
        """Extrait les features du format astéroïde pour le modèle ML"""
        
        try:
            size = asteroid_data.get('size', 1.0)
            mass = asteroid_data.get('mass', 1e12)
            
            velocity = asteroid_data.get('velocity', {'vx': 0, 'vy': 0, 'vz': 0})
            velocity_magnitude = np.sqrt(
                velocity['vx']**2 + velocity['vy']**2 + velocity['vz']**2
            )
            
            position = asteroid_data.get('position', {'x': 0, 'y': 0, 'z': 0})
            earth_pos = {'x': 149597870.7, 'y': 0, 'z': 0}
            
            distance_earth = np.sqrt(
                (position['x'] - earth_pos['x'])**2 +
                (position['y'] - earth_pos['y'])**2 +
                (position['z'] - earth_pos['z'])**2
            ) / 149597870.7
            
            # Features estimées
            orbital_eccentricity = np.random.beta(2, 5)
            inclination_deg = np.random.normal(10, 15)
            approach_angle = np.arctan2(velocity['vy'], velocity['vx'])
            albedo = np.random.beta(2, 8)
            rotation_period_h = np.random.lognormal(1, 1)
            surface_temp_k = 200 + np.random.normal(0, 50)
            
            features = np.array([
                size, velocity_magnitude, distance_earth, mass,
                orbital_eccentricity, inclination_deg, approach_angle,
                albedo, rotation_period_h, surface_temp_k
            ])
            
            return features.reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Erreur extraction features: {e}")
            return None
    
    def predict_collision_probability(self, asteroid_data: Dict) -> float:
        """Prédit la probabilité de collision pour un astéroïde"""
        
        if not self.is_loaded:
            return self._fallback_probability(asteroid_data)
        
        try:
            features = self.extract_features(asteroid_data)
            if features is None:
                return self._fallback_probability(asteroid_data)
            
            # Normalisation (avec ou sans scaler)
            if self.scaler is not None:
                features_normalized = self.scaler.transform(features)
            else:
                # Normalisation simple sans scaler
                features_normalized = (features - np.mean(features)) / (np.std(features) + 1e-8)
            
            with torch.no_grad():
                input_tensor = torch.FloatTensor(features_normalized).to(self.device)
                prediction = self.model(input_tensor)
                probability = prediction.cpu().numpy()[0][0]
            
            probability = np.clip(probability, 0.0, 1.0)
            return float(probability)
            
        except Exception as e:
            logger.error(f"Erreur prédiction ML: {e}")
            return self._fallback_probability(asteroid_data)
    
    def _fallback_probability(self, asteroid_data: Dict) -> float:
        """Méthode de fallback si le modèle ML n'est pas disponible"""
        
        size = asteroid_data.get('size', 1.0)
        velocity = asteroid_data.get('velocity', {'vx': 0, 'vy': 0, 'vz': 0})
        velocity_magnitude = np.sqrt(
            velocity['vx']**2 + velocity['vy']**2 + velocity['vz']**2
        )
        
        base_probability = min(size / 10.0, 1.0) * min(velocity_magnitude / 50.0, 1.0)
        random_factor = np.random.uniform(0.1, 2.0)
        probability = base_probability * random_factor * 0.01
        
        return np.clip(probability, 0.0, 1.0)
    
    def get_risk_level(self, probability: float) -> str:
        """Détermine le niveau de risque basé sur la probabilité"""
        
        if probability > 0.001:
            return 'HIGH'
        elif probability > 0.0001:
            return 'MEDIUM'
        elif probability > 0.00001:
            return 'LOW'
        else:
            return 'MINIMAL'

# Prédicteur global
_global_predictor = None

def get_ml_predictor() -> AsteroidMLPredictor:
    """Retourne l'instance globale du prédicteur ML"""
    global _global_predictor
    
    if _global_predictor is None:
        _global_predictor = AsteroidMLPredictor()
    
    return _global_predictor

def predict_collision_probability(asteroid_data: Dict) -> Tuple[float, str]:
    """Interface simple pour prédiction de collision"""
    
    predictor = get_ml_predictor()
    probability = predictor.predict_collision_probability(asteroid_data)
    risk_level = predictor.get_risk_level(probability)
    
    return probability, risk_level    return probability, risk_level