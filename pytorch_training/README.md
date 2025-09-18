# PyTorch Training Environment

Environnement d'entraînement PyTorch pour le modèle de prédiction de collision d'astéroïdes.

## Installation

```bash
# Créer un environnement virtuel
python -m venv pytorch_env
pytorch_env\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

## Utilisation

### 1. Test de l'environnement
```bash
python test_setup.py
```

### 2. Entraînement du modèle
```bash
python asteroid_collision_pytorch.py
```

### 3. Utilisation dans Docker
Une fois l'entraînement terminé:
1. Le fichier `best_model.pth` est généré
2. Copier vers `../app/models/asteroid_collision_model.pth`
3. Le modèle est automatiquement utilisé dans l'application Docker

## Architecture du Modèle

- **Input:** 10 features d'astéroïdes
- **Hidden Layers:** [64, 32, 16] neurones
- **Output:** Probabilité de collision (0-1)
- **Activation:** ReLU + Sigmoid finale
- **Régularisation:** Dropout + BatchNorm

## Features d'Astéroïdes

1. `diameter_km` - Diamètre en kilomètres
2. `velocity_km_s` - Vitesse en km/s
3. `distance_earth_au` - Distance à la Terre en UA
4. `mass_kg` - Masse en kilogrammes
5. `orbital_eccentricity` - Excentricité orbitale
6. `inclination_deg` - Inclinaison orbitale en degrés
7. `approach_angle` - Angle d'approche
8. `albedo` - Albédo de surface
9. `rotation_period_h` - Période de rotation en heures
10. `surface_temp_k` - Température de surface en Kelvin

## Fichiers Générés

- `best_model.pth` - Meilleur modèle entraîné
- `models/asteroid_collision_model.pth` - Version avec configuration
- `models/scaler.pkl` - Normalisation des données

## Performance Attendue

- **Accuracy:** > 85%
- **AUC-ROC:** > 0.90
- **Temps d'entraînement:** ~5-10 minutes

## Notes

- L'entraînement utilise early stopping (patience=15)
- Le modèle est sauvegardé automatiquement au meilleur epoch
- Les données sont synthétiques mais réalistes
- Le modèle est intégré automatiquement dans l'application Docker