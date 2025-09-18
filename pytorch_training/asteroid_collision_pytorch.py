#!/usr/bin/env python3
"""
Entraînement PyTorch pour Prédiction de Collision d'Astéroïdes
Script autonome pour développer et tester le modèle ML en dehors de Docker
"""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Configuration du device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Utilisation du device: {device}")

class AsteroidCollisionNet(nn.Module):
    """Réseau de neurones PyTorch pour prédiction de collision d'astéroïdes"""
    
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

class AsteroidDataGenerator:
    """Générateur de données d'astéroïdes réalistes"""
    
    def __init__(self, seed=42):
        np.random.seed(seed)
        
    def generate_data(self, n_samples=10000):
        """Génère des données synthétiques d'astéroïdes"""
        print(f"📊 Génération de {n_samples} données d'astéroïdes...")
        
        data = {
            'diameter_km': np.random.lognormal(0, 1.5, n_samples),
            'velocity_km_s': np.random.normal(20, 8, n_samples),
            'distance_earth_au': np.random.exponential(2.0, n_samples),
            'mass_kg': np.random.lognormal(15, 2.5, n_samples),
            'orbital_eccentricity': np.random.beta(2, 5, n_samples),
            'inclination_deg': np.random.normal(10, 20, n_samples),
            'approach_angle': np.random.uniform(0, 2*np.pi, n_samples),
            'albedo': np.random.beta(2, 8, n_samples),
            'rotation_period_h': np.random.lognormal(1, 1, n_samples),
            'surface_temp_k': np.random.normal(200, 60, n_samples)
        }
        
        X = pd.DataFrame(data)
        collision_risk = self._calculate_collision_risk(X)
        threshold = np.percentile(collision_risk, 85)
        y = (collision_risk > threshold).astype(int)
        
        collisions = np.sum(y)
        print(f"✓ Dataset généré: {len(X)} astéroïdes")
        print(f"  - {collisions} collisions potentielles ({collisions/len(y)*100:.1f}%)")
        print(f"  - {len(y)-collisions} astéroïdes sûrs ({(1-collisions/len(y))*100:.1f}%)")
        
        return X, y
    
    def _calculate_collision_risk(self, X):
        """Calcule la probabilité de collision basée sur critères physiques"""
        
        size_risk = np.clip((X['diameter_km'] - 0.5) / 2.0, 0, 1)
        velocity_risk = np.clip((X['velocity_km_s'] - 15) / 20, 0, 1)
        distance_risk = np.clip(2.0 / (X['distance_earth_au'] + 0.1), 0, 1)
        orbit_risk = X['orbital_eccentricity']
        angle_risk = np.abs(np.sin(X['approach_angle']))
        
        total_risk = (
            0.3 * size_risk +
            0.25 * velocity_risk +
            0.25 * distance_risk +
            0.1 * orbit_risk +
            0.1 * angle_risk
        )
        
        noise = np.random.normal(0, 0.05, len(X))
        return np.clip(total_risk + noise, 0, 1)

class AsteroidTrainer:
    """Classe pour entraîner le modèle PyTorch"""
    
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.scaler = StandardScaler()
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        
    def prepare_data(self, X, y, test_size=0.2, batch_size=64):
        """Prépare les données pour l'entraînement"""
        
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train.values).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val_scaled).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val.values).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test.values).to(self.device)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        print(f"✓ Données préparées:")
        print(f"  - Train: {len(X_train)} échantillons")
        print(f"  - Validation: {len(X_val)} échantillons") 
        print(f"  - Test: {len(X_test)} échantillons")
        
        return X_test, y_test
    
    def train(self, epochs=100, lr=0.001, patience=10):
        """Entraîne le modèle avec early stopping"""
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"🚀 Début de l'entraînement ({epochs} epochs max)...")
        
        for epoch in range(epochs):
            # Phase d'entraînement
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in self.train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                predicted = (outputs > 0.5).float()
                train_correct += (predicted == batch_y).sum().item()
                train_total += batch_y.size(0)
            
            # Phase de validation
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in self.val_loader:
                    outputs = self.model(batch_X).squeeze()
                    loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    predicted = (outputs > 0.5).float()
                    val_correct += (predicted == batch_y).sum().item()
                    val_total += batch_y.size(0)
            
            train_loss_avg = train_loss / len(self.train_loader)
            val_loss_avg = val_loss / len(self.val_loader)
            train_acc = train_correct / train_total
            val_acc = val_correct / val_total
            
            self.history['train_loss'].append(train_loss_avg)
            self.history['val_loss'].append(val_loss_avg)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
            
            scheduler.step(val_loss_avg)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}:")
                print(f"  Train Loss: {train_loss_avg:.4f}, Acc: {train_acc:.4f}")
                print(f"  Val Loss: {val_loss_avg:.4f}, Acc: {val_acc:.4f}")
            
            if patience_counter >= patience:
                print(f"🛑 Early stopping à l'epoch {epoch+1}")
                break
        
        self.model.load_state_dict(torch.load('best_model.pth'))
        print("✅ Entraînement terminé !")
    
    def evaluate(self, X_test, y_test):
        """Évalue le modèle sur les données de test"""
        
        self.model.eval()
        all_predictions = []
        all_probabilities = []
        all_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in self.test_loader:
                outputs = self.model(batch_X).squeeze()
                predictions = (outputs > 0.5).float()
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(outputs.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
        
        accuracy = accuracy_score(all_targets, all_predictions)
        auc_score = roc_auc_score(all_targets, all_probabilities)
        
        print("\n📊 RÉSULTATS FINAUX:")
        print("=" * 50)
        print(f"🎯 Accuracy: {accuracy:.4f}")
        print(f"📈 AUC-ROC: {auc_score:.4f}")
        print("\n📋 Rapport de Classification:")
        print(classification_report(all_targets, all_predictions))
        
        return accuracy, auc_score
    
    def save_model_for_production(self, model_path='models/asteroid_collision_model.pth', scaler_path='models/scaler.pkl'):
        """Sauvegarde le modèle et le scaler pour utilisation en production"""
        
        os.makedirs('models', exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_size': 10,
                'hidden_sizes': [64, 32, 16],
                'dropout_rate': 0.3
            }
        }, model_path)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"✅ Modèle sauvegardé: {model_path}")
        print(f"✅ Scaler sauvegardé: {scaler_path}")

def main():
    """Fonction principale d'entraînement"""
    
    print("🌌 ENTRAÎNEMENT PYTORCH - PRÉDICTION COLLISION ASTÉROÏDES")
    print("=" * 60)
    
    # Génération des données
    generator = AsteroidDataGenerator()
    X, y = generator.generate_data(n_samples=15000)
    
    # Création du modèle
    model = AsteroidCollisionNet(input_size=X.shape[1])
    print(f"🧠 Modèle créé: {sum(p.numel() for p in model.parameters())} paramètres")
    
    # Entraînement
    trainer = AsteroidTrainer(model, device)
    X_test, y_test = trainer.prepare_data(X, y, batch_size=128)
    trainer.train(epochs=150, lr=0.001, patience=15)
    
    # Évaluation
    accuracy, auc = trainer.evaluate(X_test, y_test)
    
    # Sauvegarde pour production
    trainer.save_model_for_production()
    
    print("\n🎉 ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
    print(f"Final Accuracy: {accuracy:.4f}")
    print(f"Final AUC: {auc:.4f}")
    print("\n💡 Pour utiliser dans Docker:")
    print("   1. Copier best_model.pth vers ../app/models/asteroid_collision_model.pth")
    print("   2. Copier models/scaler.pkl vers ../app/models/scaler.pkl")
    print("   3. Lancer docker-compose up --build")

if __name__ == "__main__":
    main()