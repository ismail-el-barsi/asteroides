#!/usr/bin/env python3
"""
Machine Learning Models for Asteroid Collision Prediction
Implements various ML/DL algorithms for predicting asteroid collision probabilities
"""

import logging
import os
import pickle
import sys
from datetime import datetime, timedelta

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.ensemble import (GradientBoostingClassifier,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, mean_squared_error,
                             precision_recall_curve, r2_score,
                             regression_report_if_exists, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import (GridSearchCV, cross_val_score,
                                     train_test_split)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from tensorflow import keras
from tensorflow.keras import callbacks, layers, optimizers

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AsteroidMLPredictor:
    def __init__(self, data_path="/tmp/asteroid_data"):
        """Initialize the ML predictor"""
        self.data_path = data_path
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.results = {}
        
        # Create output directories
        os.makedirs(f"{data_path}/models", exist_ok=True)
        os.makedirs(f"{data_path}/results", exist_ok=True)
        os.makedirs(f"{data_path}/plots", exist_ok=True)
        
        logger.info("ML Predictor initialized")
    
    def load_processed_data(self):
        """Load processed data from Spark processing"""
        try:
            # Try to load from different possible locations
            data_files = [
                f"{self.data_path}/ml_training_data/ml_training_data.parquet",
                f"{self.data_path}/asteroid_data/processed/processed_asteroids.parquet",
                f"{self.data_path}/asteroid_data/processed/*.parquet"
            ]
            
            for file_path in data_files:
                if os.path.exists(file_path) or "*" in file_path:
                    try:
                        if "*" in file_path:
                            # Load all parquet files in directory
                            import glob
                            files = glob.glob(file_path)
                            if files:
                                dfs = []
                                for f in files:
                                    dfs.append(pd.read_parquet(f))
                                self.data = pd.concat(dfs, ignore_index=True)
                        else:
                            self.data = pd.read_parquet(file_path)
                        
                        logger.info(f"Loaded {len(self.data)} records from {file_path}")
                        return True
                    except Exception as e:
                        logger.warning(f"Failed to load from {file_path}: {e}")
                        continue
            
            # If no parquet files, generate synthetic data
            logger.warning("No processed data found, generating synthetic data for demonstration")
            self.generate_synthetic_data()
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def generate_synthetic_data(self, n_samples=10000):
        """Generate synthetic asteroid data for ML training"""
        logger.info("Generating synthetic asteroid data...")
        
        np.random.seed(42)
        
        # Generate features
        size = np.random.lognormal(0, 1, n_samples)
        mass = size ** 3 * np.random.uniform(2000, 5000, n_samples)
        density = mass / ((4/3) * np.pi * (size * 500) ** 3)
        
        # Position features (distance from sun)
        distance_from_sun = np.random.uniform(1.5e8, 5.0e8, n_samples)  # km
        distance_from_earth = np.random.uniform(1e6, 2e8, n_samples)  # km
        
        # Velocity features
        velocity_magnitude = np.random.uniform(10, 50, n_samples)  # km/s
        relative_velocity = np.random.uniform(5, 40, n_samples)  # km/s
        
        # Orbital features
        orbital_energy = np.random.uniform(-50, 10, n_samples)
        angular_momentum_magnitude = np.random.uniform(1e12, 1e15, n_samples)
        eccentricity = np.random.uniform(0.05, 0.3, n_samples)
        approach_angle = np.random.uniform(0, np.pi, n_samples)
        
        # Time features
        hour_of_day = np.random.randint(0, 24, n_samples)
        day_of_year = np.random.randint(1, 366, n_samples)
        
        # Create enhanced collision probability based on physical relationships
        earth_proximity_factor = np.exp(-distance_from_earth / 1e7)
        size_factor = np.minimum(size / 10.0, 1.0)
        velocity_factor = 1.0 / (1.0 + relative_velocity / 30.0)
        
        enhanced_collision_probability = (
            earth_proximity_factor * size_factor * velocity_factor * 0.001 +
            np.random.normal(0, 0.0001, n_samples)
        )
        enhanced_collision_probability = np.maximum(0, enhanced_collision_probability)
        
        # Create risk levels
        risk_level = np.where(enhanced_collision_probability > 0.00001, 'HIGH',
                            np.where(enhanced_collision_probability > 0.000001, 'MEDIUM', 'LOW'))
        
        # Create binary target for classification
        potentially_hazardous = (enhanced_collision_probability > 0.000001).astype(int)
        
        # Create DataFrame
        self.data = pd.DataFrame({
            'size': size,
            'mass': mass,
            'density': density,
            'velocity_magnitude': velocity_magnitude,
            'distance_from_sun': distance_from_sun,
            'distance_from_earth': distance_from_earth,
            'orbital_energy': orbital_energy,
            'angular_momentum_magnitude': angular_momentum_magnitude,
            'eccentricity': eccentricity,
            'relative_velocity': relative_velocity,
            'approach_angle': approach_angle,
            'hour_of_day': hour_of_day,
            'day_of_year': day_of_year,
            'enhanced_collision_probability': enhanced_collision_probability,
            'risk_level': risk_level,
            'potentially_hazardous': potentially_hazardous
        })
        
        logger.info(f"Generated {len(self.data)} synthetic asteroid records")
    
    def prepare_features(self):
        """Prepare features for ML training"""
        logger.info("Preparing features for ML training...")
        
        # Define feature columns
        self.feature_names = [
            'size', 'mass', 'density', 'velocity_magnitude', 
            'distance_from_sun', 'distance_from_earth', 'orbital_energy',
            'angular_momentum_magnitude', 'eccentricity', 'relative_velocity',
            'approach_angle', 'hour_of_day', 'day_of_year'
        ]
        
        # Ensure all features exist
        for feature in self.feature_names:
            if feature not in self.data.columns:
                logger.warning(f"Feature {feature} not found, filling with zeros")
                self.data[feature] = 0.0
        
        # Prepare feature matrix
        self.X = self.data[self.feature_names].fillna(0)
        
        # Prepare targets
        self.y_classification = self.data['potentially_hazardous'].fillna(0).astype(int)
        self.y_regression = self.data['enhanced_collision_probability'].fillna(0)
        
        # Feature scaling
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        logger.info(f"Prepared features: {self.X.shape}")
        logger.info(f"Feature names: {self.feature_names}")
    
    def split_data(self, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        # For classification
        self.X_train_cls, self.X_test_cls, self.y_train_cls, self.y_test_cls = train_test_split(
            self.X_scaled, self.y_classification, test_size=test_size, 
            random_state=random_state, stratify=self.y_classification
        )
        
        # For regression
        self.X_train_reg, self.X_test_reg, self.y_train_reg, self.y_test_reg = train_test_split(
            self.X_scaled, self.y_regression, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"Data split - Training: {len(self.X_train_cls)}, Testing: {len(self.X_test_cls)}")
    
    def train_classical_ml_models(self):
        """Train classical ML models"""
        logger.info("Training classical ML models...")
        
        # Classification models
        classification_models = {
            'RandomForest_Classifier': RandomForestClassifier(n_estimators=100, random_state=42),
            'GradientBoosting_Classifier': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM_Classifier': SVC(probability=True, random_state=42),
            'MLP_Classifier': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000)
        }
        
        # Regression models
        regression_models = {
            'RandomForest_Regressor': RandomForestRegressor(n_estimators=100, random_state=42),
            'LinearRegression': LinearRegression(),
            'SVR': SVR(),
            'MLP_Regressor': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000)
        }
        
        # Train classification models
        for name, model in classification_models.items():
            logger.info(f"Training {name}...")
            try:
                model.fit(self.X_train_cls, self.y_train_cls)
                y_pred = model.predict(self.X_test_cls)
                y_pred_proba = model.predict_proba(self.X_test_cls)[:, 1]
                
                accuracy = accuracy_score(self.y_test_cls, y_pred)
                auc = roc_auc_score(self.y_test_cls, y_pred_proba)
                
                self.models[name] = model
                self.results[name] = {
                    'type': 'classification',
                    'accuracy': accuracy,
                    'auc': auc,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                logger.info(f"{name} - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
        
        # Train regression models
        for name, model in regression_models.items():
            logger.info(f"Training {name}...")
            try:
                model.fit(self.X_train_reg, self.y_train_reg)
                y_pred = model.predict(self.X_test_reg)
                
                mse = mean_squared_error(self.y_test_reg, y_pred)
                r2 = r2_score(self.y_test_reg, y_pred)
                
                self.models[name] = model
                self.results[name] = {
                    'type': 'regression',
                    'mse': mse,
                    'rmse': np.sqrt(mse),
                    'r2': r2,
                    'predictions': y_pred
                }
                
                logger.info(f"{name} - MSE: {mse:.6f}, R2: {r2:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
    
    def build_deep_learning_model(self, model_type='classification'):
        """Build deep learning model using TensorFlow/Keras"""
        if model_type == 'classification':
            model = keras.Sequential([
                layers.Dense(128, activation='relu', input_shape=(len(self.feature_names),)),
                layers.Dropout(0.3),
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(32, activation='relu'),
                layers.Dense(1, activation='sigmoid')
            ])
            
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy', 'precision', 'recall']
            )
        else:  # regression
            model = keras.Sequential([
                layers.Dense(128, activation='relu', input_shape=(len(self.feature_names),)),
                layers.Dropout(0.3),
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(32, activation='relu'),
                layers.Dense(1)
            ])
            
            model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
        
        return model
    
    def train_deep_learning_models(self):
        """Train deep learning models"""
        logger.info("Training deep learning models...")
        
        # Early stopping callback
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        
        # Classification model
        try:
            logger.info("Training DL Classification model...")
            dl_classifier = self.build_deep_learning_model('classification')
            
            history_cls = dl_classifier.fit(
                self.X_train_cls, self.y_train_cls,
                validation_split=0.2,
                epochs=100,
                batch_size=32,
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Evaluate
            y_pred_proba = dl_classifier.predict(self.X_test_cls).flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            accuracy = accuracy_score(self.y_test_cls, y_pred)
            auc = roc_auc_score(self.y_test_cls, y_pred_proba)
            
            self.models['DeepLearning_Classifier'] = dl_classifier
            self.results['DeepLearning_Classifier'] = {
                'type': 'classification',
                'accuracy': accuracy,
                'auc': auc,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'history': history_cls.history
            }
            
            logger.info(f"DL Classifier - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
            
        except Exception as e:
            logger.error(f"Error training DL classifier: {e}")
        
        # Regression model
        try:
            logger.info("Training DL Regression model...")
            dl_regressor = self.build_deep_learning_model('regression')
            
            history_reg = dl_regressor.fit(
                self.X_train_reg, self.y_train_reg,
                validation_split=0.2,
                epochs=100,
                batch_size=32,
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Evaluate
            y_pred = dl_regressor.predict(self.X_test_reg).flatten()
            
            mse = mean_squared_error(self.y_test_reg, y_pred)
            r2 = r2_score(self.y_test_reg, y_pred)
            
            self.models['DeepLearning_Regressor'] = dl_regressor
            self.results['DeepLearning_Regressor'] = {
                'type': 'regression',
                'mse': mse,
                'rmse': np.sqrt(mse),
                'r2': r2,
                'predictions': y_pred,
                'history': history_reg.history
            }
            
            logger.info(f"DL Regressor - MSE: {mse:.6f}, R2: {r2:.4f}")
            
        except Exception as e:
            logger.error(f"Error training DL regressor: {e}")
    
    def evaluate_models(self):
        """Evaluate all trained models"""
        logger.info("Evaluating all models...")
        
        classification_results = []
        regression_results = []
        
        for name, result in self.results.items():
            if result['type'] == 'classification':
                classification_results.append({
                    'Model': name,
                    'Accuracy': result['accuracy'],
                    'AUC': result['auc']
                })
            else:
                regression_results.append({
                    'Model': name,
                    'MSE': result['mse'],
                    'RMSE': result['rmse'],
                    'R2': result['r2']
                })
        
        # Print results
        if classification_results:
            print("\n=== CLASSIFICATION RESULTS ===")
            cls_df = pd.DataFrame(classification_results)
            cls_df = cls_df.sort_values('AUC', ascending=False)
            print(cls_df.to_string(index=False))
        
        if regression_results:
            print("\n=== REGRESSION RESULTS ===")
            reg_df = pd.DataFrame(regression_results)
            reg_df = reg_df.sort_values('R2', ascending=False)
            print(reg_df.to_string(index=False))
        
        # Find best models
        if classification_results:
            best_classifier = max(self.results.items(), 
                                key=lambda x: x[1]['auc'] if x[1]['type'] == 'classification' else 0)
            logger.info(f"Best classifier: {best_classifier[0]} (AUC: {best_classifier[1]['auc']:.4f})")
        
        if regression_results:
            best_regressor = max(self.results.items(), 
                               key=lambda x: x[1]['r2'] if x[1]['type'] == 'regression' else -float('inf'))
            logger.info(f"Best regressor: {best_regressor[0]} (R2: {best_regressor[1]['r2']:.4f})")
    
    def save_models(self):
        """Save trained models"""
        logger.info("Saving trained models...")
        
        for name, model in self.models.items():
            try:
                if 'DeepLearning' in name:
                    # Save Keras model
                    model.save(f"{self.data_path}/models/{name}.h5")
                else:
                    # Save scikit-learn model
                    joblib.dump(model, f"{self.data_path}/models/{name}.pkl")
                
                logger.info(f"Saved model: {name}")
            except Exception as e:
                logger.error(f"Error saving model {name}: {e}")
        
        # Save scaler
        joblib.dump(self.scaler, f"{self.data_path}/models/scaler.pkl")
        
        # Save feature names
        with open(f"{self.data_path}/models/feature_names.txt", 'w') as f:
            f.write('\n'.join(self.feature_names))
    
    def create_visualizations(self):
        """Create visualization plots"""
        logger.info("Creating visualizations...")
        
        try:
            # Feature importance plot (for tree-based models)
            if 'RandomForest_Classifier' in self.models:
                rf_model = self.models['RandomForest_Classifier']
                feature_importance = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': rf_model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                plt.figure(figsize=(10, 6))
                sns.barplot(data=feature_importance, x='importance', y='feature')
                plt.title('Feature Importance (Random Forest Classifier)')
                plt.tight_layout()
                plt.savefig(f"{self.data_path}/plots/feature_importance.png")
                plt.close()
            
            # ROC curves for classification models
            plt.figure(figsize=(10, 8))
            for name, result in self.results.items():
                if result['type'] == 'classification':
                    fpr, tpr, _ = roc_curve(self.y_test_cls, result['probabilities'])
                    plt.plot(fpr, tpr, label=f"{name} (AUC: {result['auc']:.3f})")
            
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves Comparison')
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{self.data_path}/plots/roc_curves.png")
            plt.close()
            
            # Prediction vs Actual for regression models
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.flatten()
            
            reg_models = [(name, result) for name, result in self.results.items() 
                         if result['type'] == 'regression']
            
            for i, (name, result) in enumerate(reg_models[:4]):
                ax = axes[i]
                ax.scatter(self.y_test_reg, result['predictions'], alpha=0.6)
                ax.plot([self.y_test_reg.min(), self.y_test_reg.max()], 
                       [self.y_test_reg.min(), self.y_test_reg.max()], 'r--', lw=2)
                ax.set_xlabel('Actual')
                ax.set_ylabel('Predicted')
                ax.set_title(f'{name} (R²: {result["r2"]:.3f})')
                ax.grid(True)
            
            plt.tight_layout()
            plt.savefig(f"{self.data_path}/plots/regression_predictions.png")
            plt.close()
            
            logger.info("Visualizations saved successfully")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
    
    def predict_new_asteroid(self, asteroid_features):
        """Predict collision probability for a new asteroid"""
        # Use the best performing models
        best_classifier = max(
            [(name, result) for name, result in self.results.items() 
             if result['type'] == 'classification'],
            key=lambda x: x[1]['auc']
        )[0]
        
        best_regressor = max(
            [(name, result) for name, result in self.results.items() 
             if result['type'] == 'regression'],
            key=lambda x: x[1]['r2']
        )[0]
        
        # Prepare features
        features_array = np.array([asteroid_features[name] for name in self.feature_names]).reshape(1, -1)
        features_scaled = self.scaler.transform(features_array)
        
        # Make predictions
        hazard_probability = self.models[best_classifier].predict_proba(features_scaled)[0, 1]
        collision_probability = self.models[best_regressor].predict(features_scaled)[0]
        
        return {
            'hazard_probability': hazard_probability,
            'collision_probability': collision_probability,
            'risk_level': 'HIGH' if collision_probability > 0.00001 else 
                         'MEDIUM' if collision_probability > 0.000001 else 'LOW'
        }
    
    def run_ml_pipeline(self):
        """Run the complete ML pipeline"""
        logger.info("Starting ML pipeline...")
        
        # Load and prepare data
        if not self.load_processed_data():
            logger.error("Failed to load data. Exiting...")
            return
        
        self.prepare_features()
        self.split_data()
        
        # Train models
        self.train_classical_ml_models()
        self.train_deep_learning_models()
        
        # Evaluate and visualize
        self.evaluate_models()
        self.create_visualizations()
        
        # Save models
        self.save_models()
        
        logger.info("ML pipeline completed successfully")

def main():
    """Main function to run the ML pipeline"""
    predictor = AsteroidMLPredictor()
    predictor.run_ml_pipeline()
    
    # Example prediction
    example_asteroid = {
        'size': 2.5,
        'mass': 1e13,
        'density': 3000,
        'velocity_magnitude': 25.0,
        'distance_from_sun': 2.5e8,
        'distance_from_earth': 1e7,
        'orbital_energy': -10.0,
        'angular_momentum_magnitude': 1e13,
        'eccentricity': 0.15,
        'relative_velocity': 20.0,
        'approach_angle': 0.5,
        'hour_of_day': 12,
        'day_of_year': 180
    }
    
    prediction = predictor.predict_new_asteroid(example_asteroid)
    print(f"\n=== EXAMPLE PREDICTION ===")
    print(f"Hazard Probability: {prediction['hazard_probability']:.6f}")
    print(f"Collision Probability: {prediction['collision_probability']:.8f}")
    print(f"Risk Level: {prediction['risk_level']}")

if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()