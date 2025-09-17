#!/usr/bin/env python3
"""
Spark Data Processing for Asteroid Collision Prediction
Processes asteroid and planet data from HDFS, cleans data, and calculates trajectories
"""

import logging
import math
import os
import sys
from datetime import datetime, timedelta

import numpy as np
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import (BinaryClassificationEvaluator,
                                   RegressionEvaluator)
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AsteroidDataProcessor:
    def __init__(self):
        """Initialize Spark session and configuration"""
        self.spark = SparkSession.builder \
            .appName("AsteroidCollisionPrediction") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "64MB") \
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("WARN")
        
        # Physical constants
        self.AU = 149597870.7  # 1 AU in km
        self.EARTH_RADIUS = 6371  # km
        self.EARTH_SOI = 0.01 * self.AU  # Earth's sphere of influence (simplified)
        self.G = 6.67430e-11  # Gravitational constant
        self.SOLAR_MASS = 1.989e30  # kg
        
        logger.info("Spark session initialized successfully")
    
    def define_schemas(self):
        """Define schemas for different data types"""
        self.asteroid_schema = StructType([
            StructField("id", StringType(), True),
            StructField("timestamp", StringType(), True),
            StructField("position", StructType([
                StructField("x", DoubleType(), True),
                StructField("y", DoubleType(), True),
                StructField("z", DoubleType(), True)
            ]), True),
            StructField("velocity", StructType([
                StructField("vx", DoubleType(), True),
                StructField("vy", DoubleType(), True),
                StructField("vz", DoubleType(), True)
            ]), True),
            StructField("size", DoubleType(), True),
            StructField("mass", DoubleType(), True),
            StructField("density", DoubleType(), True),
            StructField("classification", StringType(), True),
            StructField("orbital_period", DoubleType(), True),
            StructField("eccentricity", DoubleType(), True),
            StructField("potentially_hazardous", BooleanType(), True),
            StructField("collision_probability", DoubleType(), True),
            StructField("processed_timestamp", StringType(), True)
        ])
        
        self.planet_schema = StructType([
            StructField("planet", StringType(), True),
            StructField("timestamp", StringType(), True),
            StructField("position", StructType([
                StructField("x", DoubleType(), True),
                StructField("y", DoubleType(), True),
                StructField("z", DoubleType(), True)
            ]), True),
            StructField("velocity", StructType([
                StructField("vx", DoubleType(), True),
                StructField("vy", DoubleType(), True),
                StructField("vz", DoubleType(), True)
            ]), True),
            StructField("mass", DoubleType(), True),
            StructField("orbital_period", DoubleType(), True),
            StructField("processed_timestamp", StringType(), True)
        ])
    
    def load_data_from_hdfs(self, path_pattern: str, schema: StructType):
        """Load data from HDFS with error handling"""
        try:
            if self.data_exists_in_hdfs(path_pattern):
                df = self.spark.read.schema(schema).parquet(path_pattern)
                logger.info(f"Loaded {df.count()} records from {path_pattern}")
                return df
            else:
                logger.warning(f"No data found at {path_pattern}")
                return self.spark.createDataFrame([], schema)
        except Exception as e:
            logger.error(f"Error loading data from {path_pattern}: {e}")
            return self.spark.createDataFrame([], schema)
    
    def data_exists_in_hdfs(self, path: str) -> bool:
        """Check if data exists in HDFS path"""
        try:
            # Try to list files in the path
            files = self.spark.sparkContext._jvm.org.apache.hadoop.fs.FileSystem.get(
                self.spark.sparkContext._jsc.hadoopConfiguration()
            ).listStatus(
                self.spark.sparkContext._jvm.org.apache.hadoop.fs.Path(path)
            )
            return len(files) > 0
        except:
            return False
    
    def clean_asteroid_data(self, df):
        """Clean and validate asteroid data"""
        logger.info("Cleaning asteroid data...")
        
        # Remove records with null essential fields
        df_clean = df.filter(
            col("position").isNotNull() & 
            col("velocity").isNotNull() & 
            col("size").isNotNull() & 
            col("mass").isNotNull()
        )
        
        # Extract position and velocity components
        df_clean = df_clean.withColumn("pos_x", col("position.x")) \
                           .withColumn("pos_y", col("position.y")) \
                           .withColumn("pos_z", col("position.z")) \
                           .withColumn("vel_x", col("velocity.vx")) \
                           .withColumn("vel_y", col("velocity.vy")) \
                           .withColumn("vel_z", col("velocity.vz"))
        
        # Calculate derived features
        df_clean = self.add_calculated_features(df_clean)
        
        # Remove outliers
        df_clean = self.remove_outliers(df_clean)
        
        logger.info(f"Cleaned data: {df_clean.count()} records remaining")
        return df_clean
    
    def add_calculated_features(self, df):
        """Add calculated features for analysis"""
        # Distance from origin (Sun)
        df = df.withColumn("distance_from_sun", 
                          sqrt(col("pos_x")**2 + col("pos_y")**2 + col("pos_z")**2))
        
        # Velocity magnitude
        df = df.withColumn("velocity_magnitude", 
                          sqrt(col("vel_x")**2 + col("vel_y")**2 + col("vel_z")**2))
        
        # Distance from Earth (assuming Earth at 1 AU on x-axis for simplification)
        df = df.withColumn("distance_from_earth", 
                          sqrt((col("pos_x") - lit(self.AU))**2 + 
                               col("pos_y")**2 + col("pos_z")**2))
        
        # Orbital energy (specific energy)
        df = df.withColumn("orbital_energy", 
                          (col("velocity_magnitude")**2 / 2) - 
                          (lit(1.327e11) / col("distance_from_sun")))
        
        # Angular momentum magnitude
        df = df.withColumn("angular_momentum_x", 
                          col("pos_y") * col("vel_z") - col("pos_z") * col("vel_y"))
        df = df.withColumn("angular_momentum_y", 
                          col("pos_z") * col("vel_x") - col("pos_x") * col("vel_z"))
        df = df.withColumn("angular_momentum_z", 
                          col("pos_x") * col("vel_y") - col("pos_y") * col("vel_x"))
        df = df.withColumn("angular_momentum_magnitude", 
                          sqrt(col("angular_momentum_x")**2 + 
                               col("angular_momentum_y")**2 + 
                               col("angular_momentum_z")**2))
        
        # Semi-major axis (from orbital energy)
        df = df.withColumn("semi_major_axis", 
                          when(col("orbital_energy") < 0, 
                               -lit(1.327e11) / (2 * col("orbital_energy")))
                          .otherwise(lit(None)))
        
        # Time-based features
        df = df.withColumn("timestamp_unix", 
                          unix_timestamp(col("timestamp"), "yyyy-MM-dd'T'HH:mm:ss.SSSSSS"))
        df = df.withColumn("hour_of_day", 
                          hour(to_timestamp(col("timestamp"), "yyyy-MM-dd'T'HH:mm:ss.SSSSSS")))
        df = df.withColumn("day_of_year", 
                          dayofyear(to_timestamp(col("timestamp"), "yyyy-MM-dd'T'HH:mm:ss.SSSSSS")))
        
        return df
    
    def remove_outliers(self, df):
        """Remove statistical outliers from the data"""
        # Define outlier bounds for key numerical features
        numerical_features = ["size", "mass", "velocity_magnitude", "distance_from_sun"]
        
        for feature in numerical_features:
            # Calculate quartiles
            quantiles = df.stat.approxQuantile(feature, [0.25, 0.75], 0.05)
            if len(quantiles) == 2:
                q1, q3 = quantiles
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                # Remove outliers
                df = df.filter((col(feature) >= lower_bound) & (col(feature) <= upper_bound))
        
        return df
    
    def calculate_collision_trajectories(self, asteroid_df, planet_df):
        """Calculate asteroid trajectories and collision probabilities"""
        logger.info("Calculating collision trajectories...")
        
        # Get latest Earth position
        earth_df = planet_df.filter(col("planet") == "Earth") \
                           .orderBy(col("timestamp").desc()) \
                           .limit(1)
        
        if earth_df.count() == 0:
            logger.warning("No Earth position data found, using default position")
            earth_pos = {"x": self.AU, "y": 0, "z": 0}
        else:
            earth_row = earth_df.collect()[0]
            earth_pos = {
                "x": earth_row["position"]["x"],
                "y": earth_row["position"]["y"], 
                "z": earth_row["position"]["z"]
            }
        
        # Calculate enhanced collision probabilities
        asteroid_df = self.calculate_enhanced_collision_probability(asteroid_df, earth_pos)
        
        # Predict future positions
        asteroid_df = self.predict_future_positions(asteroid_df, time_steps=[1, 7, 30, 365])
        
        return asteroid_df
    
    def calculate_enhanced_collision_probability(self, df, earth_pos):
        """Calculate enhanced collision probability using multiple factors"""
        
        # Distance to Earth
        df = df.withColumn("earth_distance", 
                          sqrt((col("pos_x") - lit(earth_pos["x"]))**2 + 
                               (col("pos_y") - lit(earth_pos["y"]))**2 + 
                               (col("pos_z") - lit(earth_pos["z"]))**2))
        
        # Relative velocity vector to Earth
        # Simplified: assuming Earth velocity is approximately orbital velocity
        earth_vel_magnitude = 2 * math.pi * self.AU / (365.25 * 24 * 3600)  # km/s
        
        df = df.withColumn("relative_velocity", 
                          sqrt((col("vel_x") - lit(earth_vel_magnitude))**2 + 
                               col("vel_y")**2 + col("vel_z")**2))
        
        # Calculate approach angle
        df = df.withColumn("approach_angle", 
                          abs(atan2(col("pos_y") - lit(earth_pos["y"]), 
                                   col("pos_x") - lit(earth_pos["x"]))))
        
        # Enhanced collision probability calculation
        df = df.withColumn("size_factor", 
                          least(col("size") / 10.0, lit(1.0)))
        
        df = df.withColumn("distance_factor", 
                          when(col("earth_distance") > lit(self.EARTH_SOI), lit(0.0))
                          .otherwise(greatest(lit(0.0), 
                                            1.0 - (col("earth_distance") / lit(self.EARTH_SOI)))))
        
        df = df.withColumn("velocity_factor", 
                          1.0 / (1.0 + col("relative_velocity") / 30.0))  # Normalize by typical impact velocity
        
        df = df.withColumn("enhanced_collision_probability", 
                          col("size_factor") * col("distance_factor") * 
                          col("velocity_factor") * lit(0.001))
        
        # Classification: High risk if probability > 0.001%
        df = df.withColumn("risk_level", 
                          when(col("enhanced_collision_probability") > 0.00001, "HIGH")
                          .when(col("enhanced_collision_probability") > 0.000001, "MEDIUM")
                          .otherwise("LOW"))
        
        return df
    
    def predict_future_positions(self, df, time_steps):
        """Predict future asteroid positions using simplified orbital mechanics"""
        for days in time_steps:
            time_delta = days * 24 * 3600  # Convert to seconds
            
            # Simple numerical integration (Euler method)
            # In practice, this would use more sophisticated orbital mechanics
            
            # Calculate gravitational acceleration
            df = df.withColumn(f"acc_magnitude_{days}d", 
                              lit(1.327e11) / (col("distance_from_sun")**2))
            
            df = df.withColumn(f"acc_x_{days}d", 
                              -col(f"acc_magnitude_{days}d") * col("pos_x") / col("distance_from_sun"))
            df = df.withColumn(f"acc_y_{days}d", 
                              -col(f"acc_magnitude_{days}d") * col("pos_y") / col("distance_from_sun"))
            df = df.withColumn(f"acc_z_{days}d", 
                              -col(f"acc_magnitude_{days}d") * col("pos_z") / col("distance_from_sun"))
            
            # Update velocity
            df = df.withColumn(f"vel_x_{days}d", 
                              col("vel_x") + col(f"acc_x_{days}d") * lit(time_delta))
            df = df.withColumn(f"vel_y_{days}d", 
                              col("vel_y") + col(f"acc_y_{days}d") * lit(time_delta))
            df = df.withColumn(f"vel_z_{days}d", 
                              col("vel_z") + col(f"acc_z_{days}d") * lit(time_delta))
            
            # Update position
            df = df.withColumn(f"pos_x_{days}d", 
                              col("pos_x") + col(f"vel_x_{days}d") * lit(time_delta))
            df = df.withColumn(f"pos_y_{days}d", 
                              col("pos_y") + col(f"vel_y_{days}d") * lit(time_delta))
            df = df.withColumn(f"pos_z_{days}d", 
                              col("pos_z") + col(f"vel_z_{days}d") * lit(time_delta))
            
            # Calculate future Earth distance
            df = df.withColumn(f"earth_distance_{days}d", 
                              sqrt((col(f"pos_x_{days}d") - lit(self.AU))**2 + 
                                   col(f"pos_y_{days}d")**2 + 
                                   col(f"pos_z_{days}d")**2))
        
        return df
    
    def prepare_ml_features(self, df):
        """Prepare features for machine learning"""
        logger.info("Preparing ML features...")
        
        # Select features for ML
        feature_columns = [
            "size", "mass", "density", "velocity_magnitude", 
            "distance_from_sun", "distance_from_earth", "orbital_energy",
            "angular_momentum_magnitude", "eccentricity", "relative_velocity",
            "approach_angle", "hour_of_day", "day_of_year"
        ]
        
        # Handle missing values
        for col_name in feature_columns:
            df = df.fillna({col_name: 0.0})
        
        # Create feature vector
        assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
        df = assembler.transform(df)
        
        # Scale features
        scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
        scaler_model = scaler.fit(df)
        df = scaler_model.transform(df)
        
        return df, feature_columns
    
    def save_processed_data(self, df, output_path):
        """Save processed data to HDFS"""
        try:
            df.write.mode("overwrite").parquet(output_path)
            logger.info(f"Saved processed data to {output_path}")
        except Exception as e:
            logger.error(f"Error saving data to {output_path}: {e}")
    
    def generate_summary_statistics(self, df):
        """Generate summary statistics for the processed data"""
        logger.info("Generating summary statistics...")
        
        # Overall statistics
        total_asteroids = df.count()
        high_risk_asteroids = df.filter(col("risk_level") == "HIGH").count()
        medium_risk_asteroids = df.filter(col("risk_level") == "MEDIUM").count()
        
        # Classification statistics
        classification_stats = df.groupBy("classification").count().collect()
        
        # Size distribution
        size_stats = df.select(
            avg("size").alias("avg_size"),
            stddev("size").alias("stddev_size"),
            min("size").alias("min_size"),
            max("size").alias("max_size")
        ).collect()[0]
        
        # Distance statistics
        distance_stats = df.select(
            avg("distance_from_earth").alias("avg_earth_distance"),
            min("distance_from_earth").alias("min_earth_distance"),
            max("distance_from_earth").alias("max_earth_distance")
        ).collect()[0]
        
        # Print summary
        print(f"\n=== ASTEROID DATA SUMMARY ===")
        print(f"Total asteroids: {total_asteroids}")
        print(f"High risk asteroids: {high_risk_asteroids}")
        print(f"Medium risk asteroids: {medium_risk_asteroids}")
        print(f"\nSize statistics:")
        print(f"  Average: {size_stats['avg_size']:.3f} km")
        print(f"  Std dev: {size_stats['stddev_size']:.3f} km")
        print(f"  Range: {size_stats['min_size']:.3f} - {size_stats['max_size']:.3f} km")
        print(f"\nDistance from Earth statistics:")
        print(f"  Average: {distance_stats['avg_earth_distance']:.0f} km")
        print(f"  Minimum: {distance_stats['min_earth_distance']:.0f} km")
        print(f"  Maximum: {distance_stats['max_earth_distance']:.0f} km")
        print(f"\nClassification breakdown:")
        for row in classification_stats:
            print(f"  {row['classification']}: {row['count']}")
    
    def run_processing_pipeline(self):
        """Run the complete data processing pipeline"""
        logger.info("Starting asteroid data processing pipeline...")
        
        # Define schemas
        self.define_schemas()
        
        # Load data
        asteroid_df = self.load_data_from_hdfs("/asteroid_data/raw/*.parquet", self.asteroid_schema)
        planet_df = self.load_data_from_hdfs("/planet_data/raw/*.parquet", self.planet_schema)
        
        if asteroid_df.count() == 0:
            logger.warning("No asteroid data found. Exiting...")
            return
        
        # Clean data
        clean_asteroid_df = self.clean_asteroid_data(asteroid_df)
        
        # Calculate trajectories and collision probabilities
        processed_df = self.calculate_collision_trajectories(clean_asteroid_df, planet_df)
        
        # Prepare ML features
        ml_ready_df, feature_columns = self.prepare_ml_features(processed_df)
        
        # Save processed data
        self.save_processed_data(processed_df, "/asteroid_data/processed/")
        self.save_processed_data(ml_ready_df, "/ml_training_data/")
        
        # Generate summary statistics
        self.generate_summary_statistics(processed_df)
        
        logger.info("Data processing pipeline completed successfully")
        
        return processed_df, ml_ready_df

def main():
    """Main function to run the Spark data processing"""
    processor = AsteroidDataProcessor()
    
    try:
        # Run the processing pipeline
        processed_df, ml_df = processor.run_processing_pipeline()
        
        # Keep Spark session alive for interactive use
        logger.info("Processing completed. Spark session available for interactive use.")
        
    except Exception as e:
        logger.error(f"Error in data processing: {e}")
    finally:
        # Uncomment the following line if you want to stop the Spark session
        # processor.spark.stop()
        pass

if __name__ == "__main__":
    main()if __name__ == "__main__":
    main()