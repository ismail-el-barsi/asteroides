#!/usr/bin/env python3
"""
Startup script for asteroid tracking application
Runs data generator, visualization dashboard, and ML prediction system
"""

import multiprocessing
import os
import sys
import time


def run_data_generator():
    """Run the asteroid data generator"""
    import asteroid_data_generator
    asteroid_data_generator.main()

def run_hdfs_consumer():
    """Run the Kafka to HDFS consumer"""
    # Wait for Kafka and HDFS to be ready
    time.sleep(20)
    import kafka_to_hdfs_consumer
    kafka_to_hdfs_consumer.main()

def run_dashboard():
    """Run the visualization dashboard"""
    # Wait a bit for Kafka to be ready
    time.sleep(10)
    import visualization_dashboard
    visualization_dashboard.main()

def run_ml_predictor():
    """Run the ML collision predictor"""
    # Wait for data to be available
    time.sleep(30)
    print("🧠 Démarrage du système ML/DL de prédiction...")
    import asteroid_ml_final
    asteroid_ml_final.main()

def main():
    """Main function to start all processes"""
    print("🌌 Starting Asteroid Tracking Application with ML/DL...")
    print("=" * 60)
    
    # Start data generator process
    print("📊 Starting data generator...")
    generator_process = multiprocessing.Process(target=run_data_generator)
    generator_process.start()
    
    # Start HDFS consumer process
    print("💾 Starting HDFS consumer...")
    consumer_process = multiprocessing.Process(target=run_hdfs_consumer)
    consumer_process.start()
    
    # Start dashboard process
    print("📈 Starting dashboard...")
    dashboard_process = multiprocessing.Process(target=run_dashboard)
    dashboard_process.start()
    
    # Start ML predictor process
    print("🤖 Starting ML/DL collision predictor...")
    ml_process = multiprocessing.Process(target=run_ml_predictor)
    ml_process.start()
    
    try:
        # Wait for all processes
        generator_process.join()
        consumer_process.join()
        dashboard_process.join()
        ml_process.join()
    except KeyboardInterrupt:
        print("\nShutting down...")
        generator_process.terminate()
        consumer_process.terminate()
        dashboard_process.terminate()
        ml_process.terminate()
        generator_process.join()
        consumer_process.join()
        dashboard_process.join()
        ml_process.join()

if __name__ == "__main__":
    main()