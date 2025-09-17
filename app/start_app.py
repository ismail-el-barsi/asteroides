#!/usr/bin/env python3
"""
Startup script for asteroid tracking application
Runs both data generator and visualization dashboard
"""

import multiprocessing
import os
import sys
import time


def run_data_generator():
    """Run the asteroid data generator"""
    import asteroid_data_generator
    asteroid_data_generator.main()

def run_dashboard():
    """Run the visualization dashboard"""
    # Wait a bit for Kafka to be ready
    time.sleep(10)
    import visualization_dashboard
    visualization_dashboard.main()

def main():
    """Main function to start both processes"""
    print("Starting Asteroid Tracking Application...")
    print("=" * 50)
    
    # Start data generator process
    print("Starting data generator...")
    generator_process = multiprocessing.Process(target=run_data_generator)
    generator_process.start()
    
    # Start dashboard process
    print("Starting dashboard...")
    dashboard_process = multiprocessing.Process(target=run_dashboard)
    dashboard_process.start()
    
    try:
        # Wait for both processes
        generator_process.join()
        dashboard_process.join()
    except KeyboardInterrupt:
        print("\nShutting down...")
        generator_process.terminate()
        dashboard_process.terminate()
        generator_process.join()
        dashboard_process.join()

if __name__ == "__main__":
    main()    main()