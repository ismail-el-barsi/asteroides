#!/usr/bin/env python3
"""
Kafka to HDFS Consumer
Consumes asteroid and planet data from Kafka topics and stores them in HDFS
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd
from hdfs3 import HDFileSystem
from kafka import KafkaConsumer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KafkaToHDFSConsumer:
    def __init__(self, 
                 kafka_servers: List[str] = ['kafka:9092'],
                 hdfs_host: str = 'namenode',
                 hdfs_port: int = 9000):
        """Initialize the Kafka to HDFS consumer"""
        
        # Kafka configuration
        self.kafka_servers = kafka_servers
        
        # HDFS configuration
        self.hdfs_host = hdfs_host
        self.hdfs_port = hdfs_port
        
        # Data buffers for batch processing
        self.asteroid_buffer = []
        self.planet_buffer = []
        self.alert_buffer = []
        
        # Buffer size before writing to HDFS
        self.buffer_size = 100
        
        # File rotation settings
        self.current_date = datetime.now().date()
        
        # Initialize connections
        self.setup_hdfs_connection()
        self.setup_kafka_consumers()
        
    def setup_hdfs_connection(self):
        """Setup HDFS connection"""
        try:
            # Wait for HDFS to be ready
            logger.info("Waiting for HDFS to be ready...")
            time.sleep(60)  # Give HDFS time to start
            
            self.hdfs = HDFileSystem(host=self.hdfs_host, port=self.hdfs_port)
            
            # Create directory structure
            self.create_hdfs_directories()
            logger.info("HDFS connection established successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to HDFS: {e}")
            # Fallback to local file system for development
            logger.info("Using local file system as fallback")
            self.hdfs = None
            self.create_local_directories()
    
    def setup_kafka_consumers(self):
        """Setup Kafka consumers for different topics"""
        try:
            # Wait for Kafka to be ready
            logger.info("Waiting for Kafka to be ready...")
            time.sleep(30)
            
            self.asteroid_consumer = KafkaConsumer(
                'asteroid-data',
                bootstrap_servers=self.kafka_servers,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='latest',
                group_id='asteroid-hdfs-consumer'
            )
            
            self.planet_consumer = KafkaConsumer(
                'planet-data',
                bootstrap_servers=self.kafka_servers,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='latest',
                group_id='planet-hdfs-consumer'
            )
            
            self.alert_consumer = KafkaConsumer(
                'collision-alerts',
                bootstrap_servers=self.kafka_servers,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='latest',
                group_id='alert-hdfs-consumer'
            )
            
            logger.info("Kafka consumers initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Kafka consumers: {e}")
            raise
    
    def create_hdfs_directories(self):
        """Create HDFS directory structure"""
        directories = [
            '/asteroid_data',
            '/asteroid_data/raw',
            '/asteroid_data/processed',
            '/planet_data',
            '/planet_data/raw',
            '/collision_alerts',
            '/collision_alerts/raw',
            '/ml_models',
            '/ml_training_data'
        ]
        
        for directory in directories:
            try:
                if not self.hdfs.exists(directory):
                    self.hdfs.makedirs(directory)
                    logger.info(f"Created HDFS directory: {directory}")
            except Exception as e:
                logger.error(f"Failed to create HDFS directory {directory}: {e}")
    
    def create_local_directories(self):
        """Create local directory structure as fallback"""
        base_dir = "/tmp/asteroid_data"
        directories = [
            f"{base_dir}/asteroid_data/raw",
            f"{base_dir}/asteroid_data/processed",
            f"{base_dir}/planet_data/raw",
            f"{base_dir}/collision_alerts/raw",
            f"{base_dir}/ml_models",
            f"{base_dir}/ml_training_data"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created local directory: {directory}")
    
    def write_to_hdfs(self, data: List[Dict], file_path: str):
        """Write data to HDFS in Parquet format"""
        try:
            if not data:
                return
                
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            if self.hdfs:
                # Write to HDFS
                with self.hdfs.open(file_path, 'wb') as f:
                    df.to_parquet(f, index=False)
            else:
                # Write to local file system
                local_path = f"/tmp/asteroid_data{file_path}"
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                df.to_parquet(local_path, index=False)
            
            logger.info(f"Written {len(data)} records to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to write to HDFS {file_path}: {e}")
            # Fallback: write to local JSON file
            self.write_to_json_fallback(data, file_path)
    
    def write_to_json_fallback(self, data: List[Dict], file_path: str):
        """Fallback method to write data as JSON"""
        try:
            local_path = f"/tmp/asteroid_data{file_path}.json"
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            with open(local_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Written {len(data)} records to fallback JSON: {local_path}")
            
        except Exception as e:
            logger.error(f"Failed to write fallback JSON {local_path}: {e}")
    
    def process_asteroid_data(self, message):
        """Process asteroid data message"""
        try:
            data = message.value
            
            # Add processing timestamp
            data['processed_timestamp'] = datetime.now().isoformat()
            
            # Add to buffer
            self.asteroid_buffer.append(data)
            
            # Check if buffer is full or date has changed
            if (len(self.asteroid_buffer) >= self.buffer_size or 
                self.should_rotate_file()):
                self.flush_asteroid_buffer()
                
        except Exception as e:
            logger.error(f"Error processing asteroid data: {e}")
    
    def process_planet_data(self, message):
        """Process planet data message"""
        try:
            data = message.value
            
            # Add processing timestamp
            data['processed_timestamp'] = datetime.now().isoformat()
            
            # Add to buffer
            self.planet_buffer.append(data)
            
            # Check if buffer is full or date has changed
            if (len(self.planet_buffer) >= self.buffer_size or 
                self.should_rotate_file()):
                self.flush_planet_buffer()
                
        except Exception as e:
            logger.error(f"Error processing planet data: {e}")
    
    def process_alert_data(self, message):
        """Process collision alert data message"""
        try:
            data = message.value
            
            # Add processing timestamp
            data['processed_timestamp'] = datetime.now().isoformat()
            
            # Add to buffer
            self.alert_buffer.append(data)
            
            # Alerts are more urgent, use smaller buffer
            if (len(self.alert_buffer) >= 10 or 
                self.should_rotate_file()):
                self.flush_alert_buffer()
                
        except Exception as e:
            logger.error(f"Error processing alert data: {e}")
    
    def should_rotate_file(self) -> bool:
        """Check if files should be rotated (daily rotation)"""
        current = datetime.now().date()
        if current != self.current_date:
            self.current_date = current
            return True
        return False
    
    def flush_asteroid_buffer(self):
        """Flush asteroid buffer to HDFS"""
        if self.asteroid_buffer:
            timestamp = datetime.now()
            file_path = f"/asteroid_data/raw/asteroids_{timestamp.strftime('%Y%m%d_%H%M%S')}.parquet"
            self.write_to_hdfs(self.asteroid_buffer, file_path)
            self.asteroid_buffer.clear()
    
    def flush_planet_buffer(self):
        """Flush planet buffer to HDFS"""
        if self.planet_buffer:
            timestamp = datetime.now()
            file_path = f"/planet_data/raw/planets_{timestamp.strftime('%Y%m%d_%H%M%S')}.parquet"
            self.write_to_hdfs(self.planet_buffer, file_path)
            self.planet_buffer.clear()
    
    def flush_alert_buffer(self):
        """Flush alert buffer to HDFS"""
        if self.alert_buffer:
            timestamp = datetime.now()
            file_path = f"/collision_alerts/raw/alerts_{timestamp.strftime('%Y%m%d_%H%M%S')}.parquet"
            self.write_to_hdfs(self.alert_buffer, file_path)
            self.alert_buffer.clear()
    
    def consume_messages(self):
        """Main method to consume messages from all topics"""
        logger.info("Starting Kafka message consumption...")
        
        try:
            while True:
                # Process asteroid data
                for message in self.asteroid_consumer.poll(timeout_ms=1000).values():
                    for msg in message:
                        self.process_asteroid_data(msg)
                
                # Process planet data
                for message in self.planet_consumer.poll(timeout_ms=1000).values():
                    for msg in message:
                        self.process_planet_data(msg)
                
                # Process alert data
                for message in self.alert_consumer.poll(timeout_ms=1000).values():
                    for msg in message:
                        self.process_alert_data(msg)
                
                # Periodic flush (every 5 minutes)
                current_time = datetime.now()
                if current_time.minute % 5 == 0 and current_time.second < 10:
                    self.flush_all_buffers()
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Stopping consumer...")
        except Exception as e:
            logger.error(f"Error in message consumption: {e}")
        finally:
            self.cleanup()
    
    def flush_all_buffers(self):
        """Flush all buffers to HDFS"""
        self.flush_asteroid_buffer()
        self.flush_planet_buffer()
        self.flush_alert_buffer()
    
    def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up resources...")
        
        # Flush remaining data
        self.flush_all_buffers()
        
        # Close Kafka consumers
        try:
            self.asteroid_consumer.close()
            self.planet_consumer.close()
            self.alert_consumer.close()
        except Exception as e:
            logger.error(f"Error closing Kafka consumers: {e}")
        
        logger.info("Cleanup completed")

def main():
    """Main function to run the Kafka to HDFS consumer"""
    logger.info("Starting Kafka to HDFS Consumer...")
    
    consumer = KafkaToHDFSConsumer()
    consumer.consume_messages()

if __name__ == "__main__":
    main()    main()