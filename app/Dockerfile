# Dockerfile for asteroid tracking applications
FROM python:3.11.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    default-jdk \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME
ENV JAVA_HOME=/usr/lib/jvm/default-java

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY *.py ./

# Create data directories
RUN mkdir -p /tmp/asteroid_data/models \
    && mkdir -p /tmp/asteroid_data/results \
    && mkdir -p /tmp/asteroid_data/plots

# Expose ports
EXPOSE 8050 8888

# Default command
CMD ["python", "asteroid_data_generator.py"]