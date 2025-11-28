#!/bin/bash

echo "Setting up fake dataset for Airport Crime Detection System..."

# Install required packages
echo "Installing required packages..."
# pip install plotly streamlit-webrtc

# Generate fake dataset
echo "Generating fake dataset..."
python main.py generate --num_normal 5 --num_anomaly 5

# Extract frames from fake dataset
echo "Extracting frames..."
python main.py extract

# Test data generator
echo "Testing data generator..."
python -c "from src.data_preprocessing.data_generator import test_data_generator; test_data_generator()"

echo "Setup complete! You can now:"
echo "1. Train the model: python main.py train --epochs 5"
echo "2. Run the dashboard: python main.py dashboard"
echo "3. Test detection: python main.py detect"
