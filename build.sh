#!/bin/bash

# Install Python dependencies
pip install -r requirements.txt

# Train models if they don't exist
python scripts/current_drivers_predictor.py --race "Monaco Grand Prix" --year 2024

echo "Build completed successfully!" 