#!/bin/bash

# Create a virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies (This will take a few minutes for PyTorch and Torch Geometric)
echo "Installing dependencies... (PyTorch is ~500MB, please be patient)"
pip install -r requirements.txt

# Run the training pipeline
echo ""
echo "=== 1. Starting Training ==="
python train.py

# Run the evaluation pipeline
echo ""
echo "=== 2. Running Evaluation ==="
python evaluate.py

# Run a sample prediction
echo ""
echo "=== 3. Testing Single Prediction ==="
python predict.py

echo ""
echo "Pipeline finished successfully."
