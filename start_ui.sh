#!/bin/bash

# Activate the virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "Virtual environment not found. Please run ./run.sh first to set up and train the models."
    exit 1
fi

echo "Starting the GAT+RDBN UI Server..."
echo "Once the server starts, open your web browser and go to: http://localhost:8000"
echo "(Press CTRL+C to quit)"
echo ""

# Run the server
python serve.py
