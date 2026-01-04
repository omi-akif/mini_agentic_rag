#!/bin/bash

# Script to ingest knowledge into Qdrant for Agentic RAG

# Use the same Python environment as chat.sh
PYTHON_PATH="/home/akif/miniconda3/envs/coder_agent/bin/python"

# Set the current directory as the base path
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$PYTHONPATH:$BASE_DIR/src"

if [ ! -f "$PYTHON_PATH" ]; then
    echo "Error: Python interpreter not found at $PYTHON_PATH"
    echo "Falling back to system python3..."
    PYTHON_PATH="python3"
fi

echo "--- Starting Knowledge Ingestion ---"

# Pass all arguments to the python script
"$PYTHON_PATH" -m mini_agentic_rag.ingestion "$@"

echo "--- Ingestion Script Execution Finished ---"
