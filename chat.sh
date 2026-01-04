#!/bin/bash

# Script to run the interactive Agentic RAG chatbot

PYTHON_PATH="/home/akif/miniconda3/envs/coder_agent/bin/python"

if [ ! -f "$PYTHON_PATH" ]; then
    echo "Error: Python interpreter not found at $PYTHON_PATH"
    echo "Falling back to 'conda run'..."
    conda run -n coder_agent python -m mini_agentic_rag.main
    exit 0
fi

echo "Starting Mini Agentic RAG Chatbot..."
echo "Press Ctrl+C to exit."

# Run the python module directly (fastest)
"$PYTHON_PATH" -m mini_agentic_rag.main
