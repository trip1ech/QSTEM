#!/bin/bash

# Fine-tuning Script for RAG-augmented MCQA Model
# Loads a base model, builds RAG training data from HF datasets, and fine-tunes the model.

set -e  # Exit on any error

echo "[INFO] Installing required Python packages..."
pip install -r code/train_rag/requirements.txt

# Configuration
MODEL_NAME="trip1ech/MCQA-experiment"    # Base model for fine-tuning
OUTPUT_DIR="rag_model"                   # Output directory for the fine-tuned model
SCRIPT_PATH="code/train_rag/rag_training.py"     # Path to your Python training script

echo "[INFO] Base Model: $MODEL_NAME"
echo "[INFO] Output Directory: $OUTPUT_DIR"
echo "[INFO] Training Script: $SCRIPT_PATH"

mkdir -p "$OUTPUT_DIR"

# Run fine-tuning
python3 "$SCRIPT_PATH" \
    --model_name "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR"

echo "[INFO] Fine-tuned model saved to: $OUTPUT_DIR"
