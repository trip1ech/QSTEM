#!/bin/bash

# Quantization Script for MCQA Model
# This script loads a pre-trained model and performs 4-bit quantization using BitsAndBytes

set -e  # Exit on any error

echo "🔧 Starting quantization of MCQA model..."

# Configuration
MODEL_NAME="trip1ech/MCQA-experiment"       # Pre-trained model to quantize
OUT_DIR="MCQA-experiment-quantized-nf4"     # Output directory for quantized model
SCRIPT_PATH="train_quantized/quantized_train.py"

# Create output directory if it doesn't exist
mkdir -p "$OUT_DIR"

echo "📂 Output directory created: $OUT_DIR"

# Step: Run quantization
echo "⚙️  Running quantization script..."
python "$SCRIPT_PATH" \
    --model_id "$MODEL_NAME" \
    --output_dir "$OUT_DIR"

echo "✅ Quantization completed successfully!"
echo "📦 Quantized model saved to: $OUT_DIR"
