#!/bin/bash

# MCQA Model Training Script
# This script trains the MNLP_M3_mcqa_model from the HuggingFace dataset

set -e  # Exit on any error

echo "Starting MCQA model training..."

# Configuration
MODEL_NAME="Qwen/Qwen3-0.6B-Base"
TOKENIZER_DIR="train_mcqa/tokenizer_mcq"
DATASET_NAME="trip1ech/MNLP_M3_mcqa_dataset"  # Your HF dataset repo
PACKED_DATA="train_mcqa/mcq_tokenised"
OUT_DIR="MNLP_M3_mcqa_model"

# Create necessary directories
mkdir -p train_mcqa/tokenizer_mcq
mkdir -p train_mcqa/mcq_tokenised
mkdir -p "$OUT_DIR"

echo "Created directory structure"

# Step 1: Download and prepare dataset
echo "Step 1: Downloading and preparing dataset..."
python train_mcqa/prepare_dataset.py \
    --dataset_name "$DATASET_NAME" \
    --base_model "$MODEL_NAME" \
    --output_dir "train_mcqa/raw_dataset"

echo "Dataset preparation completed"

# Step 2: Tokenize and pack dataset
echo "Step 2: Tokenizing and packing dataset..."
python train_mcqa/tokenize_data.py \
    --data_dir "trip1ech/NLP4Education_mcqs" \
    --base_model "$MODEL_NAME" \
    --tokenizer_dir "$TOKENIZER_DIR" \
    --output_dir "$PACKED_DATA" \
    --max_seq_len 1024

echo "Tokenization completed"

# Step 3: Train the model
echo "Step 3: Training MCQA model..."
python train_mcqa/train_model.py \
    --data "$PACKED_DATA" \
    --tok_dir "$TOKENIZER_DIR" \
    --model "$MODEL_NAME" \
    --out_dir "$OUT_DIR" \
    --epochs 4 \
    --batch_size 4 \
    --lr 5e-5

echo "Training completed"

echo "MCQA model training pipeline completed successfully!"
echo "Model saved to: $OUT_DIR"
echo "Training logs and loss curves available in: $OUT_DIR/logs"