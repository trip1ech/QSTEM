"""
Download and prepare the MCQA dataset from HuggingFace.
"""

import argparse
from pathlib import Path
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare MCQA dataset")
    parser.add_argument("--dataset_name", default="trip1ech/MNLP_M3_mcqa_dataset",
                       help="HuggingFace dataset name")
    parser.add_argument("--base_model", default="Qwen/Qwen3-0.6B-Base",
                       help="Base model name")
    parser.add_argument("--output_dir", default="train_mcqa/raw_dataset",
                       help="Output directory for dataset")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Loading dataset: {args.dataset_name}")
    
    # Load the dataset from HuggingFace
    dataset = load_dataset(args.dataset_name)
    
    # Ensure we have train split (required for training)
    if "train" not in dataset:
        raise ValueError("Dataset must contain 'train' split")
    
    print(f"Dataset loaded successfully:")
    for split_name, split_data in dataset.items():
        print(f"  {split_name}: {len(split_data)} examples")
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save dataset to disk
    dataset.save_to_disk(output_path)
    
    print(f"Dataset saved to: {output_path}")
    
    # Print sample to verify format
    print("\nSample from train split:")
    sample = dataset["train"][0]
    for key, value in sample.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()