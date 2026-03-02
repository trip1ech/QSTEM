"""
Tokenize and pack MCQA dataset for training.
"""

import argparse
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoTokenizer
from itertools import chain

def parse_args():
    parser = argparse.ArgumentParser(description="Tokenize MCQA dataset")
    parser.add_argument("--data_dir", default="train_mcqa/raw_dataset",
                       help="Input dataset directory")
    parser.add_argument("--base_model", default="Qwen/Qwen3-0.6B-Base",
                       help="Base model for tokenizer")
    parser.add_argument("--tokenizer_dir", default="train_mcqa/tokenizer_mcq",
                       help="Output directory for tokenizer")
    parser.add_argument("--output_dir", default="train_mcqa/mcq_tokenised",
                       help="Output directory for tokenized data")
    parser.add_argument("--max_seq_len", type=int, default=1024,
                       help="Maximum sequence length")
    return parser.parse_args()

def format_example(ex):
    """Format a single MCQA example into text prompt."""
    stem, opts, lbl = ex["question"], ex["choices"], ex["answer"][0]
    
    prompt = ("The following are multiple choice questions (with answers) "
              "about knowledge and skills in advanced master-level STEM courses.\n\n")
    prompt += stem + "\n"
    prompt += f"A. {opts[0]}\nB. {opts[1]}\nC. {opts[2]}\nD. {opts[3]}\n"
    prompt += "Answer:"
    prompt += f" {lbl}"
    
    return prompt

def main():
    args = parse_args()
    
    print("Loading dataset...")
    ds = load_from_disk(args.data_dir)
    
    print("Setting up tokenizer...")
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    
    # Add special tokens if needed
    extra_tokens = ["###", "END", "###"]
    tok.add_tokens(extra_tokens, special_tokens=False)
    tok.pad_token = tok.eos_token
    
    # Save tokenizer
    tokenizer_path = Path(args.tokenizer_dir)
    tokenizer_path.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(tokenizer_path)
    print(f"Tokenizer saved to: {tokenizer_path}")
    
    print("Formatting examples...")
    # Map each example to formatted text
    for split in ds.keys():
        print(f"Processing {split} split...")
        ds[split] = ds[split].map(
            lambda x: {"text": format_example(x)},
            remove_columns=ds[split].column_names,
            num_proc=4
        )
    
    print("Packing sequences...")
    def pack_texts(examples):
        """Pack multiple texts into sequences of max_seq_len tokens."""
        # Tokenize all texts
        all_ids = tok(examples["text"])["input_ids"]
        # Flatten into single sequence
        concatenated = list(chain.from_iterable(all_ids))
        # Split into chunks
        chunks = [
            concatenated[i : i + args.max_seq_len]
            for i in range(0, len(concatenated), args.max_seq_len)
            if len(concatenated[i : i + args.max_seq_len]) > 16  # Skip very short chunks
        ]
        return {"input_ids": chunks}
    
    packed = ds.map(
        pack_texts,
        batched=True,
        remove_columns=["text"],
        num_proc=4
    )
    
    # Set format for torch
    packed.set_format("torch", columns=["input_ids"])
    
    # Save packed dataset
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    packed.save_to_disk(output_path)
    
    print(f"Tokenized dataset saved to: {output_path}")
    
    # Print statistics
    for split_name, split_data in packed.items():
        print(f"  {split_name}: {len(split_data)} packed sequences")

if __name__ == "__main__":
    main()