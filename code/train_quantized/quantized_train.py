import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Quantize a pre-trained model using 4-bit BitsAndBytes config.")
parser.add_argument("--model_id", type=str, required=True, help="Hugging Face model ID to quantize.")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the quantized model.")
args = parser.parse_args()

# Configure 4-bit quantization using BitsAndBytes (NF4 format)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Load the pre-trained model with quantization
print(f"Loading model: {args.model_id}")
model = AutoModelForCausalLM.from_pretrained(
    args.model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_id)

# Save the quantized model and tokenizer
print(f"Saving quantized model to: {args.output_dir}")
model.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)

print("✅ Quantized model saved successfully!")
