import os, random, json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType
from trl import SFTTrainer, DPOTrainer, DPOConfig

# ─── Configuration constants (unchanged) ─────────────────────────────────
# Replace if you verified a different public or private ID on the Hub.
MODEL_ID = "Qwen/Qwen3-0.6B-Base"   # <<< VERIFY THIS EXISTS ON YOUR HF ACCOUNT

DATASET_REPO = "danielebelfiore/MNLP_M3_dpo_dataset"  # contains only "train"
EVAL_SPLIT    = 0.0                  # % of examples held out for eval

# ─── Reproducibility helpers ────────────────────────────────────────────
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()

# ─── Tokenizer (original logic left intact) ─────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

if tokenizer.pad_token is None:
    if tokenizer.eos_token is not None:
        print("Tokenizer lacks pad_token → using eos_token instead.")
        tokenizer.pad_token = tokenizer.eos_token
    else:
        print("Tokenizer lacks eos_token → adding [PAD] as a new pad_token.")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# ─── Load dataset (train + tiny eval split) ─────────────────────────────
train_dataset = load_dataset(DATASET_REPO, split="train")

if EVAL_SPLIT > 0:
    split = train_dataset.train_test_split(test_size=EVAL_SPLIT, seed=42, shuffle=True)
    train_dataset, eval_dataset = split["train"], split["test"]
else:
    eval_dataset = None

print(f"Train size: {len(train_dataset):,} • Eval size: {len(eval_dataset) if eval_dataset else 0:,}")

# ─── SFT: format + train LoRA adapters ──────────────────────────────────
def format_for_sft(record):
    # Preserve notebook’s chat-template formatting
    messages = [
        {"role": "user",      "content": record["prompt"]},
        {"role": "assistant", "content": record["chosen"]},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

sft_dataset = train_dataset.map(format_for_sft, remove_columns=list(train_dataset.features))
print("✅ SFT dataset ready.")

sft_base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="sdpa" # Use SDPA for efficiency
)

lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

sft_args = TrainingArguments(
    output_dir="./sft_qwen_lora_expanded",
    per_device_train_batch_size=2, # Can be slightly larger than DPO's
    gradient_accumulation_steps=4, # Adjust to maintain same effective batch size
    num_train_epochs=1,
    learning_rate=2e-4, # SFT can often use a slightly higher learning rate
    logging_steps=25,
    save_strategy="epoch",
    bf16=True,
    report_to="tensorboard",
    remove_unused_columns=True, # Saves memory
)

sft_trainer = SFTTrainer(
    model=sft_base_model,
    args=sft_args,
    train_dataset=sft_dataset,
    peft_config=lora_cfg,
    processing_class=tokenizer,
)


print("Starting SFT training...")
sft_trainer.train()
print("SFT training completed.")

sft_output_dir = "./sft_qwen_lora_expanded/final_adapters"
sft_trainer.model.save_pretrained(sft_output_dir)
print(f"SFT LoRA adapters saved to {sft_output_dir}")

lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

dpo_cfg = DPOConfig(
    output_dir="./dpo_qwen_lora_expanded",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=5e-5,
    beta=0.1,
    logging_steps=25,
    save_strategy="epoch",
    bf16=True,
    push_to_hub=False,
)

dpo_trainer = DPOTrainer(
    model=sft_trainer.model,  # Use the SFT-tuned model directly
    ref_model=None,
    processing_class=tokenizer,
    args=dpo_cfg,
    train_dataset=train_dataset, # Use the original DPO dataset
    eval_dataset=eval_dataset,
    peft_config=lora_cfg,
)

print("Starting DPO training...")
dpo_trainer.train()
dpo_trainer.model.save_pretrained("./dpo_qwen_lora_expanded/adapters")


print("🚀 Starting LoRA-DPO training …")
trainer.train()

print("💾 Saving DPO LoRA adapters …")
trainer.model.save_pretrained("./dpo_qwen_lora_expanded/adapters")

print("Merging adapters with base weights …")
merged_model = dpo_trainer.model.merge_and_unload()
merged_path = "./dpo_qwen_lora_expanded/merged"
merged_model.save_pretrained(merged_path)
tokenizer.save_pretrained(merged_path)
print(f"🎉 Merged DPO model available at {merged_path}")
