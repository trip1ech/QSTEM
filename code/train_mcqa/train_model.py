"""
Train MCQA model using the prepared and tokenized dataset.
This is your existing training code adapted for command-line execution.
"""

import argparse
import os
import csv
from pathlib import Path
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer as _HfTrainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
import torch

class EvalEveryNStepsCallback(TrainerCallback):
    """A TrainerCallback that sets should_evaluate = True every eval_steps steps."""

    def __init__(self, eval_steps: int):
        super().__init__()
        self.eval_steps = eval_steps

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        if state.global_step % self.eval_steps == 0:
            control.should_evaluate = True
        return control

class LossLoggingCallback(TrainerCallback):
    """Collects training and evaluation losses and saves them to CSV."""
    
    def __init__(self, output_dir: str):
        super().__init__()
        self.output_dir = output_dir
        self.train_records = []
        self.eval_records = []

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        log_dict = state.log_history[-1]

        if "loss" in log_dict and "step" in log_dict:
            self.train_records.append((log_dict["step"], log_dict["loss"]))

        if "eval_loss" in log_dict and "step" in log_dict:
            self.eval_records.append((log_dict["step"], log_dict["eval_loss"]))

        return control

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        os.makedirs(self.output_dir, exist_ok=True)
        csv_path = os.path.join(self.output_dir, "loss_curve.csv")
        
        with open(csv_path, "w", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(["step", "train_loss", "eval_loss"])
            
            all_steps = sorted({step for step, _ in self.train_records} | {step for step, _ in self.eval_records})
            train_dict = dict(self.train_records)
            eval_dict = dict(self.eval_records)
            
            for step in all_steps:
                t = train_dict.get(step, "")
                e = eval_dict.get(step, "")
                writer.writerow([step, t, e])

        print(f"→ Saved loss curve CSV to: {csv_path}")

class Trainer(_HfTrainer):
    """Custom Trainer that doesn't move model to device (let Accelerate handle it)."""

    def _move_model_to_device(self, model, device):
        return model

def parse_args():
    parser = argparse.ArgumentParser(description="Train MCQA model")
    parser.add_argument("--data", default="train_mcqa/mcq_tokenised",
                       help="Path to packed HF dataset on disk")
    parser.add_argument("--tok_dir", default="train_mcqa/tokenizer_mcq",
                       help="Where tokenizer with special tokens is saved")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B-Base",
                       help="Base model name")
    parser.add_argument("--out_dir", default="MNLP_M3_mcqa_model",
                       help="Output directory for trained model")
    parser.add_argument("--epochs", type=int, default=4,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for training")
    parser.add_argument("--lr", type=float, default=5e-5,
                       help="Learning rate")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Loading dataset from: {args.data}")
    ds = load_from_disk(args.data)
    
    print(f"Loading tokenizer from: {args.tok_dir}")
    tok = AutoTokenizer.from_pretrained(args.tok_dir, trust_remote_code=True)
    
    print(f"Loading base model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    
    print("Resizing token embeddings...")
    model.resize_token_embeddings(len(tok))
    
    # Data collator for causal language modeling
    data_collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)
    
    # Create output directory
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        
        # Batch sizes and gradient accumulation
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,  # effective batch = 32
        per_device_eval_batch_size=args.batch_size,
        
        # Training schedule
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        weight_decay=0.05,
        lr_scheduler_type="cosine",
        
        # Mixed precision and memory optimization
        bf16=True,
        gradient_checkpointing=True,
        
        # Logging and saving
        logging_steps=10,
        logging_dir=f"{args.out_dir}/logs",
        save_steps=100,
        save_total_limit=2,
        
        report_to="tensorboard",
    )
    
    print("Setting up trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("dev", ds.get("validation", None)),
        tokenizer=tok,
        data_collator=data_collator,
        callbacks=[
            EvalEveryNStepsCallback(eval_steps=10),
            LossLoggingCallback(output_dir=args.out_dir),
        ]
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Device of model parameter:", next(model.parameters()).device)
    
    # Final evaluation
    if trainer.eval_dataset is not None:
        print(">>> Final evaluation metrics:")
        metrics = trainer.evaluate()
        for k, v in metrics.items():
            print(f"  {k}: {v}")
    
    # Save the final model
    print(f"Saving model to: {args.out_dir}")
    trainer.save_model(args.out_dir)
    
    print("Model training completed successfully!")

if __name__ == "__main__":
    main()