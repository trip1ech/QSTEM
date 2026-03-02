import argparse
from datasets import load_dataset, Dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a model on RAG-augmented MCQA dataset.")
    parser.add_argument("--model_name", type=str, required=True, help="Base model name or path.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the fine-tuned model.")
    args = parser.parse_args()

    # 1. Load RAG documents
    print("[INFO] Loading RAG corpus from Hugging Face Hub...")
    corpus = load_dataset("nililba/MNLP_M3_rag_documents", split="train")
    corpus_texts = [item['text'] for item in corpus]
    print(f"[INFO] Loaded {len(corpus_texts)} context documents.")

    # 2. Encode corpus documents
    print("[INFO] Encoding context documents...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    corpus_embeddings = embedder.encode(corpus_texts, batch_size=64, show_progress_bar=True)
    print("[INFO] Encoded corpus.")

    # 3. Load QA pairs
    print("[INFO] Loading QA dataset...")
    dataset = load_dataset("nililba/MNLP_M3_rag_dataset", split="train")
    qa_pairs = [
        {
            "question": row["question"],
            "answer": row["answer"],
            "choices": row.get("choices", None),
            "justification": row.get("justification", "")
        }
        for row in dataset
    ]
    print(f"[INFO] Loaded {len(qa_pairs)} QA pairs.")

    # 4. Retrieval function
    def retrieve_context(question, k=1):
        q_emb = embedder.encode([question])
        sims = cosine_similarity(q_emb, corpus_embeddings)[0]
        topk = sims.argsort()[-k:][::-1]
        return [corpus_texts[i] for i in topk]

    # 5. Prepare RAG examples
    rag_examples = []
    for qa in qa_pairs:
        context = retrieve_context(qa["question"], k=1)[0]
        input_str = f"Question: {qa['question']}\nContext: {context}\nAnswer:"
        output_str = qa["answer"]
        rag_examples.append({"input": input_str, "output": output_str})

    print("[INFO] Created RAG-style examples.")

    # 6. Tokenization
    print("[INFO] Tokenizing examples...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    def preprocess(example):
        enc = tokenizer(example["input"], max_length=1024, truncation=True, padding="max_length")
        with tokenizer.as_target_tokenizer():
            label_enc = tokenizer(example["output"], max_length=64, truncation=True, padding="max_length")
        labels = [-100] * len(enc["input_ids"])
        num_label = len([i for i in label_enc["input_ids"] if i != tokenizer.pad_token_id])
        if num_label > 0:
            labels[-num_label:] = label_enc["input_ids"][:num_label]
        enc["labels"] = labels
        return enc

    hf_ds = Dataset.from_list(rag_examples)
    tokenized_ds = hf_ds.map(preprocess)

    # 7. Fine-tuning
    print("[INFO] Starting fine-tuning...")
    args_train = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        logging_steps=1,
        save_strategy="epoch",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args_train,
        train_dataset=tokenized_ds,
        tokenizer=tokenizer
    )

    trainer.train()
    print("[INFO] Done! Saving model and tokenizer...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
