import glob
import json
import logging
import math
import os
import random
import time
import csv
import sys
csv.field_size_limit(sys.maxsize)

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    BertForMaskedLM,
    BertModel,
    BertTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    trainer_utils,
)

from config import BertTrainingConfig

CFG = BertTrainingConfig()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_dataset_from_jsonl(data_dir: str) -> Dataset:
    files = sorted(glob.glob(os.path.join(data_dir, "**", "*.jsonl"), recursive=True))
    if not files:
        raise FileNotFoundError(
            f"No .jsonl files found under {data_dir}. "
            "Ensure the data directory exists and contains speech data."
        )
    logger.info("Found %d .jsonl file(s) under %s", len(files), data_dir)
    records = []
    for path in files:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    logger.info("Loaded %d total records", len(records))
    return Dataset.from_list(records)

def load_dataset_from_csv(csv_dir: str) -> Dataset:
    records = []
    with open(csv_dir, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            records.append({'text': row[1]})
    logger.info("Loaded %d total records", len(records))
    return Dataset.from_list(records)

def tokenize_batch(examples: dict, tokenizer: BertTokenizer, max_length: int) -> dict:
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_special_tokens_mask=True,
    )


def main() -> None:
    set_seeds(CFG.seed)
    os.makedirs(CFG.output_dir, exist_ok=True)
    os.makedirs(CFG.log_dir, exist_ok=True)
    os.makedirs(CFG.final_dir, exist_ok=True)

    logger.info("=== BERT Training Configuration ===")
    CFG.log_config(logger)

    if CFG.data_type == "jsonl":
        raw_dataset = load_dataset_from_jsonl(CFG.data_dir)
    elif CFG.data_type == "csv":
        raw_dataset = load_dataset_from_csv(CFG.data_dir)
    else:
        raise ValueError(f"Unsupported data type: {CFG.data_type}")
    
    raw_dataset = raw_dataset.shuffle(seed=CFG.shuffle_seed)
    split = raw_dataset.train_test_split(test_size=CFG.eval_split, seed=CFG.shuffle_seed)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    logger.info("Train examples: %d, Eval examples: %d", len(train_dataset), len(eval_dataset))

    tokenizer = BertTokenizer.from_pretrained(CFG.model_name)
    tokenize_fn = lambda examples: tokenize_batch(examples, tokenizer, CFG.max_length)
    train_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    eval_dataset = eval_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    logger.info("Tokenization complete")

    model = BertForMaskedLM.from_pretrained(CFG.model_name)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable parameters: %d", num_params)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=CFG.mlm_probability,
    )

    training_args = TrainingArguments(
        output_dir=CFG.output_dir,
        num_train_epochs=CFG.num_train_epochs,
        per_device_train_batch_size=CFG.per_device_train_batch_size,
        per_device_eval_batch_size=CFG.per_device_eval_batch_size,
        gradient_accumulation_steps=CFG.gradient_accumulation_steps,
        learning_rate=CFG.learning_rate,
        weight_decay=CFG.weight_decay,
        lr_scheduler_type=CFG.lr_scheduler_type,
        fp16=CFG.fp16,
        logging_steps=CFG.logging_steps,
        save_steps=CFG.save_steps,
        eval_steps=CFG.eval_steps,
        eval_strategy="steps",
        save_total_limit=CFG.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=CFG.dataloader_num_workers,
        report_to="none",
        seed=CFG.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    last_checkpoint = trainer_utils.get_last_checkpoint(CFG.output_dir)
    if last_checkpoint:
        logger.info("Resuming from checkpoint: %s", last_checkpoint)
    else:
        logger.info("Starting fresh training run")

    start_time = time.time()
    trainer.train(resume_from_checkpoint=last_checkpoint)
    elapsed = time.time() - start_time
    hours, remainder = divmod(int(elapsed), 3600)
    minutes = remainder // 60
    logger.info("Training complete in %dh %dm", hours, minutes)

    trainer.save_model(CFG.final_dir)
    tokenizer.save_pretrained(CFG.final_dir)
    logger.info("Model saved to %s", CFG.final_dir)
    logger.info("Tokenizer saved to %s", CFG.final_dir)
    logger.info("Total training time: %dh %dm", hours, minutes)

    eval_results = trainer.evaluate()
    perplexity = math.exp(eval_results["eval_loss"])
    logger.info("Validation loss: %.4f", eval_results["eval_loss"])
    logger.info("Perplexity: %.2f", perplexity)

    try:
        verify_tokenizer = BertTokenizer.from_pretrained(CFG.final_dir)
        verify_model = BertModel.from_pretrained(CFG.final_dir)
        verify_model.eval()
        inputs = verify_tokenizer("The senate passed the bill.", return_tensors="pt")
        with torch.no_grad():
            outputs = verify_model(**inputs, output_hidden_states=True)
        assert len(outputs.hidden_states) == CFG.expected_hidden_layers, (
            f"Expected {CFG.expected_hidden_layers} hidden layers, "
            f"got {len(outputs.hidden_states)}"
        )
        assert outputs.hidden_states[-1].shape[-1] == CFG.expected_hidden_dim, (
            f"Expected hidden dim {CFG.expected_hidden_dim}, "
            f"got {outputs.hidden_states[-1].shape[-1]}"
        )
        print(
            f"Hidden state extraction verified: "
            f"{CFG.expected_hidden_layers} layers, {CFG.expected_hidden_dim} dims"
        )
    except Exception as exc:
        logger.warning("Hidden state verification failed (training is complete): %s", exc)


if __name__ == "__main__":
    main()
