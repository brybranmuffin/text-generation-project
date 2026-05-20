# Congressional Speech LM Training — Design Spec

**Date:** 2026-05-20
**Approach:** Config-first (Option A)

---

## Overview

Implement two continued pretraining scripts for fine-tuning language models on congressional speech data, to be run as SLURM jobs on a university HPC cluster. Both scripts must support checkpoint resumption, log training progress, and verify hidden state access after training.

---

## Files & Directory Layout

Seven files total:

```
models/BERT/
  config.py          — BertTrainingConfig frozen dataclass
  bert.py            — MLM continued pretraining script
  submit_bert.sh     — SLURM job script

models/GPT2/
  config.py          — GPT2TrainingConfig frozen dataclass
  gpt2.py            — CLM continued pretraining script
  submit_gpt2.sh     — SLURM job script

requirements.txt     — project root
```

Local repo uses uppercase directories (`models/BERT/`, `models/GPT2/`). SLURM scripts reference HPC-side paths (`/models/bert/`, `/models/gpt2/`) which are independent of local casing.

---

## Architecture

### Config layer (`config.py`)

Each model has a frozen dataclass (`BertTrainingConfig` / `GPT2TrainingConfig`) that:
- Defines all paths, hyperparameters, and verification constants as typed fields
- Validates invariants in `__post_init__` (raises `ValueError` on bad values)
- Exposes `to_dict()` returning `dataclasses.asdict(self)`
- Exposes `log_config(logger)` logging every field at INFO level

### Training script layer (`bert.py` / `gpt2.py`)

Each script imports its config as a module-level constant (`CFG = BertTrainingConfig()`) and executes in this order:

1. Set random seeds (`random`, `numpy`, `torch` — all seed 42)
2. Create output directories (`os.makedirs(..., exist_ok=True)`)
3. Log full config via `CFG.log_config(logger)`
4. Load + shuffle (seed 42) + split (99/1) `.jsonl` data from `CFG.data_dir`
5. Tokenize in batched mode via `dataset.map()`; remove `"text"` column
6. Load pretrained model; log trainable parameter count
7. Build `TrainingArguments` from `CFG` fields
8. Check for existing checkpoint via `get_last_checkpoint(CFG.output_dir)`
9. Train (resume from checkpoint if found, else start fresh); log which path taken
10. Save final model + tokenizer to `CFG.final_dir`; log paths and total training time
11. Evaluate on validation set; compute and log perplexity (`math.exp(eval_loss)`)
12. Verify hidden states (in `try/except` — warns on failure, does not crash)

### SLURM scripts

Set resource flags (GPU, memory, CPU, time limit, log paths) and call `python <script>.py`. No logic beyond environment setup and echoing timestamps.

---

## Model-Specific Details

### BERT (`bert-base-uncased`)

- Task: Masked Language Modeling (`BertForMaskedLM`)
- Tokenizer: `BertTokenizer`, `max_length=256`, `return_special_tokens_mask=True`
- Data collator: `DataCollatorForLanguageModeling(mlm=True, mlm_probability=0.15)`
- LR scheduler: linear; LR: 2e-5; batch: 32 train / 64 eval
- Hidden state verification: load `BertModel` from final dir, assert 13 layers × 768 dims

### GPT-2 (`gpt2-medium`)

- Task: Causal Language Modeling (`GPT2LMHeadModel`)
- Tokenizer: `GPT2Tokenizer`, `max_length=512`, `pad_token = eos_token`; set `model.config.pad_token_id = tokenizer.eos_token_id`
- Labels: copy of `input_ids` with padding positions set to `-100`
- Data collator: `DataCollatorForLanguageModeling(mlm=False)`
- LR scheduler: cosine; LR: 5e-5; batch: 8 train / 16 eval; grad accum: 8
- Hidden state verification: load `GPT2Model` from final dir, assert 25 layers × 1024 dims

---

## Error Handling

| Scenario | Behavior |
|---|---|
| No `.jsonl` files in data dir | Raise `FileNotFoundError` before model loading (fail fast) |
| Hidden state verification fails | Log warning, continue — training already complete |
| Checkpoint found | Log "Resuming from checkpoint: \<path\>" and pass to `trainer.train()` |
| No checkpoint found | Log "Starting fresh training run" |

---

## Logging Conventions

- Use Python `logging` at `INFO` level throughout
- `print()` used only for the two final verification confirmation messages (per spec)
- Log: data loading progress, train/eval example counts, model parameter count, checkpoint status, each eval result, final perplexity

---

## What Is NOT Implemented

- No classification heads or task-specific fine-tuning
- No SAE code or activation collection beyond final verification
- No inference or generation code
- No LoRA/QLoRA or PEFT methods — full continued pretraining only
- No wandb/tensorboard — `report_to = none`
