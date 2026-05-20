# Congressional Speech LM Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement two SLURM-ready continued pretraining scripts (BERT MLM and GPT-2 CLM) for congressional speech data, with config dataclasses, checkpoint resumption, perplexity logging, and hidden state verification.

**Architecture:** Config-first — each model has a frozen dataclass (`config.py`) that holds all hyperparameters and paths; training scripts (`bert.py`, `gpt2.py`) import a single `CFG` constant and reference no magic numbers. SLURM scripts just set resources and call the Python script.

**Tech Stack:** Python 3.10+, PyTorch 2.0+, HuggingFace `transformers >= 4.35.0`, `datasets >= 2.14.0`, pytest

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `models/BERT/config.py` | Create | `BertTrainingConfig` frozen dataclass — all BERT paths, hyperparameters, verification constants |
| `models/GPT2/config.py` | Create | `GPT2TrainingConfig` frozen dataclass — all GPT-2 paths, hyperparameters, verification constants |
| `models/BERT/bert.py` | Modify (fill stub) | BERT MLM continued pretraining script |
| `models/GPT2/gpt2.py` | Modify (fill stub) | GPT-2 CLM continued pretraining script |
| `models/BERT/submit_bert.sh` | Create | SLURM job script for BERT |
| `models/GPT2/submit_gpt2.sh` | Create | SLURM job script for GPT-2 |
| `requirements.txt` | Create | Project-root dependency list |
| `tests/bert/test_config.py` | Create | Unit tests for `BertTrainingConfig` |
| `tests/gpt2/test_config.py` | Create | Unit tests for `GPT2TrainingConfig` |
| `tests/bert/test_bert_data.py` | Create | Unit tests for BERT data loading |
| `tests/gpt2/test_gpt2_data.py` | Create | Unit tests for GPT-2 data loading and label masking |

---

## Task 1: BERT Config

**Files:**
- Create: `models/BERT/config.py`
- Create: `tests/bert/__init__.py`
- Create: `tests/bert/test_config.py`

- [ ] **Step 1: Create test directory and write failing tests**

```bash
mkdir -p tests/bert tests/gpt2
touch tests/__init__.py tests/bert/__init__.py tests/gpt2/__init__.py
```

Create `tests/bert/test_config.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../models/BERT"))

import pytest
from config import BertTrainingConfig


def test_default_config_creates_successfully():
    cfg = BertTrainingConfig()
    assert cfg.model_name == "bert-base-uncased"
    assert cfg.max_length == 256
    assert cfg.mlm_probability == 0.15
    assert cfg.expected_hidden_layers == 13
    assert cfg.expected_hidden_dim == 768


def test_split_must_sum_to_one():
    with pytest.raises(ValueError, match="train_split"):
        BertTrainingConfig(train_split=0.8, eval_split=0.1)


def test_mlm_probability_must_be_between_zero_and_one():
    with pytest.raises(ValueError, match="mlm_probability"):
        BertTrainingConfig(mlm_probability=1.5)
    with pytest.raises(ValueError, match="mlm_probability"):
        BertTrainingConfig(mlm_probability=0.0)


def test_max_length_must_be_positive():
    with pytest.raises(ValueError, match="max_length"):
        BertTrainingConfig(max_length=0)


def test_learning_rate_must_be_positive():
    with pytest.raises(ValueError, match="learning_rate"):
        BertTrainingConfig(learning_rate=-1e-5)


def test_to_dict_returns_all_fields():
    cfg = BertTrainingConfig()
    d = cfg.to_dict()
    assert isinstance(d, dict)
    assert d["model_name"] == "bert-base-uncased"
    assert d["max_length"] == 256
    assert d["mlm_probability"] == 0.15


def test_log_config_logs_all_fields(caplog):
    import logging
    cfg = BertTrainingConfig()
    logger = logging.getLogger("test_bert_config")
    with caplog.at_level(logging.INFO, logger="test_bert_config"):
        cfg.log_config(logger)
    assert "model_name" in caplog.text
    assert "bert-base-uncased" in caplog.text
    assert "max_length" in caplog.text


def test_config_is_frozen():
    cfg = BertTrainingConfig()
    with pytest.raises(Exception):
        cfg.model_name = "something-else"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /path/to/text-generation-project
pytest tests/bert/test_config.py -v
```

Expected: All tests FAIL with `ModuleNotFoundError: No module named 'config'`

- [ ] **Step 3: Implement `models/BERT/config.py`**

```python
import dataclasses
import logging


@dataclasses.dataclass(frozen=True)
class BertTrainingConfig:
    data_dir: str = "/data/speeches"
    output_dir: str = "/models/bert"
    final_dir: str = "/models/bert/final"
    log_dir: str = "/models/bert/logs"

    model_name: str = "bert-base-uncased"

    max_length: int = 256
    mlm_probability: float = 0.15

    train_split: float = 0.99
    eval_split: float = 0.01
    shuffle_seed: int = 42

    num_train_epochs: int = 3
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 64
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    lr_scheduler_type: str = "linear"
    fp16: bool = True

    logging_steps: int = 500
    save_steps: int = 2000
    eval_steps: int = 2000
    save_total_limit: int = 3
    dataloader_num_workers: int = 4

    seed: int = 42

    expected_hidden_layers: int = 13
    expected_hidden_dim: int = 768

    def __post_init__(self) -> None:
        if not abs(self.train_split + self.eval_split - 1.0) < 1e-9:
            raise ValueError(
                f"train_split + eval_split must equal 1.0, got {self.train_split + self.eval_split}"
            )
        if not (0 < self.mlm_probability < 1):
            raise ValueError(
                f"mlm_probability must be between 0 and 1 exclusive, got {self.mlm_probability}"
            )
        if self.max_length <= 0:
            raise ValueError(f"max_length must be > 0, got {self.max_length}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def log_config(self, logger: logging.Logger) -> None:
        for key, value in self.to_dict().items():
            logger.info("  %s: %s", key, value)
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
pytest tests/bert/test_config.py -v
```

Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add models/BERT/config.py tests/__init__.py tests/bert/__init__.py tests/bert/test_config.py
git commit -m "feat: add BertTrainingConfig with validation and tests"
```

---

## Task 2: GPT-2 Config

**Files:**
- Create: `models/GPT2/config.py`
- Create: `tests/gpt2/test_config.py`

- [ ] **Step 1: Write failing tests**

Create `tests/gpt2/test_config.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../models/GPT2"))

import pytest
from config import GPT2TrainingConfig


def test_default_config_creates_successfully():
    cfg = GPT2TrainingConfig()
    assert cfg.model_name == "gpt2-medium"
    assert cfg.max_length == 512
    assert cfg.expected_hidden_layers == 25
    assert cfg.expected_hidden_dim == 1024


def test_split_must_sum_to_one():
    with pytest.raises(ValueError, match="train_split"):
        GPT2TrainingConfig(train_split=0.8, eval_split=0.1)


def test_max_length_must_be_positive():
    with pytest.raises(ValueError, match="max_length"):
        GPT2TrainingConfig(max_length=0)


def test_learning_rate_must_be_positive():
    with pytest.raises(ValueError, match="learning_rate"):
        GPT2TrainingConfig(learning_rate=0.0)


def test_to_dict_returns_all_fields():
    cfg = GPT2TrainingConfig()
    d = cfg.to_dict()
    assert isinstance(d, dict)
    assert d["model_name"] == "gpt2-medium"
    assert d["max_length"] == 512


def test_log_config_logs_all_fields(caplog):
    import logging
    cfg = GPT2TrainingConfig()
    logger = logging.getLogger("test_gpt2_config")
    with caplog.at_level(logging.INFO, logger="test_gpt2_config"):
        cfg.log_config(logger)
    assert "model_name" in caplog.text
    assert "gpt2-medium" in caplog.text


def test_config_is_frozen():
    cfg = GPT2TrainingConfig()
    with pytest.raises(Exception):
        cfg.model_name = "something-else"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/gpt2/test_config.py -v
```

Expected: All tests FAIL with `ModuleNotFoundError: No module named 'config'`

- [ ] **Step 3: Implement `models/GPT2/config.py`**

```python
import dataclasses
import logging


@dataclasses.dataclass(frozen=True)
class GPT2TrainingConfig:
    data_dir: str = "/data/speeches"
    output_dir: str = "/models/gpt2"
    final_dir: str = "/models/gpt2/final"
    log_dir: str = "/models/gpt2/logs"

    model_name: str = "gpt2-medium"

    max_length: int = 512

    train_split: float = 0.99
    eval_split: float = 0.01
    shuffle_seed: int = 42

    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-5
    weight_decay: float = 0.1
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    fp16: bool = True

    logging_steps: int = 500
    save_steps: int = 2000
    eval_steps: int = 2000
    save_total_limit: int = 3
    dataloader_num_workers: int = 4

    seed: int = 42

    expected_hidden_layers: int = 25
    expected_hidden_dim: int = 1024

    def __post_init__(self) -> None:
        if not abs(self.train_split + self.eval_split - 1.0) < 1e-9:
            raise ValueError(
                f"train_split + eval_split must equal 1.0, got {self.train_split + self.eval_split}"
            )
        if self.max_length <= 0:
            raise ValueError(f"max_length must be > 0, got {self.max_length}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def log_config(self, logger: logging.Logger) -> None:
        for key, value in self.to_dict().items():
            logger.info("  %s: %s", key, value)
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
pytest tests/gpt2/test_config.py -v
```

Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add models/GPT2/config.py tests/gpt2/__init__.py tests/gpt2/test_config.py
git commit -m "feat: add GPT2TrainingConfig with validation and tests"
```

---

## Task 3: BERT Training Script

**Files:**
- Modify: `models/BERT/bert.py`
- Create: `tests/bert/test_bert_data.py`

- [ ] **Step 1: Write failing data-loading tests**

Create `tests/bert/test_bert_data.py`:

```python
import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../models/BERT"))
from bert import load_dataset_from_jsonl


def test_load_raises_when_no_jsonl_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(FileNotFoundError, match="No .jsonl files"):
            load_dataset_from_jsonl(tmpdir)


def test_load_reads_single_jsonl_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        records = [
            {"text": "Mr. Speaker, the bill is passed.", "year": 2000, "party": "D"},
            {"text": "The senator objected to the motion.", "year": 2001, "party": "R"},
        ]
        with open(os.path.join(tmpdir, "speeches.jsonl"), "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        dataset = load_dataset_from_jsonl(tmpdir)
        assert len(dataset) == 2
        assert dataset[0]["text"] == records[0]["text"]


def test_load_reads_nested_jsonl_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        subdir = os.path.join(tmpdir, "1985")
        os.makedirs(subdir)
        record = {"text": "Congressional speech here.", "year": 1985, "party": "D"}
        with open(os.path.join(subdir, "nested.jsonl"), "w") as f:
            f.write(json.dumps(record) + "\n")
        dataset = load_dataset_from_jsonl(tmpdir)
        assert len(dataset) == 1
        assert dataset[0]["text"] == record["text"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/bert/test_bert_data.py -v
```

Expected: All tests FAIL — `bert.py` is an empty stub so `load_dataset_from_jsonl` does not exist.

- [ ] **Step 3: Implement `models/BERT/bert.py`**

```python
import glob
import json
import logging
import math
import os
import random
import time

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

    raw_dataset = load_dataset_from_jsonl(CFG.data_dir)
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
        overwrite_output_dir=False,
        num_train_epochs=CFG.num_train_epochs,
        per_device_train_batch_size=CFG.per_device_train_batch_size,
        per_device_eval_batch_size=CFG.per_device_eval_batch_size,
        gradient_accumulation_steps=CFG.gradient_accumulation_steps,
        learning_rate=CFG.learning_rate,
        weight_decay=CFG.weight_decay,
        warmup_ratio=CFG.warmup_ratio,
        lr_scheduler_type=CFG.lr_scheduler_type,
        fp16=CFG.fp16,
        logging_dir=CFG.log_dir,
        logging_steps=CFG.logging_steps,
        save_steps=CFG.save_steps,
        eval_steps=CFG.eval_steps,
        evaluation_strategy="steps",
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
        tokenizer=tokenizer,
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
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
pytest tests/bert/test_bert_data.py -v
```

Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add models/BERT/bert.py tests/bert/test_bert_data.py
git commit -m "feat: implement BERT MLM continued pretraining script"
```

---

## Task 4: GPT-2 Training Script

**Files:**
- Modify: `models/GPT2/gpt2.py`
- Create: `tests/gpt2/test_gpt2_data.py`

- [ ] **Step 1: Write failing tests**

Create `tests/gpt2/test_gpt2_data.py`:

```python
import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../models/GPT2"))
from gpt2 import load_dataset_from_jsonl, tokenize_batch


def test_load_raises_when_no_jsonl_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(FileNotFoundError, match="No .jsonl files"):
            load_dataset_from_jsonl(tmpdir)


def test_load_reads_single_jsonl_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        records = [
            {"text": "The house voted on the measure.", "year": 1999, "party": "R"},
        ]
        with open(os.path.join(tmpdir, "speeches.jsonl"), "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        dataset = load_dataset_from_jsonl(tmpdir)
        assert len(dataset) == 1
        assert dataset[0]["text"] == records[0]["text"]


def test_tokenize_batch_sets_padding_labels_to_minus100():
    """Labels at padding positions must be -100 so the loss ignores them."""

    class FakeTokenizer:
        pad_token_id = 0

        def __call__(self, texts, truncation, padding, max_length):
            return {
                "input_ids": [[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]],
                "attention_mask": [[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]],
            }

    result = tokenize_batch({"text": ["abc", "de"]}, FakeTokenizer(), max_length=5)
    assert result["labels"][0] == [1, 2, 3, -100, -100]
    assert result["labels"][1] == [4, 5, -100, -100, -100]


def test_tokenize_batch_non_padding_labels_match_input_ids():
    class FakeTokenizer:
        pad_token_id = 0

        def __call__(self, texts, truncation, padding, max_length):
            return {
                "input_ids": [[7, 8, 9]],
                "attention_mask": [[1, 1, 1]],
            }

    result = tokenize_batch({"text": ["xyz"]}, FakeTokenizer(), max_length=3)
    assert result["labels"][0] == [7, 8, 9]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/gpt2/test_gpt2_data.py -v
```

Expected: All tests FAIL — `gpt2.py` is an empty stub.

- [ ] **Step 3: Implement `models/GPT2/gpt2.py`**

```python
import glob
import json
import logging
import math
import os
import random
import time

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2Model,
    GPT2Tokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    trainer_utils,
)

from config import GPT2TrainingConfig

CFG = GPT2TrainingConfig()

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


def tokenize_batch(examples: dict, tokenizer: GPT2Tokenizer, max_length: int) -> dict:
    result = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    result["labels"] = [
        [-100 if token_id == tokenizer.pad_token_id else token_id for token_id in ids]
        for ids in result["input_ids"]
    ]
    return result


def main() -> None:
    set_seeds(CFG.seed)
    os.makedirs(CFG.output_dir, exist_ok=True)
    os.makedirs(CFG.log_dir, exist_ok=True)
    os.makedirs(CFG.final_dir, exist_ok=True)

    logger.info("=== GPT-2 Training Configuration ===")
    CFG.log_config(logger)

    raw_dataset = load_dataset_from_jsonl(CFG.data_dir)
    raw_dataset = raw_dataset.shuffle(seed=CFG.shuffle_seed)
    split = raw_dataset.train_test_split(test_size=CFG.eval_split, seed=CFG.shuffle_seed)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    logger.info("Train examples: %d, Eval examples: %d", len(train_dataset), len(eval_dataset))

    tokenizer = GPT2Tokenizer.from_pretrained(CFG.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    tokenize_fn = lambda examples: tokenize_batch(examples, tokenizer, CFG.max_length)
    train_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    eval_dataset = eval_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    logger.info("Tokenization complete")

    model = GPT2LMHeadModel.from_pretrained(CFG.model_name)
    model.config.pad_token_id = tokenizer.eos_token_id
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable parameters: %d", num_params)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=CFG.output_dir,
        overwrite_output_dir=False,
        num_train_epochs=CFG.num_train_epochs,
        per_device_train_batch_size=CFG.per_device_train_batch_size,
        per_device_eval_batch_size=CFG.per_device_eval_batch_size,
        gradient_accumulation_steps=CFG.gradient_accumulation_steps,
        learning_rate=CFG.learning_rate,
        weight_decay=CFG.weight_decay,
        warmup_ratio=CFG.warmup_ratio,
        lr_scheduler_type=CFG.lr_scheduler_type,
        fp16=CFG.fp16,
        logging_dir=CFG.log_dir,
        logging_steps=CFG.logging_steps,
        save_steps=CFG.save_steps,
        eval_steps=CFG.eval_steps,
        evaluation_strategy="steps",
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
        tokenizer=tokenizer,
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
        verify_tokenizer = GPT2Tokenizer.from_pretrained(CFG.final_dir)
        verify_tokenizer.pad_token = verify_tokenizer.eos_token
        verify_model = GPT2Model.from_pretrained(CFG.final_dir)
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
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
pytest tests/gpt2/test_gpt2_data.py -v
```

Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add models/GPT2/gpt2.py tests/gpt2/test_gpt2_data.py
git commit -m "feat: implement GPT-2 CLM continued pretraining script"
```

---

## Task 5: SLURM Job Scripts

**Files:**
- Create: `models/BERT/submit_bert.sh`
- Create: `models/GPT2/submit_gpt2.sh`

- [ ] **Step 1: Create `models/BERT/submit_bert.sh`**

```bash
#!/bin/bash
#SBATCH --job-name=congressional_bert
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=/models/bert/logs/slurm_%j.out
#SBATCH --error=/models/bert/logs/slurm_%j.err

mkdir -p /models/bert/logs

conda activate congressional_nlp

echo "Starting BERT training at $(date)"
echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi

python /models/bert/bert.py

echo "BERT training finished at $(date)"
```

Make it executable:
```bash
chmod +x models/BERT/submit_bert.sh
```

- [ ] **Step 2: Create `models/GPT2/submit_gpt2.sh`**

```bash
#!/bin/bash
#SBATCH --job-name=congressional_gpt2
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/models/gpt2/logs/slurm_%j.out
#SBATCH --error=/models/gpt2/logs/slurm_%j.err

mkdir -p /models/gpt2/logs

conda activate congressional_nlp

echo "Starting GPT-2 training at $(date)"
echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi

python /models/gpt2/gpt2.py

echo "GPT-2 training finished at $(date)"
```

Make it executable:
```bash
chmod +x models/GPT2/submit_gpt2.sh
```

- [ ] **Step 3: Verify SLURM script contents**

```bash
head -15 models/BERT/submit_bert.sh
head -15 models/GPT2/submit_gpt2.sh
```

Verify: BERT requests `--time=08:00:00` and `--mem=40G`; GPT-2 requests `--time=12:00:00` and `--mem=48G`.

- [ ] **Step 4: Commit**

```bash
git add models/BERT/submit_bert.sh models/GPT2/submit_gpt2.sh
git commit -m "feat: add SLURM job submission scripts for BERT and GPT-2"
```

---

## Task 6: Requirements File

**Files:**
- Create: `requirements.txt`

- [ ] **Step 1: Create `requirements.txt` at project root**

```
torch>=2.0.0
transformers>=4.35.0
datasets>=2.14.0
accelerate>=0.24.0
tokenizers>=0.14.0
numpy>=1.24.0
scikit-learn>=1.3.0
```

- [ ] **Step 2: Verify full test suite passes**

```bash
pytest tests/ -v
```

Expected: All tests PASS (22 tests total across 4 test files).

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "feat: add requirements.txt"
```
