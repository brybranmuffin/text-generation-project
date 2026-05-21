import dataclasses
import logging


@dataclasses.dataclass(frozen=True)
class GPT2TrainingConfig:
    data_dir: str = "./data/raw_data/filtered_speeches.csv"
    data_type: str = "csv" # csv or jsonl
    output_dir: str = "./outputs/gpt2"
    final_dir: str = "./outputs/gpt2/final"
    log_dir: str = "./outputs/gpt2/logs"

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
