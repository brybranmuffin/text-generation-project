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
