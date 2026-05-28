import argparse
import csv
import json
import logging
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from transformers import GPT2Model, GPT2Tokenizer

DEFAULT_GPT2_MODEL_DIR = "./outputs/gpt2/final"
DEFAULT_DATA_FILE      = "./data/raw_data/filtered_speeches.csv"
DEFAULT_OUTPUT_DIR     = "./activations"

CSV_COLUMNS = [
    "speech_id", "text", "speaker", "bioguide_id", "date",
    "last_name", "first_name", "middle_name", "gender", "party", "chamber", "year",
]

csv.field_size_limit(sys.maxsize)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def validate_paths(args: argparse.Namespace) -> None:
    errors = []

    if not os.path.isfile(args.data_file):
        errors.append(f"Data file not found:  {args.data_file}")

    if not os.path.isdir(args.model_dir):
        errors.append(f"Model directory not found:  {args.model_dir}")
    else:
        for required in ("config.json", "tokenizer_config.json"):
            if not os.path.isfile(os.path.join(args.model_dir, required)):
                errors.append(f"Missing model file:  {os.path.join(args.model_dir, required)}")

    try:
        os.makedirs(args.output_dir, exist_ok=True)
    except OSError as exc:
        errors.append(f"Cannot create output directory {args.output_dir}: {exc}")

    if errors:
        for msg in errors:
            logger.error(msg)
        raise SystemExit("Path validation failed — fix the above errors before running.")


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect GPT-2 hidden state activations.")
    parser.add_argument("--model_dir",     type=str,  default=DEFAULT_GPT2_MODEL_DIR)
    parser.add_argument("--n_speeches",    type=int,  default=50000)
    parser.add_argument("--layers",        type=str,  default="last",
                        help='"last", "all", or comma-separated indices e.g. "12,18,24"')
    parser.add_argument("--batch_size",    type=int,  default=32)
    parser.add_argument("--output_dir",    type=str,  default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data_file",     type=str,  default=DEFAULT_DATA_FILE)
    parser.add_argument("--seed",          type=int,  default=42)
    parser.add_argument("--save_metadata", type=lambda x: x.lower() != "false", default=True)
    return parser.parse_args()


def load_csv(data_file: str) -> pd.DataFrame:
    with open(data_file, "r", encoding="utf-8", errors="replace") as fh:
        first = fh.readline()
    has_header = first.lstrip().startswith("speech_id") or "text" in first.split(",")[:3]

    df = pd.read_csv(
        data_file,
        header=0 if has_header else None,
        names=None if has_header else CSV_COLUMNS,
        usecols=["text", "party", "year"],
        dtype=str,
        engine="python",
    )
    return df


def decade_stratified_sample(data_file: str, n_speeches: int, seed: int) -> list[dict]:
    logger.info("Loading speeches from %s", data_file)
    df = load_csv(data_file)

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year", "text"])
    df = df[df["year"].between(1870, 2030)]
    df["decade"] = ((df["year"] // 10) * 10).astype(int)

    total = len(df)
    logger.info("Total valid speeches: %d", total)

    if total < n_speeches * 0.5:
        raise ValueError(
            f"Only {total} valid speeches found — corpus may be too small "
            f"for {n_speeches} samples (need at least {int(n_speeches * 0.5)})."
        )

    decade_counts = df.groupby("decade").size()
    allocations = {
        int(decade): round(n_speeches * (count / total))
        for decade, count in decade_counts.items()
    }
    diff = n_speeches - sum(allocations.values())
    if diff != 0:
        largest = max(allocations, key=allocations.__getitem__)
        allocations[largest] += diff

    rng = random.Random(seed)
    sampled: list[dict] = []

    logger.info("Decade sampling summary:")
    for decade in sorted(allocations):
        pool   = df[df["decade"] == decade].to_dict("records")
        n_alloc = allocations[decade]
        if len(pool) < n_alloc:
            logger.warning(
                "  %ds: only %d available, requested %d — taking all",
                decade, len(pool), n_alloc,
            )
            n_alloc = len(pool)
        chosen = rng.sample(pool, n_alloc)
        sampled.extend(chosen)
        pct = len(pool) / total * 100
        logger.info("  %ds: %7d total -> %6d sampled  (%.1f%%)", decade, len(pool), n_alloc, pct)

    logger.info("  %s", "─" * 45)
    logger.info("  Total: %d sampled from %d available", len(sampled), total)

    rng.shuffle(sampled)
    return sampled


def resolve_layer_indices(layers_arg: str, n_hidden_states: int) -> list[int]:
    if layers_arg == "last":
        return [-1]
    if layers_arg == "all":
        return list(range(n_hidden_states))
    return [int(x) for x in layers_arg.split(",")]


def layer_tag(layers_arg: str, idx: int) -> str:
    return "last" if layers_arg == "last" else f"layer{idx}"


def main() -> None:
    args = parse_args()
    validate_paths(args)
    set_seeds(args.seed)

    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    logger.info(
        "Note: GPT-2 activations are causal; mean pool reflects "
        "left-context representations averaged across token positions."
    )

    speeches = decade_stratified_sample(args.data_file, args.n_speeches, args.seed)
    n_total  = len(speeches)

    logger.info("Loading GPT-2 from %s", args.model_dir)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_dir)
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2Model.from_pretrained(args.model_dir, output_hidden_states=True)
    model.config.pad_token_id = tokenizer.eos_token_id
    model.eval()
    model = model.to(device)
    logger.info("Parameters: %d", sum(p.numel() for p in model.parameters()))

    n_hidden_states = model.config.n_layer + 1  # 25 for gpt2-medium
    layer_indices   = resolve_layer_indices(args.layers, n_hidden_states)

    accumulators: dict[int, list[np.ndarray]] = {idx: [] for idx in layer_indices}
    metadata:     list[dict]                   = []
    n_processed   = 0
    n_skipped     = 0
    prev_milestone = 0

    for batch_start in range(0, n_total, args.batch_size):
        batch = speeches[batch_start: batch_start + args.batch_size]
        texts = [str(r.get("text", "")) for r in batch]

        try:
            encoded = tokenizer(
                texts,
                max_length=512,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            with torch.no_grad():
                outputs = model(
                    input_ids=encoded["input_ids"].to(device),
                    attention_mask=encoded["attention_mask"].to(device),
                    output_hidden_states=True,
                )

            if batch_start == 0 and torch.cuda.is_available():
                mem_gb = torch.cuda.memory_allocated() / 1e9
                logger.info("GPU memory after first batch: %.2f GB", mem_gb)

            mask = encoded["attention_mask"].unsqueeze(-1).float().to(device)

            for idx in layer_indices:
                hidden = outputs.hidden_states[idx]       # (B, seq, 1024)
                summed = (hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1)
                pooled = summed / counts                  # (B, 1024)
                accumulators[idx].append(pooled.float().cpu().numpy())

            base_idx = n_processed
            for i, r in enumerate(batch):
                yr = r.get("year")
                metadata.append({
                    "index":        base_idx + i,
                    "year":         int(float(yr)) if yr not in (None, "") else None,
                    "party":        r.get("party"),
                    "decade":       int((float(yr) // 10) * 10) if yr not in (None, "") else None,
                    "text_preview": str(r.get("text", ""))[:100],
                })

            n_processed += len(batch)

            del outputs, mask
            torch.cuda.empty_cache()

        except Exception as exc:
            logger.warning("Batch at offset %d failed: %s — skipping", batch_start, exc)
            n_skipped += len(batch)
            continue

        milestone = (n_processed // 5000) * 5000
        if milestone > prev_milestone:
            logger.info("Processed %d/%d speeches", n_processed, n_total)
            prev_milestone = milestone

    logger.info("Complete — processed: %d  skipped: %d", n_processed, n_skipped)

    # Save activations
    first_path = None
    for idx in layer_indices:
        arr      = np.vstack(accumulators[idx])
        tag      = layer_tag(args.layers, idx)
        filename = f"gpt2_{tag}_{args.n_speeches}.npy"
        out_path = os.path.join(args.output_dir, filename)
        np.save(out_path, arr)
        logger.info("Saved %s  shape=%s  dtype=%s", out_path, arr.shape, arr.dtype)
        if first_path is None:
            first_path = out_path

    # Save metadata
    if args.save_metadata:
        meta_path = os.path.join(args.output_dir, f"gpt2_metadata_{args.n_speeches}.json")
        with open(meta_path, "w") as fh:
            json.dump(metadata, fh)
        logger.info("Saved metadata: %s", meta_path)

    # Verify
    arr = np.load(first_path)
    assert arr.shape[1] == 1024,        f"Expected dim 1024, got {arr.shape[1]}"
    assert not np.isnan(arr).any(),     "NaN values found in activations"
    assert not np.isinf(arr).any(),     "Inf values found in activations"
    assert arr.shape[0] == n_processed, f"Row count mismatch: {arr.shape[0]} vs {n_processed}"
    print(f"GPT-2 activation verification passed: shape {arr.shape}")

    elapsed = time.time() - start_time
    h, rem  = divmod(elapsed, 3600)
    m, s    = divmod(rem, 60)
    logger.info("Total time: %dh %dm %ds", int(h), int(m), int(s))


if __name__ == "__main__":
    main()
