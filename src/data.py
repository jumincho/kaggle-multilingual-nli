"""Data loading, splitting, and tokenization for the multilingual NLI task."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, PreTrainedTokenizerBase


LABEL_NAMES = {0: "entailment", 1: "neutral", 2: "contradiction"}

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Columns dropped before training: a multilingual backbone consumes the raw
# text directly, so language metadata is informational only.
_META_COLUMNS = ("language", "lang_abv")


def load_csvs(data_dir: Path = DEFAULT_DATA_DIR) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load `train.csv` and `test.csv` as pandas DataFrames."""
    return (
        pd.read_csv(data_dir / "train.csv"),
        pd.read_csv(data_dir / "test.csv"),
    )


def build_datasets(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    val_size: float = 0.25,
    seed: int = 42,
) -> DatasetDict:
    """Split train into train/val and wrap everything as a `DatasetDict`."""
    train_df = train_df.drop(columns=list(_META_COLUMNS), errors="ignore")
    test_df = test_df.drop(columns=list(_META_COLUMNS), errors="ignore")

    train_split, val_split = train_test_split(
        train_df, test_size=val_size, random_state=seed
    )
    return DatasetDict(
        {
            "train": Dataset.from_pandas(train_split, preserve_index=False),
            "validation": Dataset.from_pandas(val_split, preserve_index=False),
            "test": Dataset.from_pandas(test_df, preserve_index=False),
        }
    )


def make_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(model_name)


def tokenize_datasets(
    ds: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
) -> DatasetDict:
    """Pair-encode `premise` and `hypothesis` as a single sequence."""

    def _encode(batch):
        return tokenizer(batch["premise"], batch["hypothesis"], truncation=True)

    return ds.map(_encode, batched=True)
