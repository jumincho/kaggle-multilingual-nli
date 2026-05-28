"""End-to-end training and prediction pipeline.

Importable from a notebook, runnable from the CLI:

    python -m src.train --epochs 5 --batch-size 16
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import evaluate
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

from .data import (
    DEFAULT_DATA_DIR,
    build_datasets,
    load_csvs,
    make_tokenizer,
    tokenize_datasets,
)
from .model import DEFAULT_MODEL_NAME, HeadConfig, XLMRobertaNLIClassifier


@dataclass
class TrainConfig:
    model_name: str = DEFAULT_MODEL_NAME
    data_dir: Path = DEFAULT_DATA_DIR
    output_dir: Path = Path("./outputs")
    submission_path: Path = Path("submission.csv")
    epochs: int = 5
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    val_size: float = 0.25
    seed: int = 42
    head: HeadConfig = field(default_factory=HeadConfig)


def _build_compute_metrics():
    f1 = evaluate.load("f1")

    def _compute(eval_preds):
        logits, labels = eval_preds
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1.compute(
                predictions=preds, references=labels, average="micro"
            )["f1"],
        }

    return _compute


def train_and_predict(cfg: TrainConfig) -> Path:
    """Fine-tune the classifier and write a Kaggle submission CSV.

    Returns the submission CSV path.
    """
    train_df, test_df = load_csvs(cfg.data_dir)
    ds = build_datasets(train_df, test_df, val_size=cfg.val_size, seed=cfg.seed)

    tokenizer = make_tokenizer(cfg.model_name)
    tokenized = tokenize_datasets(ds, tokenizer)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    args = TrainingArguments(
        output_dir=str(cfg.output_dir),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        optim="adamw_torch",
        eval_strategy="epoch",
        logging_steps=500,
        report_to="none",
        seed=cfg.seed,
    )

    model = XLMRobertaNLIClassifier(model_name=cfg.model_name, head=cfg.head)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=_build_compute_metrics(),
    )
    trainer.train()

    pred_out = trainer.predict(tokenized["test"])
    preds = pred_out.predictions.argmax(axis=-1)

    sample = pd.read_csv(cfg.data_dir / "sample_submission.csv")
    pd.DataFrame({"id": sample["id"], "prediction": preds}).to_csv(
        cfg.submission_path, index=False
    )
    return cfg.submission_path


def _parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(
        description="Fine-tune XLM-RoBERTa on the Contradictory, My Dear Watson NLI competition."
    )
    p.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--output-dir", type=Path, default=Path("./outputs"))
    p.add_argument("--submission-path", type=Path, default=Path("submission.csv"))
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, dest="learning_rate", default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--val-size", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)
    ns = p.parse_args()
    return TrainConfig(**vars(ns))


def main() -> None:
    out = train_and_predict(_parse_args())
    print(f"submission: {out}")


if __name__ == "__main__":
    main()
