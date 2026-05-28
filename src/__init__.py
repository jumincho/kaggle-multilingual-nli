"""Multilingual NLI on the Kaggle "Contradictory, My Dear Watson" competition.

Sub-modules:
    src.data   — CSV loading, train/val split, tokenization.
    src.model  — XLM-RoBERTa backbone + (768 → 512 → 3) classification head.
    src.train  — fine-tuning + prediction pipeline (notebook and CLI entry).
"""

__all__ = ["data", "model", "train"]
