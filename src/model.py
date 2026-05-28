"""XLM-RoBERTa backbone with a custom three-class NLI head."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import XLMRobertaModel
from transformers.modeling_outputs import SequenceClassifierOutput


DEFAULT_MODEL_NAME = "symanto/xlm-roberta-base-snli-mnli-anli-xnli"


@dataclass(frozen=True)
class HeadConfig:
    hidden_size: int = 768
    inter_size: int = 512
    dropout: float = 0.25
    num_labels: int = 3


class XLMRobertaNLIClassifier(nn.Module):
    """[CLS] pooling → (Dropout · Linear · LayerNorm · ReLU · Dropout · Linear)."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        head: HeadConfig = HeadConfig(),
    ) -> None:
        super().__init__()
        self.backbone = XLMRobertaModel.from_pretrained(model_name)
        self.head = nn.Sequential(
            nn.Dropout(head.dropout),
            nn.Linear(head.hidden_size, head.inter_size),
            nn.LayerNorm(head.inter_size),
            nn.ReLU(),
            nn.Dropout(head.dropout),
            nn.Linear(head.inter_size, head.num_labels),
        )
        self.num_labels = head.num_labels
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **_: object,
    ) -> SequenceClassifierOutput:
        backbone_out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls_repr = backbone_out.last_hidden_state[:, 0, :]
        logits = self.head(cls_repr)
        loss = self.loss_fn(logits, labels) if labels is not None else None
        return SequenceClassifierOutput(loss=loss, logits=logits)
