"""Frozen BERT WordPiece-to-word embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
from typing import Sequence

import numpy as np


@dataclass
class BertWordEmbedder:
    """Lazy frozen BERT encoder that averages WordPieces belonging to each word."""

    model_name_or_path: str = "bert-base-uncased"
    device: str = "cpu"

    def __post_init__(self) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "BERT extraction requires: pip install -e '.[extract]'"
            ) from exc
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)
        if not self.tokenizer.is_fast:
            raise ValueError("a fast tokenizer is required for word_ids() mapping")
        self.model = AutoModel.from_pretrained(self.model_name_or_path).to(self.device)
        if int(self.model.config.hidden_size) != 768:
            raise ValueError(
                f"Problem 1 requires a 768-D BERT encoder; got hidden_size={self.model.config.hidden_size}"
            )
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.metadata = {
            "name": self.model_name_or_path,
            "transformers_version": version("transformers"),
            "torch_version": version("torch"),
            "hidden_size": int(self.model.config.hidden_size),
            "frozen": True,
            "wordpiece_pool": "arithmetic_mean",
        }

    def encode(self, words: Sequence[str]) -> np.ndarray:
        if not words:
            return np.zeros((0, int(self.model.config.hidden_size)), dtype=np.float32)
        encoded = self.tokenizer(
            list(words),
            is_split_into_words=True,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=False,
        )
        if encoded["input_ids"].shape[1] > int(self.model.config.max_position_embeddings):
            raise ValueError(
                "transcript exceeds BERT position limit; encode sentence chunks before alignment"
            )
        word_ids = encoded.word_ids(batch_index=0)
        inputs = {key: value.to(self.device) for key, value in encoded.items()}
        with self._torch.inference_mode():
            hidden = self.model(**inputs).last_hidden_state[0].detach().cpu().numpy()
        result = np.zeros((len(words), hidden.shape[-1]), dtype=np.float32)
        counts = np.zeros(len(words), dtype=np.int32)
        for token_index, word_index in enumerate(word_ids):
            if word_index is not None:
                result[word_index] += hidden[token_index]
                counts[word_index] += 1
        if np.any(counts == 0):
            missing = np.flatnonzero(counts == 0).tolist()
            raise ValueError(f"tokenizer produced no WordPieces for word indices {missing}")
        result /= counts[:, None]
        return result
