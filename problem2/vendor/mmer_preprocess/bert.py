"""Optional frozen-BERT embedding generation.

Torch and Transformers are optional so that mask/statistics preprocessing can
run in a lightweight environment. Install the project with the ``bert`` extra
before enabling this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FrozenBertEmbedder:
    model_name: str
    device: str | None = None
    batch_size: int = 32

    def __post_init__(self) -> None:
        try:
            import torch
            from transformers import AutoModel
        except ImportError as exc:
            raise RuntimeError(
                "BERT embedding requires the optional dependencies. "
                "Install with: pip install -e '.[bert]'"
            ) from exc

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._torch = torch
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.eval()
        self._model.to(self.device)
        for parameter in self._model.parameters():
            parameter.requires_grad_(False)

    def encode(self, text_bert: np.ndarray) -> np.ndarray:
        """Encode ``(N,3,T)`` BERT inputs to ``(N,T,H)`` float32 vectors."""

        values = np.asarray(text_bert)
        if values.ndim != 3 or values.shape[1] != 3:
            raise ValueError(f"text_bert must have shape (N,3,T); got {values.shape}")
        values = values.astype(np.int64, copy=False)
        outputs: list[np.ndarray] = []
        torch = self._torch
        with torch.inference_mode():
            for start in range(0, values.shape[0], self.batch_size):
                batch = values[start : start + self.batch_size]
                input_ids = torch.as_tensor(batch[:, 0, :], device=self.device)
                attention_mask = torch.as_tensor(batch[:, 1, :], device=self.device)
                token_type_ids = torch.as_tensor(batch[:, 2, :], device=self.device)
                hidden = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                ).last_hidden_state
                hidden = hidden * attention_mask.unsqueeze(-1)
                outputs.append(hidden.cpu().numpy().astype(np.float32, copy=False))
        return np.concatenate(outputs, axis=0)
