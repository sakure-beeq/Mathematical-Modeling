"""Missing-aware aligned multimodal sentiment model."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


MODALITIES = ("text", "audio", "vision")


class RobustFusion(nn.Module):
    def __init__(self, hidden: int = 128, heads: int = 4, dropout: float = 0.2,
                 fixed_gate: bool = False):
        super().__init__()
        self.fixed_gate = fixed_gate
        self.projections = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.GELU())
            for name, dim in zip(MODALITIES, (768, 74, 35))
        })
        self.missing = nn.Parameter(torch.zeros(3, hidden))
        self.position = nn.Embedding(50, hidden)
        self.modality = nn.Embedding(3, hidden)
        temporal = lambda: nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden, heads, hidden * 2, dropout,
                                       batch_first=True, norm_first=True), 2,
            enable_nested_tensor=False,
        )
        self.temporal = nn.ModuleDict({name: temporal() for name in MODALITIES})
        self.cross = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden, heads, hidden * 2, dropout,
                                       batch_first=True, norm_first=True), 1,
            enable_nested_tensor=False,
        )
        self.reconstruct = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(3)])
        self.uncertainty = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(3)])
        self.gate = nn.Sequential(nn.Linear(hidden + 3, hidden // 2), nn.GELU(),
                                  nn.Linear(hidden // 2, 1))
        self.pool = nn.Linear(hidden, 1)
        self.classifier = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(dropout),
                                        nn.Linear(hidden, 3))
        self.regressor = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(dropout),
                                       nn.Linear(hidden, 1))

    def forward(self, batch: dict[str, torch.Tensor],
                availability: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        # All masks have order text, audio, vision and shape (batch, 50, 3).
        padding = batch["padding"].bool()
        quality = batch["quality"].bool()
        natural = batch["availability"].bool()
        available = natural if availability is None else natural & availability.bool()
        observed = padding & quality & available
        b, t, _ = observed.shape
        pos = self.position(torch.arange(t, device=padding.device))[None]
        encoded = []
        targets = []
        for m, name in enumerate(MODALITIES):
            projected = self.projections[name](batch[name])
            targets.append(projected)
            inp = torch.where(observed[:, :, m, None], projected,
                              self.missing[m][None, None])
            inp = inp + pos + self.modality.weight[m][None, None]
            encoded.append(self.temporal[name](inp, src_key_padding_mask=~padding[:, :, m]))
        # (B,T,3,H): cross attention sees neighboring times and other modalities.
        sequence = torch.stack(encoded, dim=2).reshape(b, t * 3, -1)
        cross = self.cross(sequence, src_key_padding_mask=~padding.reshape(b, t * 3))
        cross = cross.reshape(b, t, 3, -1)
        recon = torch.stack([head(cross[:, :, m]) for m, head in enumerate(self.reconstruct)], dim=2)
        uncertainty = torch.cat([F.softplus(head(cross[:, :, m]))
                                 for m, head in enumerate(self.uncertainty)], dim=2)
        encoded = torch.stack(encoded, dim=2)
        combined = torch.where(observed[..., None], encoded, recon)
        gate_input = torch.cat((combined, observed[..., None].float(),
                                quality[..., None].float(), uncertainty[..., None]), dim=-1)
        logits = (torch.zeros_like(uncertainty) if self.fixed_gate else
                  self.gate(gate_input).squeeze(-1) - uncertainty)
        logits = logits.masked_fill(~padding, -1e4)
        weights = torch.softmax(logits, dim=2) * padding.float()
        weights = weights / weights.sum(dim=2, keepdim=True).clamp_min(1e-8)
        fused = (weights[..., None] * combined).sum(dim=2)
        valid_time = padding.any(dim=2)
        attention = self.pool(fused).squeeze(-1).masked_fill(~valid_time, -1e4)
        attention = torch.softmax(attention, dim=1)
        pooled = (attention[..., None] * fused).sum(dim=1)
        return {
            "logits": self.classifier(pooled),
            "score": 3 * torch.tanh(self.regressor(pooled).squeeze(-1)),
            "reconstruction": recon,
            "targets": torch.stack(targets, dim=2),
            "uncertainty": uncertainty,
            "gate_weights": weights,
            "observed": observed,
        }
