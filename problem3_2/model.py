"""Interaction based multimodal predictor with explicit subset support.

This is a separate problem 3 model. It does not load problem 2 or the
additive problem 3 model's weights. Its observed-modality subsets are trained
so that Shapley interventions have an in-distribution interpretation.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

MODALITIES = ("text", "audio", "vision")
DIMS = (768, 74, 35)


class TemporalEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.depthwise = nn.Conv1d(hidden, hidden, 5, padding=2, groups=hidden)
        self.mix = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.attention = nn.Linear(hidden, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        mask = valid.unsqueeze(-1)
        state = self.project(x) * mask
        update = self.depthwise(state.transpose(1, 2)).transpose(1, 2)
        state = self.norm(state + self.dropout(F.gelu(self.mix(update)))) * mask
        scores = self.attention(state).squeeze(-1).masked_fill(~valid, -1e4)
        weights = torch.softmax(scores, dim=1) * valid.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return (weights.unsqueeze(-1) * state).sum(dim=1)


class InteractionModel(nn.Module):
    """Predictor with one inference call for predictions and explanations.

    ``forward`` supplies differentiable task outputs for training. After
    training, ``predict_with_explanation`` returns the full problem 3 result
    for one sample using this same predictor's counterfactual outputs.
    """

    def __init__(self, hidden: int = 96, dropout: float = 0.25):
        super().__init__()
        self.hidden = hidden
        self.dropout = dropout
        self.encoders = nn.ModuleDict({
            name: TemporalEncoder(dim, hidden, dropout)
            for name, dim in zip(MODALITIES, DIMS)
        })
        # Explicit products allow the fusion head to represent agreement and
        # conflict that an additive evidence model cannot represent directly.
        width = hidden * 7 + 3
        self.fusion = nn.Sequential(
            nn.Linear(width, hidden * 2), nn.LayerNorm(hidden * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden * 2, hidden), nn.GELU(),
        )
        self.classifier = nn.Linear(hidden, 3)
        self.regressor = nn.Linear(hidden, 1)

    def forward(self, batch: dict[str, torch.Tensor],
                availability: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        observed = (batch["padding"].bool() & batch["quality"].bool()
                    & batch["availability"].bool())
        if availability is not None:
            observed = observed & availability.bool()
        pooled = [self.encoders[name](batch[name], observed[:, :, m])
                  for m, name in enumerate(MODALITIES)]
        text, audio, vision = pooled
        proportions = observed.float().mean(dim=1)
        features = torch.cat((text, audio, vision, text * audio, text * vision,
                              audio * vision, text * audio * vision, proportions), dim=1)
        state = self.fusion(features)
        return {"logits": self.classifier(state),
                "score": 3.0 * torch.tanh(self.regressor(state).squeeze(-1)),
                "observed": observed}

    def predict_with_explanation(self, sample: dict[str, torch.Tensor],
                                 timeline: dict | None = None, *,
                                 selection_method: str = "occlusion",
                                 rng=None) -> dict:
        """Return class, intensity, modality effects, and located evidence.

        ``sample`` is one unbatched prepared example. ``timeline`` maps its
        aligned positions to transcript text and original-video seconds.
        The output is one explanation card; callers can serialize it to JSON
        or flatten it into the submission CSV.
        """
        from explain import explain_one

        self.eval()
        return explain_one(self, sample, next(self.parameters()).device,
                           timeline=timeline, rng=rng,
                           selection_method=selection_method)
