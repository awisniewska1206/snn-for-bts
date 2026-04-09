import torch
from torch import nn
import torch.nn.functional as F


class PhonemeTransitionBias(nn.Module):
    """
    Adds a small one-step transition bias to framewise logits using the previous
    frame's class distribution.

    The transition operates over all CTC classes, including blank at index 0.
    Timestep 0 receives no bias.
    """

    def __init__(
        self,
        n_classes: int,
        rank: int = 8,
        coefficient: float = 0.05,
        mode: str = "low_rank",
        detach_prev: bool = True,
    ):
        super().__init__()
        self.n_classes = int(n_classes)
        self.rank = int(rank)
        self.coefficient = float(coefficient)
        self.mode = str(mode).lower()
        self.detach_prev = bool(detach_prev)

        if self.mode == "low_rank":
            self.transition_left = nn.Parameter(torch.zeros(self.n_classes, self.rank))
            self.transition_right = nn.Parameter(torch.zeros(self.rank, self.n_classes))
        elif self.mode == "full":
            self.transition = nn.Parameter(torch.zeros(self.n_classes, self.n_classes))
        else:
            raise ValueError(
                f"Unsupported phoneme transition bias mode: {mode}. "
                "Expected 'low_rank' or 'full'."
            )

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.size(1) <= 1 or self.coefficient == 0.0:
            return logits

        prev_probs = F.softmax(logits[:, :-1, :], dim=-1)
        if self.detach_prev:
            prev_probs = prev_probs.detach()

        if self.mode == "low_rank":
            bias = prev_probs.matmul(self.transition_left).matmul(self.transition_right)
        else:
            bias = prev_probs.matmul(self.transition)

        padded_bias = torch.zeros_like(logits)
        padded_bias[:, 1:, :] = bias
        return logits + (self.coefficient * padded_bias)


def build_phoneme_transition_bias(n_classes: int, cfg):
    if cfg is None:
        return None

    enabled = bool(cfg.get("enabled", False))
    if not enabled:
        return None

    return PhonemeTransitionBias(
        n_classes=n_classes,
        rank=cfg.get("rank", 8),
        coefficient=cfg.get("coefficient", 0.05),
        mode=cfg.get("mode", "low_rank"),
        detach_prev=cfg.get("detach_prev", True),
    )
