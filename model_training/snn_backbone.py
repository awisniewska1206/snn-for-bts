import copy
import math
import torch
import torch.nn as nn
from spikingjelly.activation_based import functional as sj_functional

from snn_modules_alternative_core import SpikingNeuralNet
from models.phoneme_transition_bias import build_phoneme_transition_bias

class SpikingModel(nn.Module):
    """
    Wrapper that only does:
    - day transform
    - optional patching
    - call original SpikingNeuralNet

    No extra head. No behavior change.
    """
    def __init__(
        self,
        cfg,
        neural_dim: int,
        n_days: int,
        n_classes: int,
        input_dropout: float = 0.0,
        patch_size: int = 0,
        patch_stride: int = 0,
        debug: bool = False,
        collect_day_stats: bool = False,
        stats_every: int = 1000,
        session_adaptation_cfg = None,
        phoneme_transition_bias_cfg = None,
    ):
        super().__init__()

        self.cfg = cfg
        self.neural_dim = neural_dim
        self.n_days = n_days
        self.n_classes = n_classes
        self.input_dropout = input_dropout
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.debug = debug
        self.stats_every = stats_every
        self._step = 0
        self.session_adaptation_cfg = session_adaptation_cfg
        self.phoneme_transition_bias = build_phoneme_transition_bias(
            n_classes=n_classes,
            cfg=phoneme_transition_bias_cfg,
        )

        self.day_layer_activation = nn.Softsign()
        self.day_weights = nn.ParameterList(
            [nn.Parameter(torch.eye(self.neural_dim)) for _ in range(self.n_days)]
        )
        self.day_biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.neural_dim)) for _ in range(self.n_days)]
        )
        self.day_layer_dropout = nn.Dropout(input_dropout)

        self.last_day_stats = {}
        self.last_matrix_stats = {}

        backbone_cfg = copy.deepcopy(cfg)

        if self.patch_size > 0:
            patched_input_size = int(self.neural_dim * self.patch_size)

        self.backbone = SpikingNeuralNet(backbone_cfg)

    def apply_day_transform(self, x, day_idx):
        day_weights = torch.stack([self.day_weights[i] for i in day_idx], dim=0)
        day_biases = torch.cat([self.day_biases[i] for i in day_idx], dim=0).unsqueeze(1)

        x = torch.einsum("btd,bdk->btk", x, day_weights) + day_biases
        x = self.day_layer_activation(x)

        if self.input_dropout > 0:
            x = self.day_layer_dropout(x)

        return x

    def apply_patching(self, x):
        assert self.patch_size > 0 and self.patch_stride > 0

        x = x.unsqueeze(1)                    # [B,1,T,D]
        x = x.permute(0, 3, 1, 2)            # [B,D,1,T]
        x_unfold = x.unfold(3, self.patch_size, self.patch_stride)
        x_unfold = x_unfold.squeeze(2)       # [B,D,T',patch]
        x_unfold = x_unfold.permute(0, 2, 3, 1)  # [B,T',patch,D]
        x = x_unfold.reshape(x.size(0), x_unfold.size(1), -1)
        return x

    def forward(self, x, day_idx, session_summary=None):
        """
        x: [B, T, neural_dim]
        returns: [B, T', n_classes]
        """
        self._step += 1

        x = self.apply_day_transform(x, day_idx)

        if self.patch_size > 0:
            x = self.apply_patching(x)

        sj_functional.reset_net(self.backbone)

        x = x.permute(1, 0, 2).contiguous()   # [T,B,C]
        logits = self.backbone(x)             # [T,B,Cout]
        logits = logits.permute(1, 0, 2).contiguous()
        if self.phoneme_transition_bias is not None:
            logits = self.phoneme_transition_bias(logits)

        return logits

    def get_spiking_layer_statistics(self):
        stats = {}
        for name, module in self.named_modules():
            layer_stats = getattr(module, "last_forward_stats", None)
            if layer_stats:
                stats[name] = dict(layer_stats)
        return stats

    def get_spiking_gradient_statistics(self):
        grad_stats = {}
        for name, module in self.named_modules():
            layer_stats = getattr(module, "last_forward_stats", None)
            if not layer_stats:
                continue

            grad_sq_sum = 0.0
            has_grad = False
            for param in module.parameters(recurse=False):
                if param.grad is None:
                    continue
                grad_sq_sum += float(param.grad.detach().float().pow(2).sum().item())
                has_grad = True

            grad_stats[name] = math.sqrt(grad_sq_sum) if has_grad else None

        return grad_stats

    def save_lag_selection_snapshots(self, output_dir):
        if not hasattr(self.backbone, "save_lag_selection_snapshots"):
            return []
        return self.backbone.save_lag_selection_snapshots(output_dir)

    def get_lag_values(self, block_idx):
        if not hasattr(self.backbone, "get_lag_values"):
            raise AttributeError("Backbone does not expose lag values")
        return self.backbone.get_lag_values(block_idx)

    def set_inference_lag_mask(self, block_idx, lag_mask):
        if not hasattr(self.backbone, "set_inference_lag_mask"):
            raise AttributeError("Backbone does not support inference lag masking")
        self.backbone.set_inference_lag_mask(block_idx, lag_mask)

    def clear_inference_lag_masks(self):
        if not hasattr(self.backbone, "clear_inference_lag_masks"):
            return
        self.backbone.clear_inference_lag_masks()

    def get_global_neuron_parameters(self):
        if not hasattr(self.backbone, "get_global_neuron_parameters"):
            return {}
        return self.backbone.get_global_neuron_parameters()
