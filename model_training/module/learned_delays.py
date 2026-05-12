import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from DCLS.construct.modules import Dcls1d


class SynapticDelayMixer(nn.Module):
    """
    Drop-in replacement for LaggedMixer.

    Input:  x of shape [T, B, C]
    Output: y of shape [T, B, C_out]

    Mirrors the synaptic-delay implementation style:
    - permute (T, B, C) -> (B, C, T)
    - temporal padding
    - apply Dcls1d delayed kernel
    - permute back to (T, B, C_out)

    Supports the same constructor pattern as LaggedMixer where practical,
    so it can be swapped in with minimal edits.
    """

    def __init__(
        self,
        channels,
        lags=(0, 1, 2, 4),
        one_random_lag_per_pair=False,   # kept for API compatibility, not used by DCLS directly
        num_lags_per_connection=1,       # kept for API compatibility, not used by DCLS directly
        lag_distribution_type="uniform", # kept for API compatibility, not used by DCLS directly
        lag_decay_strength=1.0,          # kept for API compatibility, not used by DCLS directly
        seed=0,                          # kept for API compatibility
        out_channels=None,
        bias=True,
        dcls_version="gauss",
        init_pos_method="uniform",
        init_pos_a=0.0,
        init_pos_b=None,
        sig_init=1.0,
        kernel_count=1,
        learn_positions=True,
        learn_sigma=False,
    ):
        super().__init__()

        self.channels = int(channels)
        self.out_channels = int(out_channels) if out_channels is not None else int(channels)
        self.lags = tuple(int(l) for l in lags)

        if len(self.lags) == 0:
            raise ValueError("lags must contain at least one delay.")

        if min(self.lags) < 0:
            raise ValueError("All lags must be >= 0.")

        # In the original DCLS code, dilated_kernel_size corresponds to max delay.
        # To include lag 0 ... max_lag cleanly, use max_lag + 1 as the effective temporal field.
        self.max_lag = max(self.lags)
        self.left_padding = self.max_lag
        self.right_padding = 0

        if init_pos_b is None:
            init_pos_b = float(self.max_lag)

        # Dcls1d operates on [B, C, T]
        self.mix = Dcls1d(
            in_channels=self.channels,
            out_channels=self.out_channels,
            kernel_count=int(kernel_count),
            groups=1,
            dilated_kernel_size=self.max_lag + 1,
            bias=bool(bias),
            version=dcls_version,
        )

        self.one_random_lag_per_pair = bool(one_random_lag_per_pair)
        self.num_lags_per_connection = int(num_lags_per_connection)
        self.lag_distribution_type = str(lag_distribution_type)
        self.lag_decay_strength = float(lag_decay_strength)
        self.seed = int(seed)

        self.learn_positions = bool(learn_positions)
        self.learn_sigma = bool(learn_sigma)

        self._init_delay_parameters(
            init_pos_method=init_pos_method,
            init_pos_a=float(init_pos_a),
            init_pos_b=float(init_pos_b),
            sig_init=float(sig_init),
        )

        self.register_buffer(
            "inference_lag_mask",
            torch.ones(self.max_lag + 1, dtype=torch.float32),
            persistent=False,
        )
        self.use_inference_lag_mask = False

    def _init_delay_parameters(self, init_pos_method, init_pos_a, init_pos_b, sig_init):
        with torch.no_grad():
            if init_pos_method == "uniform":
                nn.init.uniform_(self.mix.P, a=init_pos_a, b=init_pos_b)
            elif init_pos_method == "constant":
                nn.init.constant_(self.mix.P, init_pos_a)
            else:
                raise ValueError(f"Unsupported init_pos_method: {init_pos_method}")

            self.mix.clamp_parameters()
            nn.init.constant_(self.mix.SIG, sig_init)

        self.mix.P.requires_grad = self.learn_positions
        self.mix.SIG.requires_grad = self.learn_sigma

    def clamp_parameters(self):
        self.mix.clamp_parameters()

    def set_inference_lag_mask(self, lag_mask):
        lag_mask = torch.as_tensor(
            lag_mask,
            dtype=self.inference_lag_mask.dtype,
            device=self.inference_lag_mask.device,
        )
        expected = self.max_lag + 1
        if lag_mask.ndim != 1 or lag_mask.numel() != expected:
            raise ValueError(f"lag_mask must have shape ({expected},), got {tuple(lag_mask.shape)}")
        self.inference_lag_mask.copy_(lag_mask)
        self.use_inference_lag_mask = True

    def clear_inference_lag_mask(self):
        self.inference_lag_mask.fill_(1.0)
        self.use_inference_lag_mask = False

    def _apply_inference_lag_mask(self):
        """
        Approximate inference-time lag ablation by masking DCLS support locations.
        This is not identical to the old LaggedMixer column masking because DCLS
        parameterizes delays differently.
        """
        if self.training or (not self.use_inference_lag_mask):
            return

        with torch.no_grad():
            # Keep only positions whose rounded location corresponds to an enabled lag.
            rounded_p = self.mix.P.round().to(torch.long)
            valid = (rounded_p >= 0) & (rounded_p <= self.max_lag)
            enabled = torch.zeros_like(rounded_p, dtype=self.mix.weight.dtype)

            valid_idx = valid.nonzero(as_tuple=True)
            if len(valid_idx[0]) > 0:
                enabled_vals = self.inference_lag_mask.to(self.mix.weight.device)[rounded_p[valid]]
                enabled[valid] = enabled_vals

            self.mix.weight.mul_(enabled)

    def get_delay_params(self, discrete=False):
        P = self.mix.P.detach().clone()
        SIG = self.mix.SIG.detach().clone()
        W = self.mix.weight.detach().clone()

        if discrete:
            P = P.round()

        return {
            "weight": W,
            "P": P,
            "SIG": SIG,
        }

    def get_learned_parameters(self, discrete=False):
        return self.get_delay_params(discrete=discrete)

    def _tensor_summary(self, name, tensor):
        tensor_cpu = tensor.detach().cpu().float()
        flat = tensor_cpu.reshape(-1)
        sample = flat[: min(5, flat.numel())].tolist()
        return {
            "parameter": name,
            "shape": list(tensor_cpu.shape),
            "numel": int(flat.numel()),
            "min": float(flat.min().item()),
            "max": float(flat.max().item()),
            "mean": float(flat.mean().item()),
            "std": float(flat.std(unbiased=False).item()) if flat.numel() > 1 else 0.0,
            "sample_values": [float(value) for value in sample],
        }

    def get_parameter_summaries(self, discrete=False):
        params = self.get_delay_params(discrete=discrete)
        return [
            self._tensor_summary("P", params["P"]),
            self._tensor_summary("SIG", params["SIG"]),
            self._tensor_summary("weight", params["weight"]),
        ]

    def get_parameter_rows(
        self,
        epoch=None,
        step=None,
        event_name=None,
        block_idx=None,
        layer_name=None,
        discrete=False,
    ):
        params = self.get_delay_params(discrete=discrete)
        rows = []
        for parameter_name, tensor in params.items():
            summary = self._tensor_summary(parameter_name, tensor)
            rows.append(
                {
                    "epoch": None if epoch is None else int(epoch),
                    "step": None if step is None else int(step),
                    "event_name": event_name,
                    "block_idx": None if block_idx is None else int(block_idx),
                    "layer_name": layer_name,
                    "parameter": parameter_name,
                    "shape_json": json.dumps(summary["shape"]),
                    "numel": summary["numel"],
                    "min": summary["min"],
                    "max": summary["max"],
                    "mean": summary["mean"],
                    "std": summary["std"],
                    "values_json": json.dumps(tensor.detach().cpu().tolist()),
                }
            )
        return rows

    def save_snapshot(self, path, epoch=None, step=None, layer_name=None, store_full=True):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        snapshot = {
            "epoch": epoch,
            "step": step,
            "layer_name": layer_name,
            "lags": list(self.lags),
            "channels": int(self.channels),
            "out_channels": int(self.out_channels),
            "max_lag": int(self.max_lag),
            "left_padding": int(self.left_padding),
            "right_padding": int(self.right_padding),
            "learn_positions": bool(self.learn_positions),
            "learn_sigma": bool(self.learn_sigma),
        }

        if store_full:
            snapshot.update(self.get_delay_params(discrete=False))

        torch.save(snapshot, path)
        return snapshot

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(f"Expected x with shape [T, B, C], got {tuple(x.shape)}")

        T, B, C = x.shape
        if C != self.channels:
            raise ValueError(f"Expected input channels={self.channels}, got {C}")

        # [T, B, C] -> [B, C, T]
        x = x.permute(1, 2, 0)

        # same padding convention as your SnnDelays forward
        x = F.pad(x, (self.left_padding, self.right_padding), mode="constant", value=0.0)

        self._apply_inference_lag_mask()
        y = self.mix(x)

        # [B, C_out, T] -> [T, B, C_out]
        y = y.permute(2, 0, 1)
        return y
