import numpy as np
import torch
import torch.nn as nn
import hydra
from einops.layers.torch import Rearrange
from omegaconf import DictConfig
from spikingjelly.activation_based import functional
from spikingjelly.activation_based import layer, surrogate
import torch.nn.functional as F

from models.neurons import LTLIFNode, MultiTauLIFNode
import pandas as pd


def pick_surrogate(surr_name: str):
    if surr_name == "atan":
        return surrogate.ATan()
    if surr_name == "sigmoid":
        return surrogate.Sigmoid()
    raise ValueError(
        f"Unsupported surrogate function: {surr_name}. "
        "Expected one of {'atan', 'sigmoid'}."
    )

class LaggedMixer(nn.Module):
    def __init__(
        self,
        channels,
        lags=(0, 1, 2, 4),
        one_random_lag_per_pair=False,
        seed=0,
    ):
        super().__init__()
        self.lags = list(lags)
        self.channels = channels
        self.K = len(self.lags)

        self.mix = nn.Linear(channels * self.K, channels)

        print(lags)

        # ---- create mask (reproducible) ----
        g = torch.Generator()
        g.manual_seed(seed)

        mask = torch.ones(channels, channels * self.K)

        if one_random_lag_per_pair:
            mask.zero_()
            for out_ch in range(channels):
                for in_ch in range(channels):
                    k_idx = torch.randint(0, self.K, (1,), generator=g).item()
                    col = k_idx * channels + in_ch
                    mask[out_ch, col] = 1.0

        self.register_buffer("mask", mask)

        # ---- zero masked weights ----
        with torch.no_grad():
            self.mix.weight.mul_(self.mask)

        # ---- mask gradients ----
        self.mix.weight.register_hook(lambda grad: grad * self.mask)

    def forward(self, x):  # [T, B, C]
        T, B, C = x.shape

        xs = []
        for lag in self.lags:
            if lag == 0:
                xs.append(x)
            else:
                pad = x.new_zeros(lag, B, C)
                xs.append(torch.cat([pad, x[:-lag]], dim=0))

        x_cat = torch.cat(xs, dim=-1)  # [T, B, C*K]

        weight = self.mix.weight * self.mask
        return F.linear(x_cat, weight, self.mix.bias)

    def export_weights(self, path):
        W = (self.mix.weight * self.mask).detach().cpu()
        C = W.shape[0]
        K = self.K

        W = W.view(C, K, C).permute(0, 2, 1).contiguous()  # [out, in, lag]

        rows = []
        for i in range(C):
            for j in range(C):
                for k_idx, lag in enumerate(self.lags):
                    rows.append({
                        "out_channel": i,
                        "in_channel": j,
                        "lag": lag,
                        "weight": W[i, j, k_idx].item(),
                    })

        pd.DataFrame(rows).to_csv(path, index=False)

'''class LaggedMixer(nn.Module):
    def __init__(
        self,
        channels,
        lags=(0, 1, 2, 4),
        learnable_lag_gates=False,
        gate_init="uniform",
        gate_temperature=1.0,
    ):
        super().__init__()
        self.lags = list(lags)
        self.channels = channels
        self.K = len(self.lags)
        self.learnable_lag_gates = learnable_lag_gates
        self.gate_temperature = gate_temperature

        if self.learnable_lag_gates:
            # one base weight per (out, in)
            self.pair_weight = nn.Parameter(torch.randn(channels, channels) * 0.02)
            self.bias = nn.Parameter(torch.zeros(channels))

            if gate_init == "uniform":
                init_logits = torch.zeros(channels, channels, self.K)
            elif gate_init == "short_bias":
                base = torch.tensor([-0.3 * i for i in range(self.K)], dtype=torch.float32)
                init_logits = base.view(1, 1, self.K).repeat(channels, channels, 1)
            elif gate_init == "random":
                init_logits = 0.01 * torch.randn(channels, channels, self.K)
            else:
                raise ValueError(f"Unknown gate_init: {gate_init}")

            self.lag_logits = nn.Parameter(init_logits)

        else:
            self.mix = nn.Linear(channels * self.K, channels)

    def _build_lagged_inputs(self, x):
        T, B, C = x.shape
        xs = []
        for lag in self.lags:
            if lag == 0:
                xs.append(x)
            else:
                pad = x.new_zeros(lag, B, C)
                xs.append(torch.cat([pad, x[:-lag]], dim=0))
        return torch.cat(xs, dim=-1)  # [T, B, C*K]

    def get_lag_probs(self):
        return F.softmax(self.lag_logits / self.gate_temperature, dim=-1)

    def forward(self, x):
        x_cat = self._build_lagged_inputs(x)  # [T,B,C*K]

        if self.learnable_lag_gates:
            lag_probs = self.get_lag_probs()                     # [out,in,K]
            eff_weight = self.pair_weight.unsqueeze(-1) * lag_probs
            eff_weight = eff_weight.permute(0, 2, 1).contiguous().view(self.channels, self.channels * self.K)
            return F.linear(x_cat, eff_weight, self.bias)
        else:
            return self.mix(x_cat)
            
    def export_weights(self, path):
        rows = []

        if self.learnable_lag_gates:
            lag_probs = self.get_lag_probs().detach().cpu()                 # [out,in,lag]
            eff_weight = (self.pair_weight.unsqueeze(-1) * self.get_lag_probs()).detach().cpu()
            # [out,in,lag]

            for i in range(self.channels):
                for j in range(self.channels):
                    best_k = int(lag_probs[i, j].argmax().item())
                    for k_idx, lag in enumerate(self.lags):
                        rows.append({
                            "out_channel": i,
                            "in_channel": j,
                            "lag": lag,
                            "pair_weight": self.pair_weight[i, j].detach().cpu().item(),
                            "lag_prob": lag_probs[i, j, k_idx].item(),
                            "effective_weight": eff_weight[i, j, k_idx].item(),
                            "best_lag_for_pair": self.lags[best_k],
                        })

        else:
            W = self.mix.weight.detach().cpu()
            W = W.view(self.channels, self.K, self.channels).permute(0, 2, 1).contiguous()  # [out,in,lag]

            for i in range(self.channels):
                for j in range(self.channels):
                    best_k = int(W[i, j].abs().argmax().item())
                    for k_idx, lag in enumerate(self.lags):
                        rows.append({
                            "out_channel": i,
                            "in_channel": j,
                            "lag": lag,
                            "effective_weight": W[i, j, k_idx].item(),
                            "best_lag_for_pair": self.lags[best_k],
                        })

        pd.DataFrame(rows).to_csv(path, index=False)


    def export_learned_lag_summary(self, path):
        if not self.learnable_lag_gates:
            raise RuntimeError("Only valid when learnable_lag_gates=True")

        lag_probs = self.get_lag_probs().detach().cpu()   # [out,in,lag]
        eff_weight = (self.pair_weight.unsqueeze(-1) * self.get_lag_probs()).detach().cpu()
        best_idx = lag_probs.argmax(dim=-1)               # [out,in]

        rows = []
        for i in range(self.channels):
            for j in range(self.channels):
                row = {
                    "out_channel": i,
                    "in_channel": j,
                    "pair_weight": self.pair_weight[i, j].detach().cpu().item(),
                    "best_lag": self.lags[int(best_idx[i, j].item())],
                }
                for k_idx, lag in enumerate(self.lags):
                    row[f"p_lag_{lag}"] = lag_probs[i, j, k_idx].item()
                    row[f"w_lag_{lag}"] = eff_weight[i, j, k_idx].item()
                rows.append(row)

        pd.DataFrame(rows).to_csv(path, index=False)
'''
class ResidualSNNBlock(nn.Module):
    def __init__(
        self,
        tau: float,
        v_threshold: float,
        v_reset: float,
        channels: int,
        p_drop: float = 0.0,
        surrogate_func: str = "atan",
        backend: str = "cupy",
    ):
        super().__init__()

        surr = pick_surrogate(surrogate_func)

        self.lagged = LaggedMixer(
            channels,
            lags=(0, 1, 2, 3, 4, 5, 6, 7),
            one_random_lag_per_pair=True,
            seed=10
        )
        self.neuron = LTLIFNode(
            init_tau=tau,
            surrogate_function=surr,
            step_mode="m",
            backend=backend,
            v_threshold=v_threshold,
            v_reset=v_reset,
            decay_input=False,
        )
        self.proj = layer.Linear(channels, channels, step_mode="m")
        self.bn_in = Rearrange("t b c -> t b c 1")
        self.bn = layer.BatchNorm1d(channels, step_mode="m")
        self.bn_out = Rearrange("t b c 1 -> t b c")
        self.drop = layer.Dropout(p=p_drop, step_mode="m")

        self.last_spikes = None

    def forward(self, x):
        residual = x

        z = self.lagged(x)
        s = self.neuron(z)
        self.last_spikes = s

        z = self.proj(s)
        z = self.bn_in(z)
        z = self.bn(z)
        z = self.bn_out(z)
        z = self.drop(z)

        return z + residual

    def get_spike_params(self):
        net_params = []
        if isinstance(self.neuron, MultiTauLIFNode):
            net_params.append(self.neuron.get_spike_params())
        return net_params

class SpikingNeuralNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        setup = cfg.model.setup
        self._step = 0
        hidden_size = setup.hidden_size

        self.input_proj = nn.Linear(setup.input_size * setup.patch_size, hidden_size)

        self.snn_core = nn.ModuleList([
            ResidualSNNBlock(
                tau=setup.snn_core.tau,
                v_threshold=setup.snn_core.v_threshold,
                v_reset=setup.snn_core.v_reset,
                channels=hidden_size,
                p_drop=setup.snn_core.p_drop,
                surrogate_func=setup.surrogate,
                backend=setup.backend,
            )
            for _ in range(setup.snn_core.n_blocks)
        ])

        self.classifier = layer.Linear(hidden_size, setup.output_size, step_mode="m")
        self.readout = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 41),
        )

        self.fr_low = 0.15
        self.fr_high = 0.30
        self.last_fr_loss = None
        self.last_fr_rate = None

    def compute_firing_rate_loss(self, spikes, low=0.15, high=0.30):
        rate_c = spikes.mean(dim=(0, 1))              # [C]
        loss_low = torch.relu(low - rate_c) ** 2
        loss_high = torch.relu(rate_c - high) ** 2
        loss = (loss_low + loss_high).mean()
        rate = rate_c.mean()
        return loss, rate

    def forward(self, x):
        self._step += 1
        assert x.dim() == 3, f"Expected (T,B,C), got shape {tuple(x.shape)}"

        x = self.input_proj(x)

        for block in self.snn_core:
            x = block(x)

        block0_spikes = self.snn_core[0].last_spikes
        self.last_fr_loss, self.last_fr_rate = self.compute_firing_rate_loss(
            block0_spikes, low=self.fr_low, high=self.fr_high
        )

        if self._step % 1000 == 0:
            xt = x.detach()
            print(
                f"[step {self._step}] hidden(min/max)=({xt.min().item():.3f},{xt.max().item():.3f}) "
                f"mean={xt.mean().item():.6f} std={xt.std().item():.6f}"
            )
            print(
                f"[step {self._step}] block0 spike_rate={self.last_fr_rate.item():.4f} "
                f"fr_loss={self.last_fr_loss.item():.6f}"
            )

        logits = self.readout(x)

        if self._step % 1000 == 0:
            lt = logits.detach()
            print(
                f"[step {self._step}] logits(min/max)=({lt.min().item():.3f},{lt.max().item():.3f}) "
                f"mean={lt.mean().item():.6f} std={lt.std().item():.6f}"
            )

        return logits