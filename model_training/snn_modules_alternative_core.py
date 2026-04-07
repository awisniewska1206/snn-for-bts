import numpy as np
import torch
import torch.nn as nn
import hydra
from pathlib import Path
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
        self.register_buffer("inference_lag_mask", torch.ones(self.K), persistent=False)
        self.use_inference_lag_mask = False

        # ---- zero masked weights ----
        with torch.no_grad():
            self.mix.weight.mul_(self.mask)

        # ---- mask gradients ----
        self.mix.weight.register_hook(lambda grad: grad * self.mask)

    def set_inference_lag_mask(self, lag_mask):
        lag_mask = torch.as_tensor(lag_mask, dtype=self.mask.dtype, device=self.mask.device)
        if lag_mask.ndim != 1 or lag_mask.numel() != self.K:
            raise ValueError(f"lag_mask must have shape ({self.K},), got {tuple(lag_mask.shape)}")

        self.inference_lag_mask.copy_(lag_mask)
        self.use_inference_lag_mask = True

    def clear_inference_lag_mask(self):
        self.inference_lag_mask.fill_(1.0)
        self.use_inference_lag_mask = False

    def _expanded_inference_lag_mask(self, ref_tensor):
        lag_mask = self.inference_lag_mask.to(device=ref_tensor.device, dtype=ref_tensor.dtype)
        return lag_mask.view(self.K, 1).expand(self.K, self.channels).reshape(-1)

    def _mask_by_lag(self):
        mask = self.mask.detach().cpu()
        return mask.view(self.channels, self.K, self.channels).permute(0, 2, 1).contiguous()

    def get_lag_selection_rows(self):
        mask = self._mask_by_lag()

        rows = []
        for out_ch in range(self.channels):
            for in_ch in range(self.channels):
                active_lag_indices = torch.nonzero(mask[out_ch, in_ch] > 0, as_tuple=False).flatten().tolist()
                active_lags = [self.lags[k_idx] for k_idx in active_lag_indices]
                rows.append(
                    {
                        "out_channel": out_ch,
                        "in_channel": in_ch,
                        "num_selected_lags": len(active_lags),
                        "selected_lag_indices": "|".join(str(k_idx) for k_idx in active_lag_indices),
                        "selected_lags": "|".join(str(lag) for lag in active_lags),
                    }
                )

        return rows

    def get_lag_distribution_rows(self):
        mask = self._mask_by_lag()
        counts = mask.sum(dim=(0, 1)).to(torch.int64)
        total = int(counts.sum().item())

        rows = []
        for k_idx, lag in enumerate(self.lags):
            count = int(counts[k_idx].item())
            rows.append(
                {
                    "lag_index": k_idx,
                    "lag": lag,
                    "count": count,
                    "fraction": (count / total) if total > 0 else 0.0,
                }
            )

        return rows

    def export_lag_selection(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.get_lag_selection_rows()).to_csv(path, index=False)

    def export_lag_distribution(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.get_lag_distribution_rows()).to_csv(path, index=False)

    def save_snapshot(self, path, epoch=None, step=None, layer_name=None, store_full=True):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        snapshot = {
            "epoch": epoch,
            "step": step,
            "layer_name": layer_name,
            "lags": list(self.lags),
            "channels": int(self.channels),
            "num_lags": int(self.K),
            "selection_rows": self.get_lag_selection_rows(),
            "distribution_rows": self.get_lag_distribution_rows(),
        }

        if store_full:
            snapshot["mask"] = self.mask.detach().cpu()
            snapshot["masked_weight"] = (self.mix.weight * self.mask).detach().cpu()
            if self.mix.bias is not None:
                snapshot["bias"] = self.mix.bias.detach().cpu()

        torch.save(snapshot, path)
        return snapshot

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
        # Optional inference-only lag masking for one-lag-at-a-time ablation.
        if (not self.training) and self.use_inference_lag_mask:
            weight = weight * self._expanded_inference_lag_mask(weight).view(1, -1)
        return F.linear(x_cat, weight, self.mix.bias)

    def export_weights(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        W = (self.mix.weight * self.mask).detach().cpu()
        mask = self._mask_by_lag()
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
                        "lag_index": k_idx,
                        "lag": lag,
                        "selected": int(mask[i, j, k_idx].item()),
                        "weight": W[i, j, k_idx].item(),
                    })

        pd.DataFrame(rows).to_csv(path, index=False)


'''import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd


class LearnableDelayLaggedMixer(nn.Module):
    """Lagged mixer with learnable weights and learnable delays.

    Input shape: [T, B, C]

    Stores enough information to analyze:
    - soft delay learning over training
    - effective continuous delays
    - final rounded / discrete delays at inference
    - weight importance per connection
    - per-checkpoint summaries for global lag usage
    """

    def __init__(
        self,
        channels,
        lags=(0, 1, 2, 4),
        bias=True,
        delay_temperature=4.0,
        discrete_inference=True,
        delay_init=None,
        store_history=True,
        history_on_cpu=True,
    ):
        super().__init__()

        lags = sorted(set(int(lag) for lag in lags))
        if not lags:
            raise ValueError("lags must be non-empty")
        if lags[0] < 0:
            raise ValueError("lags must be non-negative")

        self.channels = int(channels)
        self.lags = lags
        self.K = len(lags)
        self.max_lag = max(lags)
        self.delay_temperature = float(delay_temperature)
        self.discrete_inference = bool(discrete_inference)

        self.weight = nn.Parameter(torch.empty(self.channels, self.channels))
        self.delay_param = nn.Parameter(torch.empty(self.channels, self.channels))
        self.bias = nn.Parameter(torch.zeros(self.channels)) if bias else None

        self.register_buffer("lag_positions", torch.tensor(self.lags, dtype=torch.float32))
        self.register_buffer("lag_indices", torch.tensor(self.lags, dtype=torch.long))

        self.last_analysis = None

        self.store_history = bool(store_history)
        self.history_on_cpu = bool(history_on_cpu)
        self.training_history = []

        self.reset_parameters(delay_init=delay_init)

    def reset_parameters(self, delay_init=None):
        nn.init.xavier_uniform_(self.weight)

        if delay_init is None:
            nn.init.uniform_(self.delay_param, -2.0, 2.0)
        else:
            low = float(self.lags[0])
            high = float(self.lags[-1])
            target = min(max(float(delay_init), low), high)
            frac = (target - low) / max(high - low, 1.0)
            frac = min(max(frac, 1e-4), 1.0 - 1e-4)
            nn.init.constant_(self.delay_param, torch.logit(torch.tensor(frac)).item())

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def effective_delays(self):
        low = float(self.lags[0])
        high = float(self.lags[-1])
        return low + (high - low) * torch.sigmoid(self.delay_param)

    def rounded_delays(self):
        delays = self.effective_delays().unsqueeze(-1)
        distances = (delays - self.lag_positions.view(1, 1, -1)).abs()
        best_idx = distances.argmin(dim=-1)
        return self.lag_positions[best_idx].to(torch.long)

    def rounded_delay_indices(self):
        delays = self.effective_delays().unsqueeze(-1)
        distances = (delays - self.lag_positions.view(1, 1, -1)).abs()
        return distances.argmin(dim=-1)

    def delay_probs(self, discrete=None):
        if discrete is None:
            discrete = self.discrete_inference and (not self.training)

        delays = self.effective_delays().unsqueeze(-1)
        lag_positions = self.lag_positions.view(1, 1, -1).to(delays.device, delays.dtype)
        distances_sq = (delays - lag_positions).pow(2)

        if discrete:
            best_idx = distances_sq.argmin(dim=-1, keepdim=True)
            probs = torch.zeros_like(distances_sq)
            probs.scatter_(-1, best_idx, 1.0)
            return probs

        return F.softmax(-self.delay_temperature * distances_sq, dim=-1)

    def effective_weight_by_lag(self, discrete=None):
        return self.weight.unsqueeze(-1) * self.delay_probs(discrete=discrete)

    def discrete_delay_onehot(self):
        return self.delay_probs(discrete=True)

    def get_learned_parameters(self, discrete=None):
        return {
            "weights": self.weight.detach().cpu(),
            "effective_delays": self.effective_delays().detach().cpu(),
            "rounded_delays": self.rounded_delays().detach().cpu(),
            "rounded_delay_indices": self.rounded_delay_indices().detach().cpu(),
            "delay_probs": self.delay_probs(discrete=discrete).detach().cpu(),
            "effective_weight_by_lag": self.effective_weight_by_lag(discrete=discrete).detach().cpu(),
            "discrete_delay_onehot": self.discrete_delay_onehot().detach().cpu(),
            "lags": torch.tensor(self.lags, dtype=torch.long),
            "temperature": self.delay_temperature,
        }

    def _lagged_inputs(self, x):
        if x.dim() != 3:
            raise ValueError(f"Expected x with shape [T, B, C], got {tuple(x.shape)}")

        t_steps, batch_size, channels = x.shape
        if channels != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {channels}")

        x_btc = x.permute(1, 0, 2)
        x_pad = F.pad(x_btc, (0, 0, self.max_lag, 0))
        windows = x_pad.unfold(1, self.max_lag + 1, 1)
        windows = windows.flip(dims=(-1,))
        windows = windows.index_select(-1, self.lag_indices)
        return windows.permute(1, 0, 2, 3).contiguous()

    def forward(self, x, discrete=None, return_info=False, return_contributions=False):
        x_lagged = self._lagged_inputs(x)                        # [T, B, Cin, K]
        weight_by_lag = self.effective_weight_by_lag(discrete)  # [Cout, Cin, K]

        y = torch.einsum("tbik,oik->tbo", x_lagged, weight_by_lag)
        if self.bias is not None:
            y = y + self.bias.view(1, 1, -1)

        if not (return_info or return_contributions):
            return y

        info = {
            "weights": self.weight,
            "effective_delays": self.effective_delays(),
            "rounded_delays": self.rounded_delays(),
            "rounded_delay_indices": self.rounded_delay_indices(),
            "delay_probs": self.delay_probs(discrete=discrete),
            "effective_weight_by_lag": weight_by_lag,
            "discrete_delay_onehot": self.discrete_delay_onehot(),
            "lagged_inputs": x_lagged,
            "lags": self.lag_positions,
            "temperature": self.delay_temperature,
        }

        if return_contributions:
            info["delay_contributions"] = torch.einsum(
                "tbik,oik->tboik",
                x_lagged,
                weight_by_lag,
            )

        self.last_analysis = info
        return y, info

    def analyze(self, x, phoneme_labels=None, discrete=None):
        _, info = self.forward(
            x,
            discrete=discrete,
            return_info=True,
            return_contributions=True,
        )

        contributions = info["delay_contributions"].detach()
        delay_usage = contributions.abs().sum(dim=(2, 3))  # [T, B, K]
        info["delay_usage"] = delay_usage

        if phoneme_labels is not None:
            info["delay_by_phoneme"] = self.delay_usage_by_label(delay_usage, phoneme_labels)

        return info

    @staticmethod
    def delay_usage_by_label(delay_usage, labels, num_labels=None, normalize=True):
        if delay_usage.shape[:2] != labels.shape:
            raise ValueError(
                f"delay_usage[:2]={tuple(delay_usage.shape[:2])} does not match "
                f"labels={tuple(labels.shape)}"
            )

        flat_usage = delay_usage.reshape(-1, delay_usage.shape[-1])
        flat_labels = labels.reshape(-1).to(torch.long)

        if num_labels is None:
            num_labels = int(flat_labels.max().item()) + 1

        out = flat_usage.new_zeros(num_labels, flat_usage.shape[-1])
        out.index_add_(0, flat_labels, flat_usage)

        if normalize:
            counts = torch.bincount(flat_labels, minlength=num_labels).to(out.dtype).unsqueeze(-1)
            out = out / counts.clamp_min(1.0)

        return out

    def _maybe_cpu(self, x):
        if self.history_on_cpu and torch.is_tensor(x):
            return x.detach().cpu()
        if torch.is_tensor(x):
            return x.detach()
        return x

    def _compute_summary_stats(self, delay_probs, rounded_delays, weights):
        abs_w = weights.abs()
        mean_prob_per_lag = delay_probs.mean(dim=(0, 1))
        weighted_prob_per_lag = (delay_probs * abs_w.unsqueeze(-1)).sum(dim=(0, 1))
        rounded_counts = torch.stack([(rounded_delays == lag).sum() for lag in self.lags]).to(torch.float32)

        return {
            "mean_effective_delay": self._maybe_cpu(self.effective_delays().mean()),
            "std_effective_delay": self._maybe_cpu(self.effective_delays().std()),
            "mean_abs_weight": self._maybe_cpu(abs_w.mean()),
            "mean_prob_per_lag": self._maybe_cpu(mean_prob_per_lag),
            "weighted_prob_per_lag": self._maybe_cpu(weighted_prob_per_lag),
            "rounded_counts_per_lag": self._maybe_cpu(rounded_counts),
        }

    def snapshot_state(self, epoch=None, step=None, layer_name=None, store_full=True):
        """Store one training checkpoint in memory.

        Call this every few epochs, not every batch.
        """
        if not self.store_history:
            return None

        with torch.no_grad():
            delay_probs = self.delay_probs(discrete=False)
            effective_delays = self.effective_delays()
            rounded_delays = self.rounded_delays()
            rounded_delay_indices = self.rounded_delay_indices()
            weights = self.weight
            effective_weight_by_lag = self.effective_weight_by_lag(discrete=False)
            discrete_onehot = self.discrete_delay_onehot()

            record = {
                "epoch": epoch,
                "step": step,
                "layer_name": layer_name,
                "temperature": self.delay_temperature,
                "lags": self._maybe_cpu(self.lag_positions.clone()),
                "summary": self._compute_summary_stats(delay_probs, rounded_delays, weights),
            }

            if store_full:
                record.update({
                    "delay_probs": self._maybe_cpu(delay_probs.clone()),
                    "effective_delays": self._maybe_cpu(effective_delays.clone()),
                    "rounded_delays": self._maybe_cpu(rounded_delays.clone()),
                    "rounded_delay_indices": self._maybe_cpu(rounded_delay_indices.clone()),
                    "weights": self._maybe_cpu(weights.clone()),
                    "effective_weight_by_lag": self._maybe_cpu(effective_weight_by_lag.clone()),
                    "discrete_delay_onehot": self._maybe_cpu(discrete_onehot.clone()),
                })

            self.training_history.append(record)
            return record

    def save_snapshot(self, path, epoch=None, step=None, layer_name=None, store_full=True):
        """Save one checkpoint directly to disk as .pt."""
        with torch.no_grad():
            delay_probs = self.delay_probs(discrete=False)
            effective_delays = self.effective_delays()
            rounded_delays = self.rounded_delays()
            rounded_delay_indices = self.rounded_delay_indices()
            weights = self.weight
            effective_weight_by_lag = self.effective_weight_by_lag(discrete=False)
            discrete_onehot = self.discrete_delay_onehot()

            payload = {
                "epoch": epoch,
                "step": step,
                "layer_name": layer_name,
                "temperature": self.delay_temperature,
                "lags": self._maybe_cpu(self.lag_positions.clone()),
                "summary": self._compute_summary_stats(delay_probs, rounded_delays, weights),
            }

            if store_full:
                payload.update({
                    "delay_probs": self._maybe_cpu(delay_probs.clone()),
                    "effective_delays": self._maybe_cpu(effective_delays.clone()),
                    "rounded_delays": self._maybe_cpu(rounded_delays.clone()),
                    "rounded_delay_indices": self._maybe_cpu(rounded_delay_indices.clone()),
                    "weights": self._maybe_cpu(weights.clone()),
                    "effective_weight_by_lag": self._maybe_cpu(effective_weight_by_lag.clone()),
                    "discrete_delay_onehot": self._maybe_cpu(discrete_onehot.clone()),
                })

            torch.save(payload, path)

    def save_history(self, path):
        """Save all in-memory checkpoints to disk."""
        if not self.store_history:
            raise ValueError("store_history=False, so no history was collected.")
        torch.save(self.training_history, path)

    def clear_history(self):
        self.training_history = []

    def export_weights(self, path, discrete=True):
        probs = self.delay_probs(discrete=discrete).detach().cpu()
        eff_delays = self.effective_delays().detach().cpu()
        rounded = self.rounded_delays().detach().cpu()
        rounded_idx = self.rounded_delay_indices().detach().cpu()
        discrete_onehot = self.discrete_delay_onehot().detach().cpu()
        eff_weight = self.effective_weight_by_lag(discrete=discrete).detach().cpu()
        base_weight = self.weight.detach().cpu()

        rows = []
        for out_ch in range(self.channels):
            for in_ch in range(self.channels):
                for k_idx, lag in enumerate(self.lags):
                    rows.append({
                        "out_channel": out_ch,
                        "in_channel": in_ch,
                        "lag_index": k_idx,
                        "lag": lag,
                        "weight": base_weight[out_ch, in_ch].item(),
                        "effective_delay": eff_delays[out_ch, in_ch].item(),
                        "rounded_delay": rounded[out_ch, in_ch].item(),
                        "rounded_delay_index": rounded_idx[out_ch, in_ch].item(),
                        "delay_prob": probs[out_ch, in_ch, k_idx].item(),
                        "discrete_onehot": discrete_onehot[out_ch, in_ch, k_idx].item(),
                        "effective_weight": eff_weight[out_ch, in_ch, k_idx].item(),
                    })

        pd.DataFrame(rows).to_csv(path, index=False)
'''
class ResidualSNNBlock(nn.Module):
    def __init__(
        self,
        tau: float,
        v_threshold: float,
        v_reset: float,
        channels: int,
        lagged_lags=(0, 1, 2, 3, 4, 5, 6, 7),
        lagged_one_random_lag_per_pair: bool = False,
        lagged_seed: int = 0,
        p_drop: float = 0.0,
        surrogate_func: str = "atan",
        backend: str = "cupy",
    ):
        super().__init__()

        surr = pick_surrogate(surrogate_func)

        self.lagged = LaggedMixer(
            channels,
            lags=tuple(int(lag) for lag in lagged_lags),
            one_random_lag_per_pair=lagged_one_random_lag_per_pair,
            seed=int(lagged_seed),
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

    def forward(self, x):
        z = self.lagged(x)
        z = self.neuron(z)
        z = self.proj(z)
        z = self.bn_in(z)
        z = self.bn(z)
        z = self.bn_out(z)
        z = self.drop(z)
        return z + x

    def get_delay_params(self, discrete=None):
        return self.lagged.get_learned_parameters(discrete=discrete)

    def get_spike_params(self):
        net_params = []
        if isinstance(self.neuron, MultiTauLIFNode):
            net_params.append(self.neuron.get_spike_params())
        return net_params

'''class ResidualSNNBlock(nn.Module):
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
            lags=(0, 1, 2, 4, 5, 6, 7),
            one_random_lag_per_pair=True,
            seed=0,
        )

        self.neuron1 = LTLIFNode(
            init_tau=tau,
            surrogate_function=surr,
            step_mode="m",
            backend=backend,
            v_threshold=v_threshold,
            v_reset=v_reset,
            decay_input=False,
        )
        self.proj1 = layer.Linear(channels, channels, step_mode="m")
        self.bn1_in = Rearrange("t b c -> t b c 1")
        self.bn1 = layer.BatchNorm1d(channels, step_mode="m")
        self.bn1_out = Rearrange("t b c 1 -> t b c")

        self.neuron2 = LTLIFNode(
            init_tau=tau,
            surrogate_function=surr,
            step_mode="m",
            backend=backend,
            v_threshold=v_threshold,
            v_reset=v_reset,
            decay_input=False,
        )
        self.proj2 = layer.Linear(channels, channels, step_mode="m")
        self.bn2_in = Rearrange("t b c -> t b c 1")
        self.bn2 = layer.BatchNorm1d(channels, step_mode="m")
        self.bn2_out = Rearrange("t b c 1 -> t b c")

        self.drop = layer.Dropout(p=p_drop, step_mode="m")

    def forward(self, x):
        z = self.lagged(x)

        z = self.neuron1(z)
        z = self.proj1(z)
        z = self.bn1_in(z)
        z = self.bn1(z)
        z = self.bn1_out(z)

        z = self.neuron2(z)
        z = self.proj2(z)
        z = self.bn2_in(z)
        z = self.bn2(z)
        z = self.bn2_out(z)

        z = self.drop(z)
        return z + x

    def get_delay_params(self, discrete=None):
        return self.lagged.get_learned_parameters(discrete=discrete)

    def get_spike_params(self):
        net_params = []
        for neuron in [self.neuron1, self.neuron2]:
            if isinstance(neuron, MultiTauLIFNode):
                net_params.append(neuron.get_spike_params())
        return net_params'''

class SpikingNeuralNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        setup = cfg.model.setup
        self._step = 0

        hidden_size = setup.hidden_size

        self.input_proj = nn.Linear(setup.input_size*setup.patch_size, hidden_size)

        self.snn_core = nn.Sequential(
            *[
                ResidualSNNBlock(
                    tau=setup.snn_core.tau,
                    v_threshold=setup.snn_core.v_threshold,
                    v_reset=setup.snn_core.v_reset,
                    channels=hidden_size,
                    lagged_lags=setup.snn_core.lagged_lags,
                    lagged_one_random_lag_per_pair=setup.snn_core.lagged_one_random_lag_per_pair,
                    lagged_seed=20,
                    p_drop=setup.snn_core.p_drop,
                    surrogate_func=setup.surrogate,
                    backend=setup.backend,
                )
                for _ in range(setup.snn_core.n_blocks)
            ]
        )

        self.classifier = layer.Linear(hidden_size, setup.output_size, step_mode="m")
        self.readout = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 41),
        )

        print(
            "[SNN] enabled blocks:",
            ["input_proj", f"snn_core_x{setup.snn_core.n_blocks}", "classifier"],
        )

    def get_average_spiking_rate(self):
        all_srs = 0.0
        n_srs = 0
        for _, module in self.named_modules():
            if isinstance(module, LTLIFNode) and getattr(module, "spike_rate", None) is not None:
                all_srs += module.spike_rate
                n_srs += 1
        if n_srs == 0:
            return torch.tensor(0.0)
        return all_srs / n_srs

    def forward(self, x):
        self._step += 1
        assert x.dim() == 3, f"Expected (T,B,C), got shape {tuple(x.shape)}"

        x = self.input_proj(x)
        x = self.snn_core(x)

        if self._step % 1000 == 0:
            xt = x.detach()
            print(
                f"[step {self._step}] hidden(min/max)=({xt.min().item():.3f},{xt.max().item():.3f}) "
                f"mean={xt.mean().item():.6f} std={xt.std().item():.6f}"
            )

        #logits = self.classifier(x)
        logits = self.readout(x)

        if self._step % 1000 == 0:
            lt = logits.detach()
            print(
                f"[step {self._step}] logits(min/max)=({lt.min().item():.3f},{lt.max().item():.3f}) "
                f"mean={lt.mean().item():.6f} std={lt.std().item():.6f}"
            )
            for block_idx, block in enumerate(self.snn_core):
                neuron = getattr(block, "neuron", None)
                spike_rate = getattr(neuron, "spike_rate", None)
                stats = getattr(neuron, "last_forward_stats", {}) or {}
                if spike_rate is None:
                    print(f"[step {self._step}] block_{block_idx} firing: NA")
                    continue

                mean_spike_rate = stats.get("mean_spike_rate", float(spike_rate.detach().cpu().item()))
                spike_rate_std = stats.get("spike_rate_std_across_neurons")
                fraction_silent = stats.get("fraction_silent_neurons")
                fraction_saturated = stats.get("fraction_saturated_neurons")

                line = (
                    f"[step {self._step}] block_{block_idx} firing: "
                    f"mean={mean_spike_rate:.6f}"
                )
                if spike_rate_std is not None:
                    line += f" std_across_neurons={spike_rate_std:.6f}"
                if fraction_silent is not None:
                    line += f" silent={fraction_silent:.6f}"
                if fraction_saturated is not None:
                    line += f" saturated={fraction_saturated:.6f}"
                print(line)

        return logits

    def get_spike_params(self):
        all_params = []
        for module in self.snn_core:
            if isinstance(module, ResidualSNNBlock):
                all_params.extend(module.get_spike_params())
        return all_params

    def get_delay_params(self, discrete=None):
        all_params = []
        for module in self.snn_core:
            if isinstance(module, ResidualSNNBlock):
                all_params.append(module.get_delay_params(discrete=discrete))
        return all_params

    def get_delay_mixer(self, block_idx):
        for item in self.iter_delay_mixers():
            if item["block_idx"] == block_idx:
                return item["mixer"]
        raise IndexError(f"No lag mixer found for block_idx={block_idx}")

    def get_lag_values(self, block_idx):
        return list(self.get_delay_mixer(block_idx).lags)

    def set_inference_lag_mask(self, block_idx, lag_mask):
        self.get_delay_mixer(block_idx).set_inference_lag_mask(lag_mask)

    def clear_inference_lag_masks(self):
        for item in self.iter_delay_mixers():
            item["mixer"].clear_inference_lag_mask()

    def iter_delay_mixers(self):
        for block_idx, module in enumerate(self.snn_core):
            if not isinstance(module, ResidualSNNBlock):
                continue
            yield {
                "block_idx": block_idx,
                "layer_name": f"snn_core.{block_idx}.lagged",
                "mixer": module.lagged,
            }

    def save_delay_snapshots(self, output_dir, epoch=None, step=None, store_full=True):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        for item in self.iter_delay_mixers():
            snapshot_path = output_dir / f"block_{item['block_idx']:02d}.pt"
            item["mixer"].save_snapshot(
                snapshot_path,
                epoch=epoch,
                step=step,
                layer_name=item["layer_name"],
                store_full=store_full,
            )
            saved_paths.append(
                {
                    "block_idx": item["block_idx"],
                    "layer_name": item["layer_name"],
                    "path": str(snapshot_path),
                }
            )

        return saved_paths

    def save_lag_selection_snapshots(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        for item in self.iter_delay_mixers():
            selection_path = output_dir / f"block_{item['block_idx']:02d}_lag_selections.csv"
            distribution_path = output_dir / f"block_{item['block_idx']:02d}_lag_distribution.csv"
            weights_path = output_dir / f"block_{item['block_idx']:02d}_lag_weights.csv"

            item["mixer"].export_lag_selection(selection_path)
            item["mixer"].export_lag_distribution(distribution_path)
            item["mixer"].export_weights(weights_path)

            saved_paths.append(
                {
                    "block_idx": item["block_idx"],
                    "layer_name": item["layer_name"],
                    "selection_path": str(selection_path),
                    "distribution_path": str(distribution_path),
                    "weights_path": str(weights_path),
                }
            )

        return saved_paths
