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

from models.neurons import LTLIFNode, MultiTauLIFNode, _tau_to_w_value
import pandas as pd
import math

def pick_surrogate(surr_name: str):
    if surr_name == "atan":
        return surrogate.ATan()
    if surr_name == "sigmoid":
        return surrogate.Sigmoid()
    raise ValueError(
        f"Unsupported surrogate function: {surr_name}. "
        "Expected one of {'atan', 'sigmoid'}."
    )

class DelayLaggedMixer(nn.Module):
    def __init__(
        self,
        channels,
        lags=(0, 1, 2, 3, 4, 5, 6, 7),
        one_random_lag_per_pair=True,
        num_lags_per_connection=1,
        lag_distribution_type="uniform",
        lag_decay_strength=1.0,
        seed=0,
        bias=True,
    ):
        super().__init__()

        if not one_random_lag_per_pair or num_lags_per_connection != 1:
            raise NotImplementedError(
                "DelayLaggedMixer supports exactly one fixed lag per connection."
            )

        self.channels = int(channels)
        self.lags = [int(lag) for lag in lags]
        self.K = len(self.lags)
        self.lag_distribution_type = str(lag_distribution_type)
        self.lag_decay_strength = float(lag_decay_strength)

        lag_probs = self._normalized_lag_probabilities()

        with torch.random.fork_rng():
            torch.manual_seed(int(seed))
            lag_indices = torch.multinomial(
                lag_probs.expand(self.channels * self.channels, -1),
                num_samples=1,
                replacement=False,
            ).view(self.channels, self.channels)

        self.register_buffer("lag_indices", lag_indices)
        self.register_buffer(
            "lag_values",
            torch.tensor(self.lags, dtype=torch.long)[lag_indices]
        )

        self.weight = nn.Parameter(torch.empty(self.channels, self.channels))
        self.bias = nn.Parameter(torch.zeros(self.channels)) if bias else None

        masks = []
        for k in range(self.K):
            masks.append((self.lag_indices == k).float())
        self.register_buffer("lag_masks", torch.stack(masks))

        self.reset_parameters()

    def _normalized_lag_probabilities(self):
        lag_values = torch.tensor(self.lags, dtype=torch.float32)

        if self.lag_distribution_type == "uniform":
            probs = torch.ones_like(lag_values)
        elif self.lag_distribution_type == "short-biased":
            probs = lag_values.max() - lag_values + 1.0
        elif self.lag_distribution_type == "long-biased":
            probs = lag_values - lag_values.min() + 1.0
        elif self.lag_distribution_type == "center-biased":
            center = lag_values.mean()
            probs = 1.0 / (1.0 + (lag_values - center).abs())
        elif self.lag_distribution_type == "decaying":
            probs = torch.exp(-self.lag_decay_strength * (lag_values - lag_values.min()))
        else:
            raise ValueError(f"Unsupported lag_distribution_type: {self.lag_distribution_type}")

        return probs / probs.sum()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1.0 / math.sqrt(self.channels)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        T, B, C = x.shape
        if C != self.channels:
            raise ValueError(f"Expected channels={self.channels}, got {C}")

        y = x.new_zeros(T, B, C)

        for k, lag in enumerate(self.lags):
            W_k = self.weight * self.lag_masks[k].to(self.weight.dtype)

            if lag == 0:
                x_shift = x
            else:
                x_shift = x.new_zeros(T, B, C)
                x_shift[lag:] = x[:-lag]

            y = y + torch.matmul(x_shift, W_k.T)

        if self.bias is not None:
            y = y + self.bias.view(1, 1, -1)

        return y

class LaggedMixer(nn.Module):
    def __init__(
        self,
        channels,
        lags=(0, 1, 2, 4),
        one_random_lag_per_pair=False,
        num_lags_per_connection=1,
        lag_distribution_type="uniform",
        lag_decay_strength=1.0,
        seed=0,
    ):
        super().__init__()
        self.lags = list(lags)
        self.channels = channels
        self.K = len(self.lags)
        self.lag_distribution_type = str(lag_distribution_type)
        self.lag_decay_strength = float(lag_decay_strength)
        self.one_random_lag_per_pair = bool(one_random_lag_per_pair)
        self.num_lags_per_connection = max(1, int(num_lags_per_connection))

        self.mix = nn.Linear(channels * self.K, channels)

        print(lags)

        mask = torch.ones(channels, channels * self.K)

        if one_random_lag_per_pair:
            mask.zero_()
            lag_probs = self._normalized_lag_probabilities()
            num_lags_to_sample = min(self.num_lags_per_connection, self.K)
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                sampled_lag_indices = torch.multinomial(
                    lag_probs.expand(channels * channels, -1),
                    num_samples=num_lags_to_sample,
                    replacement=False,
                ).view(channels, channels, num_lags_to_sample)
            for out_ch in range(channels):
                for in_ch in range(channels):
                    for sample_idx in range(num_lags_to_sample):
                        k_idx = int(sampled_lag_indices[out_ch, in_ch, sample_idx].item())
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

    def _normalized_lag_probabilities(self):
        lag_values = torch.tensor(self.lags, dtype=torch.float32)

        if self.lag_distribution_type == "uniform":
            probs = torch.ones_like(lag_values)
        elif self.lag_distribution_type == "short-biased":
            probs = lag_values.max() - lag_values + 1.0
        elif self.lag_distribution_type == "long-biased":
            probs = lag_values - lag_values.min() + 1.0
        elif self.lag_distribution_type == "center-biased":
            center = lag_values.mean()
            probs = 1.0 / (1.0 + (lag_values - center).abs())
        elif self.lag_distribution_type == "decaying":
            min_lag = lag_values.min()
            probs = torch.exp(-self.lag_decay_strength * (lag_values - min_lag))
        else:
            raise ValueError(
                f"Unsupported lag_distribution_type: {self.lag_distribution_type}. "
                "Expected one of {'uniform', 'short-biased', 'long-biased', "
                "'center-biased', 'decaying'}."
            )

        prob_sum = probs.sum()
        if prob_sum <= 0:
            raise ValueError("Lag sampling probabilities must sum to a positive value.")
        return probs / prob_sum

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
                        "requested_num_lags_per_connection": int(self.num_lags_per_connection),
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
            "lag_distribution_type": self.lag_distribution_type,
            "lag_decay_strength": float(self.lag_decay_strength),
            "one_random_lag_per_pair": self.one_random_lag_per_pair,
            "num_lags_per_connection": int(self.num_lags_per_connection),
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


class ResidualSNNBlock(nn.Module):
    def __init__(
        self,
        tau: float,
        v_threshold: float,
        v_reset: float,
        channels: int,
        lagged_lags=(0, 1, 2, 3, 4, 5, 6, 7),
        lagged_one_random_lag_per_pair: bool = False,
        lagged_num_lags_per_connection: int = 1,
        lagged_distribution_type: str = "uniform",
        lagged_decay_strength: float = 1.0,
        lagged_seed: int = 0,
        p_drop: float = 0.0,
        surrogate_func: str = "atan",
        backend: str = "cupy",
    ):
        super().__init__()

        surr = pick_surrogate(surrogate_func)

        '''self.lagged = LaggedMixer(
            channels,
            lags=tuple(int(lag) for lag in lagged_lags),
            one_random_lag_per_pair=lagged_one_random_lag_per_pair,
            num_lags_per_connection=lagged_num_lags_per_connection,
            lag_distribution_type=lagged_distribution_type,
            lag_decay_strength=lagged_decay_strength,
            seed=int(lagged_seed),
        )'''
        self.lagged = DelayLaggedMixer(
            channels=channels,
            lags=tuple(int(lag) for lag in lagged_lags),
            one_random_lag_per_pair=lagged_one_random_lag_per_pair,
            num_lags_per_connection=lagged_num_lags_per_connection,
            lag_distribution_type=lagged_distribution_type,
            lag_decay_strength=lagged_decay_strength,
            seed=int(lagged_seed),
            bias=True,
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
        #self.proj = layer.Linear(channels, channels, step_mode="m")
        self.bn_in = Rearrange("t b c -> t b c 1")
        self.bn = layer.BatchNorm1d(channels, step_mode="m")
        self.bn_out = Rearrange("t b c 1 -> t b c")
        self.drop = layer.Dropout(p=p_drop, step_mode="m")

    def forward(self, x):
        z = self.neuron(x)
        z = self.lagged(z)
        #z = self.proj(z)
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
    
class DepthwiseTemporalInputProjection(nn.Module):
    def __init__(
        self,
        input_features=512,
        patch_size=6,
        bias=True,
        init="mean",
    ):
        super().__init__()

        self.input_features = input_features
        self.patch_size = patch_size

        self.weight = nn.Parameter(torch.empty(input_features, patch_size))

        if bias:
            self.bias = nn.Parameter(torch.zeros(input_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters(init)

    def reset_parameters(self, init):
        if init == "mean":
            nn.init.constant_(self.weight, 1.0 / self.patch_size)
        elif init == "last":
            nn.init.zeros_(self.weight)
            self.weight.data[:, -1] = 1.0
        elif init == "random":
            nn.init.xavier_uniform_(self.weight)
        else:
            raise ValueError("init must be 'mean', 'last', or 'random'")

    def forward(self, x):
        B, T, F = x.shape
        expected = self.input_features * self.patch_size

        if F != expected:
            raise ValueError(
                f"Expected last dimension {expected}, got {F}. "
                f"Input should be flattened patched features: "
                f"[B, T, {self.input_features} * {self.patch_size}]"
            )

        x = x.view(B, T, self.input_features, self.patch_size)

        x = (x * self.weight[None, None, :, :]).sum(dim=-1)

        if self.bias is not None:
            x = x + self.bias[None, None, :]

        return x

class FactorizedInputProjection(nn.Module):
    def __init__(self, input_size=512, patch_size=14, hidden_size=384, d_model=96):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.d_model = d_model

        self.channel_proj = nn.Linear(input_size, d_model)
        self.act = nn.GELU()
        self.temporal_proj = nn.Linear(patch_size * d_model, hidden_size)

    def forward(self, x):
        T, B, Dp = x.shape
        x = x.view(T, B, self.patch_size, self.input_size)
        x = self.channel_proj(x)
        x = self.act(x)
        x = x.reshape(T, B, self.patch_size * self.d_model)
        x = self.temporal_proj(x)
        return x
    
class SpikingNeuralNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        setup = cfg.model.setup
        self._step = 0
        self._shared_neuron_params = {}
        self._neuron_adaptation = {
            "enabled": False,
            "learn_threshold": False,
            "learn_tau": False,
        }

        hidden_size = setup.hidden_size
        block_lag_configs = self._resolve_block_lag_configs(setup)
        lagged_seed = int(cfg.get("seed", 0))

        #self.input_proj = nn.Linear(setup.input_size*setup.patch_size, hidden_size)
        '''elf.input_proj = FactorizedInputProjection(
            input_size=setup.input_size,
            patch_size=setup.patch_size,
            hidden_size=hidden_size,
            d_model=128,
        )
        '''

        self.input_projection = DepthwiseTemporalInputProjection(
            input_features=setup.input_size,
            patch_size=setup.patch_size,
            bias=True,
            init="mean",
        )

        self.input_expand = nn.Linear(setup.input_size, hidden_size)

        self.snn_core = nn.Sequential(
            *[
                ResidualSNNBlock(
                    tau=setup.snn_core.tau,
                    v_threshold=setup.snn_core.v_threshold,
                    v_reset=setup.snn_core.v_reset,
                    channels=hidden_size,
                    lagged_lags=block_lag_configs[block_idx]["lags"],
                    lagged_one_random_lag_per_pair=setup.snn_core.lagged_one_random_lag_per_pair,
                    lagged_num_lags_per_connection=block_lag_configs[block_idx]["num_lags_per_connection"],
                    lagged_distribution_type=block_lag_configs[block_idx]["distribution_type"],
                    lagged_decay_strength=block_lag_configs[block_idx]["decay_strength"],
                    lagged_seed=lagged_seed,
                    p_drop=setup.snn_core.p_drop,
                    surrogate_func=setup.surrogate,
                    backend=setup.backend,
                )
                for block_idx in range(setup.snn_core.n_blocks)
            ]
        )

        #self.classifier = layer.Linear(hidden_size, setup.output_size, step_mode="m")

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
        self._configure_neuron_adaptation(setup)

    def iter_lif_nodes(self):
        for module in self.modules():
            if isinstance(module, LTLIFNode):
                yield module

    def _configure_neuron_adaptation(self, setup):
        lif_nodes = list(self.iter_lif_nodes())
        if not lif_nodes:
            return

        adaptation_cfg = getattr(setup, "neuron_adaptation", None)
        enabled = bool(adaptation_cfg.get("enabled", False)) if adaptation_cfg is not None else False
        learn_threshold = enabled and bool(adaptation_cfg.get("learn_threshold", False))
        learn_tau = enabled and bool(adaptation_cfg.get("learn_tau", False))

        self._neuron_adaptation = {
            "enabled": enabled,
            "learn_threshold": learn_threshold,
            "learn_tau": learn_tau,
        }

        if not enabled:
            for node in lif_nodes:
                node.v_threshold.requires_grad_(False)
                node.w.requires_grad_(False)
            return

        if learn_threshold:
            shared_threshold = nn.Parameter(
                lif_nodes[0].v_threshold.detach().clone(),
                requires_grad=True,
            )
            self._shared_neuron_params["v_threshold"] = shared_threshold
            for node in lif_nodes:
                node.v_threshold = shared_threshold
        else:
            for node in lif_nodes:
                node.v_threshold.requires_grad_(False)

        if learn_tau:
            initial_tau = float(lif_nodes[0].get_tau().detach().mean().item())
            tau_min = float(adaptation_cfg.get("tau_min", max(1.0001, initial_tau * 0.5)))
            tau_max = float(adaptation_cfg.get("tau_max", max(tau_min + 1e-6, initial_tau * 2.0)))
            shared_w = nn.Parameter(
                torch.tensor(_tau_to_w_value(initial_tau), dtype=lif_nodes[0].w.dtype),
                requires_grad=True,
            )
            self._shared_neuron_params["w"] = shared_w
            self._neuron_adaptation["tau_min"] = tau_min
            self._neuron_adaptation["tau_max"] = tau_max
            for node in lif_nodes:
                node.w = shared_w
                node.set_tau_bounds(tau_min=tau_min, tau_max=tau_max)
        else:
            for node in lif_nodes:
                node.w.requires_grad_(False)

        print("[SNN] neuron_adaptation:", self._neuron_adaptation)

    def _resolve_block_lag_configs(self, setup):
        default_lags = tuple(int(lag) for lag in setup.snn_core.lagged_lags)
        default_distribution_type = str(setup.snn_core.get("lagged_distribution_type", "uniform"))
        default_decay_strength = float(setup.snn_core.get("lagged_decay_strength", 1.0))
        default_num_lags_per_connection = int(setup.snn_core.get("lagged_num_lags_per_connection", 1))
        per_block_cfgs = setup.snn_core.get("block_lag_configs", None)

        if per_block_cfgs is None:
            return [
                {
                    "lags": default_lags,
                    "distribution_type": default_distribution_type,
                    "decay_strength": default_decay_strength,
                    "num_lags_per_connection": default_num_lags_per_connection,
                }
                for _ in range(int(setup.snn_core.n_blocks))
            ]

        if len(per_block_cfgs) != int(setup.snn_core.n_blocks):
            raise ValueError(
                "block_lag_configs must have one entry per residual block: "
                f"expected {int(setup.snn_core.n_blocks)}, got {len(per_block_cfgs)}"
            )

        resolved_configs = []
        for block_idx, block_cfg in enumerate(per_block_cfgs):
            lags = block_cfg.get("lags", default_lags)
            distribution_type = block_cfg.get("distribution_type", default_distribution_type)
            decay_strength = float(block_cfg.get("decay_strength", default_decay_strength))
            num_lags_per_connection = int(
                block_cfg.get("num_lags_per_connection", default_num_lags_per_connection)
            )
            resolved_configs.append(
                {
                    "lags": tuple(int(lag) for lag in lags),
                    "distribution_type": str(distribution_type),
                    "decay_strength": decay_strength,
                    "num_lags_per_connection": num_lags_per_connection,
                }
            )

        return resolved_configs

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

        x = self.input_projection(x)
        x = self.input_expand(x)
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

    def get_global_neuron_parameters(self):
        lif_nodes = list(self.iter_lif_nodes())
        if not lif_nodes:
            return {}

        first_node = lif_nodes[0]
        params = {
            "threshold": float(first_node.get_effective_v_threshold().detach().mean().item()),
        }
        if self._neuron_adaptation.get("learn_tau", False):
            params["tau"] = float(first_node.get_tau().detach().mean().item())
        return params

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

            #item["mixer"].export_lag_selection(selection_path)
            #item["mixer"].export_lag_distribution(distribution_path)
            #item["mixer"].export_weights(weights_path)

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
