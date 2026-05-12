import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from spikingjelly.activation_based import layer, surrogate

from models.neurons import LTLIFNode, MultiTauLIFNode, _tau_to_w_value


def pick_surrogate(surr_name: str):
    if surr_name == "atan":
        return surrogate.ATan()
    if surr_name == "sigmoid":
        return surrogate.Sigmoid()
    raise ValueError(
        f"Unsupported surrogate function: {surr_name}. "
        "Expected one of {'atan', 'sigmoid'}."
    )


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

        self.neuron = LTLIFNode(
            init_tau=tau,
            surrogate_function=surr,
            step_mode="m",
            backend=backend,
            v_threshold=v_threshold,
            v_reset=v_reset,
            decay_input=False,
        )
        self.bn_in = Rearrange("t b c -> t b c 1")
        self.bn = layer.BatchNorm1d(channels, step_mode="m")
        self.bn_out = Rearrange("t b c 1 -> t b c")
        self.drop = layer.Dropout(p=p_drop, step_mode="m")

    def forward(self, x):
        z = self.neuron(x)
        z = self.bn_in(z)
        z = self.bn(z)
        z = self.bn_out(z)
        z = self.drop(z)
        return z + x

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
                    p_drop=setup.snn_core.p_drop,
                    surrogate_func=setup.surrogate,
                    backend=setup.backend,
                )
                for _ in range(setup.snn_core.n_blocks)
            ]
        )

        self.readout = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 41),
        )

        print(
            "[SNN-no-delay] enabled blocks:",
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

        print("[SNN-no-delay] neuron_adaptation:", self._neuron_adaptation)

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
