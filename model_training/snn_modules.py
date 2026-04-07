import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange

from spikingjelly.activation_based import neuron, layer, surrogate

from models.spiking_attention import SSABlock
from models.neurons import LTLIFNode


def pick_surrogate(surr_name: str):
    if surr_name == "atan":
        return surrogate.ATan()
    elif surr_name == "sigmoid":
        return surrogate.Sigmoid()
    raise NotImplementedError("Only supports 'atan' and 'sigmoid' surrogate")

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

class ConnectionBlock(nn.Module):
    def __init__(
        self,
        tau: float,
        v_threshold: float,
        v_reset: float,
        hidden_size: int,
        n_layers: int = 2,
        skip_connect: bool = False,
        p_drop: float = 0.0,
        surrogate_func: str = "atan",
        backend: str = "torch",
    ):
        super().__init__()

        surr = pick_surrogate(surrogate_func)

        process = []
        for _ in range(n_layers - 1):
            process += [
                LaggedMixer(
                    hidden_size,
                    lags=(0,4),
                    one_random_lag_per_pair=True,
                    
                ),
                LTLIFNode(
                    init_tau=tau,
                    surrogate_function=surr,
                    step_mode="m",
                    backend=backend,
                    v_threshold=v_threshold,
                    v_reset=v_reset,
                    decay_input=False,
                ),
                layer.Linear(hidden_size, hidden_size, step_mode="m"),
                Rearrange("t b c -> t b c 1"),
                layer.BatchNorm1d(hidden_size, step_mode="m"),
                Rearrange("t b c 1 -> t b c"),
                layer.Dropout(p=p_drop, step_mode="m"),
            ]

        out = [
            LaggedMixer(
                hidden_size,
                lags=(0,4),
                one_random_lag_per_pair=True,
                
            ),
            LTLIFNode(
                init_tau=tau,
                surrogate_function=surr,
                step_mode="m",
                backend=backend,
                v_threshold=v_threshold,
                v_reset=v_reset,
                decay_input=False,
            ),
            layer.Linear(hidden_size, hidden_size, step_mode="m"),
            Rearrange("t b c -> t b c 1"),
            layer.BatchNorm1d(hidden_size, step_mode="m"),
            Rearrange("t b c 1 -> t b c"),
        ]

        self.skip_connection = skip_connect
        self.net = nn.Sequential(*(process + out))

    def forward(self, x):
        return self.net(x) + x if self.skip_connection else self.net(x)

    def get_spike_params(self):
        net_params = []
        for module in self.net:
            if isinstance(module, LTLIFNode):
                net_params.append(module.get_spike_params())
        return np.stack(net_params, axis=0)

class FeedForwardSNN(nn.Module):
    def __init__(
        self,
        tau: float,
        v_threshold: float,
        v_reset: float,
        input_size: int,
        hidden_size: int,
        output_size: int,
        n_layers: int = 2,
        skip_step: int = None,
        p_drop: float = 0.0,
        surrogate_func: str = "atan",
        backend: str = "torch",
    ):
        super().__init__()

        surr = pick_surrogate(surrogate_func)

        if n_layers == 0:
            self.net = nn.Sequential()

        elif n_layers == 1:
            assert skip_step is None and hidden_size is None
            self.net = nn.Sequential(
                *[
                    LaggedMixer(
                input_size,
                lags=(0,4),
                one_random_lag_per_pair=True,
                
            ),
                    LTLIFNode(
                        init_tau=tau,
                        surrogate_function=surr,
                        step_mode="m",
                        backend=backend,
                        v_threshold=v_threshold,
                        v_reset=v_reset,
                        decay_input=False,
                    ),
                    layer.Linear(input_size, output_size, step_mode="m"),
                    Rearrange("t b c -> t b c 1"),
                    layer.BatchNorm1d(output_size, step_mode="m"),
                    Rearrange("t b c 1 -> t b c"),
                ]
            )

        else:
            if (
                input_size != hidden_size
                or input_size != output_size
                or hidden_size != output_size
            ):
                assert n_layers >= 2
                n_layers -= 2
                input_block = [
                    LaggedMixer(
                input_size,
                lags=(0,4),
                one_random_lag_per_pair=True,
                
            ),
                    LTLIFNode(
                        init_tau=tau,
                        surrogate_function=surr,
                        step_mode="m",
                        backend=backend,
                        v_threshold=v_threshold,
                        v_reset=v_reset,
                        decay_input=False,
                    ),
                    layer.Linear(input_size, hidden_size, step_mode="m"),
                    Rearrange("t b c -> t b c 1"),
                    layer.BatchNorm1d(hidden_size, step_mode="m"),
                    Rearrange("t b c 1 -> t b c"),
                ]
                output_block = [
                    LaggedMixer(
                hidden_size,
                lags=(0,4),
                one_random_lag_per_pair=True,
                
            ),
                    LTLIFNode(
                        init_tau=tau,
                        surrogate_function=surr,
                        step_mode="m",
                        backend=backend,
                        v_threshold=v_threshold,
                        v_reset=v_reset,
                        decay_input=False,
                    ),
                    layer.Linear(hidden_size, output_size, step_mode="m"),
                    Rearrange("t b c -> t b c 1"),
                    layer.BatchNorm1d(output_size, step_mode="m"),
                    Rearrange("t b c 1 -> t b c"),
                ]
            else:
                input_block = output_block = []

            if skip_step is None:
                hidden_block = [
                    ConnectionBlock(
                        tau,
                        v_threshold,
                        v_reset,
                        hidden_size,
                        n_layers=n_layers,
                        skip_connect=False,
                        p_drop=p_drop,
                        backend=backend,
                        surrogate_func=surrogate_func,
                    )
                ]
            else:
                assert n_layers % skip_step == 0
                hidden_block = [
                    ConnectionBlock(
                        tau,
                        v_threshold,
                        v_reset,
                        hidden_size,
                        n_layers=skip_step,
                        skip_connect=True,
                        p_drop=p_drop,
                        backend=backend,
                        surrogate_func=surrogate_func,
                    )
                    for _ in range(n_layers // skip_step)
                ]

            self.net = nn.Sequential(*(input_block + hidden_block + output_block))

    def forward(self, x):
        return self.net(x)

    def get_spike_params(self):
        all_params = []
        for module in self.net:
            if isinstance(module, LTLIFNode):
                all_params.append(module.get_spike_params()[None, ...])
            elif isinstance(module, ConnectionBlock):
                all_params.append(module.get_spike_params())
        return np.concatenate(all_params, axis=0)

class MultiScaleSNN(nn.Module):
    def __init__(
        self,
        list_of_taus: list[float],
        v_threshold: float,
        v_reset: float,
        input_size: int,
        hidden_size: int,
        output_size: int,
        n_layers: int = 2,
        skip_step: int = 2,
        p_drop: float = 0.0,
        surrogate_func: str = "atan",
        backend: str = "torch",
    ):
        super().__init__()

        self.models = nn.ModuleList(
            FeedForwardSNN(
                tau=tau,
                v_threshold=v_threshold,
                v_reset=v_reset,
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=output_size,
                n_layers=n_layers,
                skip_step=skip_step,
                p_drop=p_drop,
                surrogate_func=surrogate_func,
                backend=backend,
            )
            for tau in list_of_taus
        )

    def forward(self, x):
        assert x.dim() == 3
        features = [torch.jit.fork(model, x) for model in self.models]
        return torch.stack([torch.jit.wait(f) for f in features], dim=-2)

    def get_spike_params(self):
        all_params = []
        for model in self.models:
            all_params.append(model.get_spike_params())
        return np.stack(all_params, axis=0)

class CausalTCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1, p_drop: float = 0.1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p_drop)

    def forward(self, x):
        x0 = x
        x = x.permute(1, 2, 0).contiguous()
        x = F.pad(x, (self.pad, 0))
        x = self.conv(x)
        x = self.act(x)
        x = self.drop(x)
        x = x.permute(2, 0, 1).contiguous()
        return x + x0

class SpikingCausalTCNBlock(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size=3,
        dilation=1,
        tau=2.0,
        v_threshold=0.5,
        v_reset=0.0,
        p_drop=0.1,
        surrogate_func="atan",
        backend="cupy",
    ):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(channels)

        surr = pick_surrogate(surrogate_func)
        self.spike = LTLIFNode(
            init_tau=tau,
            surrogate_function=surr,
            step_mode="m",
            backend=backend,
            v_threshold=v_threshold,
            v_reset=v_reset,
            decay_input=False,
        )

        self.drop = nn.Dropout(p_drop)

    def forward(self, x):
        # x: (T, B, C)
        x0 = x

        x = x.permute(1, 2, 0).contiguous()      # (B, C, T)
        x = nn.functional.pad(x, (self.pad, 0))  # causal left pad
        x = self.conv(x)
        x = self.bn(x)
        x = x.permute(2, 0, 1).contiguous()      # (T, B, C)

        x = self.spike(x)
        x = self.drop(x)

        return x + x0

class RecurrentSpikingBlock(nn.Module):
    def __init__(
        self,
        tau: float,
        v_threshold: float,
        v_reset: float,
        input_size: int,
        hidden_size: int,
        p_drop: float = 0.0,
        surrogate_func: str = "atan",
        backend: str = "torch",
        use_residual: bool = True,
        use_bn: bool = False,
    ):
        super().__init__()

        surr = pick_surrogate(surrogate_func)
        self.hidden_size = hidden_size
        self.use_residual = use_residual and (input_size == hidden_size)
        self.use_bn = use_bn

        self.in_proj = nn.Linear(input_size, hidden_size, bias=False)
        self.rec_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.neuron = LTLIFNode(
            init_tau=tau,
            surrogate_function=surr,
            step_mode="s",
            backend=backend,
            v_threshold=v_threshold,
            v_reset=v_reset,
            decay_input=False,
        )

        self.drop = nn.Dropout(p_drop)
        self.bn = nn.BatchNorm1d(hidden_size) if use_bn else nn.Identity()

        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.orthogonal_(self.rec_proj.weight)
        with torch.no_grad():
            self.rec_proj.weight.mul_(0.2)

    def forward(self, x):
        assert x.dim() == 3
        T, B, _ = x.shape
        device = x.device
        dtype = x.dtype

        if hasattr(self.neuron, "reset"):
            self.neuron.reset()

        s_prev = torch.zeros(B, self.hidden_size, device=device, dtype=dtype)
        outputs = []

        for t in range(T):
            h_t = self.in_proj(x[t]) + self.rec_proj(s_prev)
            s_t = self.neuron(h_t)
            if self.use_bn:
                s_t = self.bn(s_t)
            s_t = self.drop(s_t)
            if self.use_residual:
                s_t = s_t + x[t]
            outputs.append(s_t)
            s_prev = s_t

        return torch.stack(outputs, dim=0)

class SpikingNeuralNet(nn.Module):
    """
    Original behavior-preserving version:
    returns final logits, not features.
    Input: [T, B, C]
    Output: [T, B, output_size]
    """
    def __init__(self, cfg):
        super().__init__()

        surr = pick_surrogate(cfg.model.setup.surrogate)

        if "multi_scale_snn" in cfg.model.setup:
            self.multi_scale = MultiScaleSNN(
                list_of_taus=cfg.model.setup.multi_scale_snn.list_of_taus_ms,
                v_threshold=cfg.model.setup.multi_scale_snn.v_threshold,
                v_reset=cfg.model.setup.multi_scale_snn.v_reset,
                input_size=cfg.model.setup.multi_scale_snn.input_size,
                hidden_size=cfg.model.setup.multi_scale_snn.hidden_size,
                output_size=cfg.model.setup.multi_scale_snn.output_size,
                n_layers=cfg.model.setup.multi_scale_snn.n_layers,
                skip_step=cfg.model.setup.multi_scale_snn.skip_step,
                p_drop=cfg.model.setup.multi_scale_snn.p_drop,
                surrogate_func=cfg.model.setup.surrogate,
                backend=cfg.model.setup.backend,
            )
        else:
            self.multi_scale = nn.Sequential()

        if "feed_forward" in cfg.model.setup:
            self.feed_forward = FeedForwardSNN(
                tau=cfg.model.setup.feed_forward.tau,
                v_threshold=cfg.model.setup.feed_forward.v_threshold,
                v_reset=cfg.model.setup.feed_forward.v_reset,
                input_size=cfg.model.setup.feed_forward.input_size,
                hidden_size=cfg.model.setup.feed_forward.hidden_size,
                output_size=cfg.model.setup.feed_forward.output_size,
                n_layers=cfg.model.setup.feed_forward.n_layers,
                skip_step=cfg.model.setup.feed_forward.skip_step,
                p_drop=cfg.model.setup.feed_forward.p_drop,
                surrogate_func=cfg.model.setup.surrogate,
                backend=cfg.model.setup.backend,
            )
        else:
            self.feed_forward = nn.Sequential()

        self.output_read = neuron.ParametricLIFNode(
            init_tau=cfg.model.setup.output_tau,
            surrogate_function=surr,
            step_mode="m",
            v_reset=None,
            v_threshold=0.5,
            backend=cfg.model.setup.backend,
            store_v_seq=False,
            decay_input=False,
        )

        self.output_proj = layer.Linear(
            cfg.model.setup.pre_output_size,
            cfg.model.setup.output_size,
            step_mode="m",
        )

        self._step = 0
        
        enabled = []
        for k in ["multi_scale_snn", "feed_forward"]:
            if k in cfg.model.setup:
                enabled.append(k)

        print("[SNN] enabled blocks:", enabled)
        assert len(enabled) > 0, "No SNN blocks enabled. Check cfg.model.setup in YAML."

    def get_average_spiking_rate(self):
        all_srs = 0.0
        n_srs = 0
        for _, module in self.named_modules():
            if isinstance(module, LTLIFNode):
                spiking_rate = module.spike_rate
                all_srs += spiking_rate
                n_srs += 1
        return all_srs / n_srs

    def forward(self, x):
        self._step += 1
        assert x.dim() == 3

        x = self.multi_scale(x)

        assert x.dim() == 4
        x = rearrange(x, "t b c f -> t b (c f)") 

        #x = self.linear_down(x)                    
        #x = self.temporal(x) 
        #x = self.linear_up(x)
        x = self.feed_forward(x)
        
        if x.dim() == 4:
            x = rearrange(x, "t b c f -> t b (c f)")
        
        thr = float(self.output_read.v_threshold)

        if self._step % 1000 == 0:
            xt = x.detach()
            print(
                f"[step {self._step}] x(min/max)=({xt.min().item():.3f},{xt.max().item():.3f}) "
                f"thr={thr:.3f} p(>thr)={(xt > thr).float().mean().item():.6f}"
            )

        spk = self.output_read(x)

        if self._step % 1000 == 0:
            print(
                f"[step {self._step}] output_read spikes: "
                f"mean={spk.float().mean().item():.6f}, "
                f"std={spk.float().std().item():.6f}, "
                f"min={spk.min().item():.1f}, "
                f"max={spk.max().item():.1f}"
            )
            
        return self.output_proj(spk)

    def get_spike_params(self):
        ffn_sp = self.feed_forward.get_spike_params()
        ms_sp = self.multi_scale.get_spike_params()
        int_sp = self.integrator.get_spike_params()
        return ffn_sp, ms_sp, int_sp