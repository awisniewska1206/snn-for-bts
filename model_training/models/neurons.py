import numpy as np
import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron
from spikingjelly.activation_based import surrogate
import torch.nn.functional as F
import random
from spikingjelly.activation_based import base


def _tau_to_w_value(tau: float) -> float:
    tau = float(max(tau, 1.0001))
    return float(torch.logit(torch.tensor(1.0 / tau)).item())


def _flatten_samples_and_neurons(x: torch.Tensor):
    if x is None or not isinstance(x, torch.Tensor):
        return None
    if x.numel() == 0:
        return None
    if x.dim() == 0:
        return x.reshape(1, 1)
    if x.dim() == 1:
        return x.reshape(1, -1)
    return x.reshape(-1, int(np.prod(x.shape[1:])))


def _tensor_stats(x: torch.Tensor):
    if x is None or not isinstance(x, torch.Tensor) or x.numel() == 0:
        return {}
    x_detached = x.detach().float()
    return {
        "mean": float(x_detached.mean().item()),
        "std": float(x_detached.std(unbiased=False).item()),
        "min": float(x_detached.min().item()),
        "max": float(x_detached.max().item()),
    }


def _compute_spiking_statistics(spikes_out: torch.Tensor, membrane: torch.Tensor = None, threshold: torch.Tensor = None):
    stats = {}
    flat_spikes = _flatten_samples_and_neurons((spikes_out > 0).detach().float())
    if flat_spikes is not None:
        neuron_rates = flat_spikes.mean(dim=0)
        stats.update(
            {
                "mean_spike_rate": float(flat_spikes.mean().item()),
                "spike_rate_std_across_neurons": float(neuron_rates.std(unbiased=False).item()),
                "fraction_silent_neurons": float((neuron_rates <= 1e-6).float().mean().item()),
                "fraction_saturated_neurons": float((neuron_rates >= 0.95).float().mean().item()),
                "num_neurons": int(neuron_rates.numel()),
            }
        )

    if isinstance(membrane, torch.Tensor) and membrane.numel() > 0:
        membrane_stats = _tensor_stats(membrane)
        stats["membrane_mean"] = membrane_stats["mean"]
        stats["membrane_std"] = membrane_stats["std"]

    if isinstance(threshold, torch.Tensor) and threshold.numel() > 0:
        threshold_stats = _tensor_stats(threshold)
        stats["threshold_mean"] = threshold_stats["mean"]
        stats["threshold_std"] = threshold_stats["std"]
        stats["threshold_min"] = threshold_stats["min"]
        stats["threshold_max"] = threshold_stats["max"]

    return stats
class LTLIFNode(neuron.ParametricLIFNode):
    """
    LIF Node
    """

    def __init__(self, v_threshold=1.0, v_threshold_min=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.v_threshold = nn.Parameter(torch.tensor(float(v_threshold)))
        self.v_threshold_min = float(v_threshold_min)
        self.spike_rate = None
        self.last_forward_stats = {}
        self._threshold_delta = None
        self._tau_min = None
        self._tau_max = None
        self._w_min = None
        self._w_max = None

    def set_threshold_delta(self, delta_v_th):
        self._threshold_delta = delta_v_th

    def clear_threshold_delta(self):
        self._threshold_delta = None

    def set_tau_bounds(self, tau_min=None, tau_max=None):
        if tau_min is None or tau_max is None:
            self._tau_min = None
            self._tau_max = None
            self._w_min = None
            self._w_max = None
            return

        tau_min = float(max(tau_min, 1.0001))
        tau_max = float(max(tau_max, tau_min + 1e-6))
        self._tau_min = tau_min
        self._tau_max = tau_max
        self._w_min = _tau_to_w_value(tau_max)
        self._w_max = _tau_to_w_value(tau_min)

    def get_effective_v_threshold(self):
        effective_v_th = self.v_threshold
        if self._threshold_delta is not None:
            delta_v_th = self._threshold_delta.to(
                device=self.v_threshold.device,
                dtype=self.v_threshold.dtype,
            )
            effective_v_th = effective_v_th + delta_v_th
        return torch.clamp(effective_v_th, min=self.v_threshold_min)

    def neuronal_fire(self):
        return self.surrogate_function(self.v - self.get_effective_v_threshold())

    def neuronal_reset(self, spike):
        spike_d = spike.detach() if self.detach_reset else spike
        if self.v_reset is None:
            self.v = self.v - spike_d * self.get_effective_v_threshold()
        else:
            self.v = self.v_reset * spike_d + (1.0 - spike_d) * self.v

    def get_tau(self):
        return 1.0 / self.w.sigmoid()

    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            self.v_threshold.clamp_(min=self.v_threshold_min)
            if self._w_min is not None and self._w_max is not None:
                self.w.clamp_(min=self._w_min, max=self._w_max)
        spikes_out = super().forward(x)
        self.spike_rate = (spikes_out > 0).float().mean()  # retains gradient, I hope...
        self.last_forward_stats = _compute_spiking_statistics(
            spikes_out=spikes_out,
            membrane=getattr(self, "v", None),
            threshold=self.get_effective_v_threshold(),
        )
        return spikes_out

    def get_spike_params(self):
        thresh = float(self.get_effective_v_threshold().detach().mean().item())
        tau = float(self.get_tau().detach().mean().item())
        return np.array([thresh, tau])

    def get_explicit_neuron_count(self):
        # for the architectural context, a shape is implicitly defined
        # excluding batch dimension, product of all other dimensions
        # is exactly the number of neurons present in the module
        if hasattr(self.v, "shape"):
            shape_tensor = torch.tensor(self.v.shape)
            return torch.prod(shape_tensor[1:]).item()
        else:
            return 1

class LTLIFNode3(neuron.ParametricLIFNode):
    """
    Learnable Voltage Threshold + bounded learnable tau (via w)
    + Spike-Frequency Adaptation (ALIF-style)
    """

    def __init__(
        self,
        v_threshold=1.0,
        v_threshold_min=0.1,
        tau_min=2.0,
        tau_max=20.0,
        adapt_beta=0.1,
        adapt_tau=20.0,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        # --- threshold ---
        self.v_threshold = nn.Parameter(torch.tensor(float(v_threshold)))
        self.v_threshold_min = float(v_threshold_min)

        # --- tau bounds ---
        self.tau_min = float(max(tau_min, 1.0001))
        self.tau_max = float(max(tau_max, self.tau_min + 1e-6))

        self._w_min = float(torch.logit(torch.tensor(1.0 / self.tau_max)).item())
        self._w_max = float(torch.logit(torch.tensor(1.0 / self.tau_min)).item())

        # --- spike-frequency adaptation parameters ---
        self.adapt_beta = float(adapt_beta)     # strength of adaptation
        self.adapt_tau = float(adapt_tau)       # adaptation decay timescale

        # adaptation state
        self.register_buffer("adapt_state", None)

        # logging
        self.spike_rate = None
        self.tau_stat = None
        self.last_forward_stats = {}

    def reset(self):
        super().reset()
        self.adapt_state = None

    def forward(self, x: torch.Tensor):

        # ----- parameter projection -----
        with torch.no_grad():
            self.v_threshold.clamp_(min=self.v_threshold_min)
            self.w.clamp_(min=self._w_min, max=self._w_max)

        # ----- initialize adaptation state -----
        if self.adapt_state is None or self.adapt_state.shape != x.shape:
            self.adapt_state = torch.zeros_like(x)

        # ----- decay adaptation -----
        decay = 1.0 - 1.0 / self.adapt_tau
        self.adapt_state = self.adapt_state * decay

        # ----- apply adaptation -----
        x = x - self.adapt_beta * self.adapt_state

        # ----- normal LIF dynamics -----
        spikes_out = super().forward(x)

        # ----- update adaptation from spikes -----
        self.adapt_state = self.adapt_state + spikes_out.detach()

        # ----- logging stats -----
        self.spike_rate = spikes_out.detach().mean()
        self.tau_stat = (1.0 / self.w.sigmoid()).detach().mean()
        self.last_forward_stats = _compute_spiking_statistics(
            spikes_out=spikes_out,
            membrane=getattr(self, "v", None),
            threshold=self.v_threshold,
        )

        if random.random() < 0.001:
            print("tau:", self.tau_stat)

        return spikes_out

    def get_spike_params(self):
        thresh = float(self.v_threshold.detach().cpu().item())
        tau = float((1.0 / self.w.sigmoid()).detach().cpu().item())
        return np.array([thresh, tau])

class LTLIFNode2(neuron.ParametricLIFNode):
    """
    LTLIF + bounded tau + population inhibitory feedback (EMA of pop rate)
    """

    def __init__(self, v_threshold=1.0, v_threshold_min=0.1,
                 tau_min=2.0, tau_max=20.0,
                 pop_lambda=0.99, g_pop_init=0.0,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.v_threshold = nn.Parameter(torch.tensor(float(v_threshold)))
        self.v_threshold_min = float(v_threshold_min)

        self.tau_min = float(max(tau_min, 1.0001))
        self.tau_max = float(max(tau_max, self.tau_min + 1e-6))
        self._w_min = float(torch.logit(torch.tensor(1.0 / self.tau_max)).item())
        self._w_max = float(torch.logit(torch.tensor(1.0 / self.tau_min)).item())

        # population feedback
        self.pop_lambda = float(pop_lambda)  # 0.95-0.995 typical
        self.g_pop = nn.Parameter(torch.tensor(float(g_pop_init)))  # learnable strength
        
        self.register_buffer("r_ema", torch.tensor(0.0))

        self.spike_rate = None
        self.tau_stat = None
        self.r_ema_stat = None
        self.last_forward_stats = {}

    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            self.v_threshold.clamp_(min=self.v_threshold_min)
            self.w.clamp_(min=self._w_min, max=self._w_max)

        # apply current population inhibition (scalar) to input
        #x_eff = x - self.g_pop * self.r_ema
        x_eff = x * (1.0 - self.g_pop * self.r_ema)
        spikes_out = super().forward(x_eff)

        # pop rate this step (detach for stability; use mean over all dims)
        r = spikes_out.detach().mean()

        # EMA update
        self.r_ema = self.pop_lambda * self.r_ema + (1.0 - self.pop_lambda) * r

        # logging
        self.spike_rate = r
        self.tau_stat = (1.0 / self.w.sigmoid()).detach().mean()
        self.r_ema_stat = self.r_ema.detach()
        self.last_forward_stats = _compute_spiking_statistics(
            spikes_out=spikes_out,
            membrane=getattr(self, "v", None),
            threshold=self.v_threshold,
        )
        self.last_forward_stats["population_rate_ema"] = float(self.r_ema_stat.item())

        if random.random() < 0.001:
            print("tau:", float(self.tau_stat), "r_ema:", float(self.r_ema_stat), "g_pop:", float(self.g_pop.detach()))

        return spikes_out

class LTLIFNode1(neuron.ParametricLIFNode):
    """
    Learnable Voltage Threshold + bounded learnable tau (via w)
    """

    def __init__(
        self,
        v_threshold=1.0,
        v_threshold_min=0.1,
        tau_min=2.0,
        tau_max=20.0,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        # --- threshold (Step 1 style constraint, keep your version if you already patched it) ---
        self.v_threshold = nn.Parameter(torch.tensor(float(v_threshold)))
        self.v_threshold_min = float(v_threshold_min)

        # --- tau bounds (Step 2) ---
        # NOTE: with tau = 1/sigmoid(w), tau is always >= 1.
        self.tau_min = float(max(tau_min, 1.0001))
        self.tau_max = float(max(tau_max, self.tau_min + 1e-6))

        # precompute w bounds as python floats
        self._w_min = float(torch.logit(torch.tensor(1.0 / self.tau_max)).item())
        self._w_max = float(torch.logit(torch.tensor(1.0 / self.tau_min)).item())

        self.spike_rate = None
        self.tau_stat = None  # for logging
        self.last_forward_stats = {}

    def forward(self, x: torch.Tensor):
        # threshold projection (autograd-safe projection)
        with torch.no_grad():
            self.v_threshold.clamp_(min=self.v_threshold_min)

            # tau projection via w bounds
            # ensures tau stays in [tau_min, tau_max]
            self.w.clamp_(min=self._w_min, max=self._w_max)

        spikes_out = super().forward(x)

        # logging-only stats (detach)
        self.spike_rate = spikes_out.detach().mean()
        self.tau_stat = (1.0 / self.w.sigmoid()).detach().mean()
        self.last_forward_stats = _compute_spiking_statistics(
            spikes_out=spikes_out,
            membrane=getattr(self, "v", None),
            threshold=self.v_threshold,
        )
        
        if random.random() < 0.001:   # ~0.1% of forwards
            print("tau:", self.tau_stat)
        return spikes_out

    def get_spike_params(self):
        thresh = float(self.v_threshold.detach().cpu().mean().item())
        tau = float((1.0 / self.w.sigmoid()).detach().cpu().mean().item())
        return np.array([thresh, tau])

class MultiTauLIFNode(base.MemoryModule):
    def __init__(self, dim, taus, v_threshold=1.0, surrogate=None):
        super().__init__()
        self.dim = dim
        self.taus = taus
        self.K = len(taus)
        self.v_threshold = v_threshold
        self.surrogate = surrogate

        betas = [1.0 - 1.0 / t for t in taus]
        self.register_buffer("betas", torch.tensor(betas).float())

        self.alpha = nn.Linear(dim, self.K)

        self.register_memory('v', None)

        # logging buffers
        self.last_alpha_seq = None
        self.last_v_seq = None
        self.last_contrib_seq = None
        self.last_vmix_seq = None

    def forward(self, x):  # [T,B,D]
        T, B, D = x.shape

        if self.v is None or self.v.shape[:2] != (B, D):
            self.v = x.new_zeros(B, D, self.K)

        spk_seq = []
        alpha_seq = []
        v_seq = []
        contrib_seq = []
        vmix_seq = []

        for t in range(T):
            xt = x[t]
            self.v = self.v * self.betas.view(1,1,-1) + xt.unsqueeze(-1)   # [B,D,K]
            alpha = torch.softmax(self.alpha(xt), dim=-1)                   # [B,K]
            contrib = self.v * alpha.unsqueeze(1)                           # [B,D,K]
            v_mix = contrib.sum(dim=-1)                                     # [B,D]

            if self.surrogate is None:
                spk = (v_mix > self.v_threshold).float()
            else:
                spk = self.surrogate(v_mix - self.v_threshold)

            self.v = self.v - spk.unsqueeze(-1) * self.v_threshold
            spk_seq.append(spk)

            alpha_seq.append(alpha.detach())
            v_seq.append(self.v.detach())
            contrib_seq.append(contrib.detach())
            vmix_seq.append(v_mix.detach())

        self.last_alpha_seq = torch.stack(alpha_seq, dim=0)       # [T,B,K]
        self.last_v_seq = torch.stack(v_seq, dim=0)               # [T,B,D,K]
        self.last_contrib_seq = torch.stack(contrib_seq, dim=0)   # [T,B,D,K]
        self.last_vmix_seq = torch.stack(vmix_seq, dim=0)         # [T,B,D]

        return torch.stack(spk_seq, dim=0)
    
    def get_spike_params(self):
        out = {
            "taus": list(self.taus),
            "alpha_mean": None,
            "importance": None,
        }
        if self.last_alpha_seq is not None:
            out["alpha_mean"] = self.last_alpha_seq.mean(dim=(0, 1)).detach().cpu().numpy()
        if self.last_contrib_seq is not None:
            out["importance"] = self.last_contrib_seq.abs().mean(dim=(0, 1, 2)).detach().cpu().numpy()
        return out
