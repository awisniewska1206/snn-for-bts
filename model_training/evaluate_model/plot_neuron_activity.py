import sys
import os
import torch
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from spikingjelly.activation_based import functional as sj_functional
from spikingjelly.activation_based import surrogate
from models.neurons import LTLIFNode


def make_spike_train(T=40, spike_times=(5, 12, 20, 28), amp=1.0):
    x = torch.zeros(T, 1, 1)
    for t in spike_times:
        if 0 <= t < T:
            x[t, 0, 0] = amp
    return x


def delay_signal(x, d):
    y = torch.zeros_like(x)
    if d > 0:
        y[d:] = x[:-d]
    else:
        y = x.clone()
    return y


def reconstruct_pre_reset_membrane(input_current, spikes, tau=5.0, reset=0.0):
    beta = 1.0 - 1.0 / tau
    T = input_current.shape[0]

    v_pre = torch.zeros(T)
    v = torch.tensor(float(reset))

    for t in range(T):
        v_candidate = beta * v + input_current[t]
        v_pre[t] = v_candidate
        if spikes[t] > 0:
            v = torch.tensor(float(reset))
        else:
            v = v_candidate

    return v_pre


if __name__ == "__main__":
    T = 40
    thr = 1.0
    tau = 5.0

    x = make_spike_train(T=T, spike_times=(5, 12, 20, 28), amp=0.8)
    x_d2 = delay_signal(x, 2)
    x_mix = 0.7 * x + 0.7 * x_d2

    # Plain LIF
    lif_plain = LTLIFNode(
        init_tau=tau,
        v_threshold=thr,
        surrogate_function=surrogate.ATan(),
        step_mode="m",
        v_reset=0.0,
        decay_input=False,
        store_v_seq=True,
    )

    sj_functional.reset_net(lif_plain)
    spk_plain = lif_plain(x)

    # Delayed LIF
    lif_delay = LTLIFNode(
        init_tau=tau,
        v_threshold=thr,
        surrogate_function=surrogate.ATan(),
        step_mode="m",
        v_reset=0.0,
        decay_input=False,
        store_v_seq=True,
    )

    sj_functional.reset_net(lif_delay)
    spk_delay = lif_delay(x_mix)

    # Prepare
    x_1d = x.squeeze().cpu()
    x_d2_1d = x_d2.squeeze().cpu()
    x_mix_1d = x_mix.squeeze().cpu()
    spk_plain_1d = spk_plain.squeeze().cpu()
    spk_delay_1d = spk_delay.squeeze().cpu()

    v_plain_pre = reconstruct_pre_reset_membrane(x_1d, spk_plain_1d, tau=tau)
    v_delay_pre = reconstruct_pre_reset_membrane(x_mix_1d, spk_delay_1d, tau=tau)

    t = torch.arange(T).numpy()

    # Style
    plt.rcParams.update({
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.6,
    })

    fig, axes = plt.subplots(4, 1, figsize=(9, 7), sharex=True)

    # A
    axes[0].stem(t, x_1d.numpy(), linefmt="k-", markerfmt=" ", basefmt=" ")
    axes[0].set_title("A. Presynaptic spikes")
    axes[0].set_ylabel("spikes")

    # B
    axes[1].plot(t, v_plain_pre.numpy(), color="black")
    axes[1].axhline(thr, linestyle="--", color="gray", linewidth=1)
    axes[1].vlines(t[spk_plain_1d > 0], 0, thr, color="black", linewidth=1)
    axes[1].set_title(r"B. LIF — no delay ($v_t = \beta v_{t-1} + x_t$)")
    axes[1].set_ylabel("v")

    # C
    m1, s1, _ = axes[2].stem(t, x_1d.numpy(), linefmt="k-", markerfmt=" ", basefmt=" ")
    m2, s2, _ = axes[2].stem(t, x_d2_1d.numpy(), linefmt="-", markerfmt=" ", basefmt=" ")
    plt.setp(s2, color="dimgray", linewidth=1.2)
    axes[2].set_title("C. Synaptic inputs")
    axes[2].set_ylabel("input")
    axes[2].legend(["synapse 1 (t)", "synapse 2 (t-2)"], frameon=False)

    # D
    axes[3].plot(t, v_delay_pre.numpy(), color="black")
    axes[3].axhline(thr, linestyle="--", color="gray", linewidth=1)
    axes[3].vlines(t[spk_delay_1d > 0], 0, thr, color="black", linewidth=1)
    axes[3].set_title(
        r"D. LIF — delay ($I[t] = w_0 x[t] + w_1 x[t-2]$)"
    )
    axes[3].set_ylabel("v")
    axes[3].set_xlabel("time")

    for ax in axes:
        ax.set_xlim(-0.5, T - 0.5)

    plt.tight_layout()
    plt.savefig("neuron_activity_comparison.png", dpi=300, bbox_inches="tight")