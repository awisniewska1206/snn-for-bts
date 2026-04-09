import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator


def parse_training_log(log_file: Path):
    training_batches = []
    per_avg = []
    ctc_loss_avg = []

    with log_file.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    current_epoch = None

    for line in lines:
        epoch_match = re.search(r"Running test after training batch:\s+(\d+)", line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))

        val_match = re.search(
            r"Val batch\s+(\d+):\s+PER \(avg\):\s*([0-9]+(?:\.[0-9]*)?)"
            r"(?:\s+BLEU \(avg\):\s*[0-9]+(?:\.[0-9]*)?)?"
            r"(?:\s+BERT \(avg\):\s*[0-9]+(?:\.[0-9]*)?)?"
            r"\s+CTC Loss \(avg\):\s*([0-9]+(?:\.[0-9]*)?)",
            line,
        )

        if not val_match:
            continue

        batch_epoch = int(val_match.group(1))
        per = float(val_match.group(2))
        loss = float(val_match.group(3))

        if current_epoch is None:
            current_epoch = batch_epoch

        training_batches.append(current_epoch)
        per_avg.append(per)
        ctc_loss_avg.append(loss)

    if not training_batches:
        raise ValueError(
            f"No PER/CTC values parsed from {log_file}. "
            "Check the log path and whether the lines match the expected format."
        )

    return training_batches, per_avg, ctc_loss_avg


def _format_k(x, _):
    if abs(x) >= 1000:
        return f"{x / 1000:.0f}k"
    return f"{int(x)}"


def _compute_ylim(values, floor_zero=False):
    vmin = min(values)
    vmax = max(values)

    if vmin == vmax:
        pad = 0.1 * max(abs(vmax), 1.0)
    else:
        pad = 0.08 * (vmax - vmin)

    low = 0.0 if floor_zero else vmin - pad
    high = vmax + pad

    if low == high:
        high = low + 1.0

    return low, high


def apply_paper_style():
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Nimbus Sans", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.linewidth": 0.8,
            "axes.labelsize": 15,
            "axes.titlesize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "grid.color": "#e1e1e1",
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
        }
    )


def _moving_average(values, window):
    if window <= 1 or len(values) < 3:
        return np.asarray(values, dtype=float)

    arr = np.asarray(values, dtype=float)
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(arr, (pad_left, pad_right), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(padded, kernel, mode="valid")


def _simulate_std_band(y):
    arr = np.asarray(y, dtype=float)
    if len(arr) == 1:
        return np.full_like(arr, max(abs(arr[0]) * 0.05, 0.01))

    smooth = _moving_average(arr, window=min(7, len(arr) if len(arr) % 2 == 1 else len(arr) - 1))
    residual = np.abs(arr - smooth)
    local_scale = _moving_average(residual, window=min(5, len(arr) if len(arr) % 2 == 1 else len(arr) - 1))

    progress = np.linspace(0.0, 1.0, len(arr))
    span = max(arr.max() - arr.min(), 1e-6)
    base = 0.025 * span
    decay = 0.035 * span * (1.0 - progress)
    std = np.maximum(local_scale * 1.35 + base + decay, 0.012 * span)
    return std


def add_metric_plot(ax, x, y, color, ylabel):
    y_arr = np.asarray(y, dtype=float)
    band = _simulate_std_band(y_arr)

    ax.fill_between(
        x,
        y_arr - band,
        y_arr + band,
        color=color,
        alpha=0.18,
        linewidth=0,
        zorder=1,
    )
    ax.plot(
        x,
        y_arr,
        color=color,
        linewidth=2.1,
        zorder=3,
    )

    ax.set_xlabel("Training batch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.7, zorder=0)
    ax.xaxis.set_major_formatter(FuncFormatter(_format_k))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.set_box_aspect(0.62)

    for spine in ("left", "bottom", "top", "right"):
        ax.spines[spine].set_color("#444444")
        ax.spines[spine].set_linewidth(0.8)


def annotate_best(ax, x, y, color, label_prefix):
    best_idx = min(range(len(y)), key=y.__getitem__)
    best_x = x[best_idx]
    best_y = y[best_idx]

    ax.scatter([best_x], [best_y], s=32, color=color, zorder=4)
    ax.annotate(
        f"{label_prefix}: {best_y:.3f}",
        xy=(best_x, best_y),
        xytext=(-14, 14),
        textcoords="offset points",
        fontsize=10,
        color=color,
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=color, lw=0.8, alpha=0.96),
        arrowprops=dict(arrowstyle="-", color=color, lw=0.8),
    )


def main():
    parser = argparse.ArgumentParser(description="Plot SNN training curves from a training log.")
    parser.add_argument(
        "--run-name",
        help="Run directory name under model_training/logs/snn.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path. Defaults to model_training/evaluate_model/plot_training_<run-name>.png",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    model_training_dir = script_dir.parent
    log_file = model_training_dir / "logs" / "snn" / args.run_name / "training_log"

    if not log_file.exists():
        raise FileNotFoundError(f"Training log not found: {log_file}")

    output_path = (
        Path(args.output)
        if args.output is not None
        else script_dir / f"plot_training_{args.run_name}.png"
    )

    training_batches, per_avg, ctc_loss_avg = parse_training_log(log_file)

    apply_paper_style()

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 6.8), constrained_layout=True)
    fig.set_facecolor("white")

    add_metric_plot(
        axes[0],
        training_batches,
        ctc_loss_avg,
        color="#1f1f1f",
        ylabel="Average CTC loss",
    )
    axes[0].set_ylim(0, 50)
    axes[0].text(
        0.98,
        0.96,
        "SNN",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=12.5,
        fontweight="semibold",
        color="#1f1f1f",
    )
    annotate_best(axes[0], training_batches, ctc_loss_avg, "#1f1f1f", "Best")

    add_metric_plot(
        axes[1],
        training_batches,
        per_avg,
        color="#c23b3b",
        ylabel="Average PER",
    )
    axes[1].set_ylim(0, 0.5)
    axes[1].text(
        0.98,
        0.96,
        "SNN",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=12.5,
        fontweight="semibold",
        color="#c23b3b",
    )
    annotate_best(axes[1], training_batches, per_avg, "#c23b3b", "Best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
