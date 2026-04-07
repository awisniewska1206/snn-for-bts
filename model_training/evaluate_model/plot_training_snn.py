import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator


DEFAULT_RUN_NAME = "2026-04-06_12-20-33_991769_snn_pid695282"


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
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.labelsize": 13,
            "axes.titlesize": 12,
            "xtick.labelsize": 11.5,
            "ytick.labelsize": 11.5,
            "legend.fontsize": 11,
            "grid.color": "#d0d0d0",
            "grid.linestyle": "--",
            "grid.linewidth": 0.7,
        }
    )


def add_metric_plot(ax, x, y, color, ylabel):
    ax.plot(
        x,
        y,
        color=color,
        linewidth=1.8,
        marker="o",
        markersize=3.8,
        markerfacecolor="white",
        markeredgewidth=0.9,
        markeredgecolor=color,
        zorder=3,
    )

    ax.set_xlabel("Training batch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.75, zorder=0)
    ax.xaxis.set_major_formatter(FuncFormatter(_format_k))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    for spine in ("left", "bottom"):
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
        xytext=(-18, 16),
        textcoords="offset points",
        fontsize=10.5,
        color=color,
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=color, lw=0.8, alpha=0.96),
        arrowprops=dict(arrowstyle="-", color=color, lw=0.8),
    )


def main():
    parser = argparse.ArgumentParser(description="Plot SNN training curves from a training log.")
    parser.add_argument(
        "--run-name",
        default=DEFAULT_RUN_NAME,
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

    fig, axes = plt.subplots(2, 1, figsize=(8.6, 6.2), constrained_layout=True)
    fig.set_facecolor("white")
    fig.suptitle(args.run_name, fontsize=14.5, fontweight="semibold", y=1.02)

    add_metric_plot(
        axes[0],
        training_batches,
        ctc_loss_avg,
        color="#1f1f1f",
        ylabel="Average CTC loss",
    )
    axes[0].set_ylim(0, 50)
    annotate_best(axes[0], training_batches, ctc_loss_avg, "#1f1f1f", "Best")

    add_metric_plot(
        axes[1],
        training_batches,
        per_avg,
        color="#c23b3b",
        ylabel="Average PER",
    )
    axes[1].set_ylim(0, 0.5)
    annotate_best(axes[1], training_batches, per_avg, "#c23b3b", "Best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
