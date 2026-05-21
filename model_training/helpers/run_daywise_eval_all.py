import argparse
import csv
import glob
import math
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

# python model_training/helpers/run_daywise_eval_all.py   --runs_dir model_training/logs/lstm_final   --gpu_number 0  --bert-score

DEFAULT_RUNS_DIR = "logs/lstm"
DEFAULT_EVAL_SCRIPT = Path(__file__).with_name("evaluate_daywise_test_split.py")
#SPLITS = [0, 1, 2, 3, 4]*3  # Will cycle through splits if more than 5 runs
SPLITS = [0]

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run daywise evaluation across all run folders and summarize PER, memory, and optional BERT score."
    )
    parser.add_argument(
        "--runs_dir",
        default=DEFAULT_RUNS_DIR,
        help=f"Directory containing run folders. Default: {DEFAULT_RUNS_DIR}",
    )
    parser.add_argument(
        "--eval_script",
        default=str(DEFAULT_EVAL_SCRIPT),
        help=f"Path to the evaluation script. Default: {DEFAULT_EVAL_SCRIPT}",
    )
    parser.add_argument(
        "--split_dir",
        default=None,
        help="Optional split directory passed through to evaluate_daywise_test_split.py.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional evaluation seed passed through to evaluate_daywise_test_split.py.",
    )
    parser.add_argument(
        "--gpu_number",
        default=None,
        help="Optional GPU number passed through to evaluate_daywise_test_split.py.",
    )
    parser.add_argument(
        "--bert-score",
        action="store_true",
        help="Request BERT score computation from evaluate_daywise_test_split.py and summarize it.",
    )
    return parser.parse_args()


def parse_optional_float(pattern, text):
    match = re.search(pattern, text)
    if not match:
        return ""
    return float(match.group(1))


def parse_output(output, include_bert_score):
    result = {
        "average_per": "",
        "average_ctc_loss": "",
        "latency_mean_ms": "",
        "latency_median_ms": "",
        "latency_p95_ms": "",
        "current_allocated_mb": "",
        "max_allocated_mb": "",
        "mean_bert_score": "",
        "bert_score_mode": "",
    }

    result["average_per"] = parse_optional_float(r"Average PER:\s*([0-9.]+)", output)
    result["average_ctc_loss"] = parse_optional_float(r"Average CTC loss:\s*([0-9.]+)", output)
    result["latency_mean_ms"] = parse_optional_float(r"Latency mean:\s*([0-9.]+)\s*ms", output)
    result["latency_median_ms"] = parse_optional_float(r"Latency median:\s*([0-9.]+)\s*ms", output)
    result["latency_p95_ms"] = parse_optional_float(r"Latency p95:\s*([0-9.]+)\s*ms", output)
    result["current_allocated_mb"] = parse_optional_float(r"Current allocated:\s*([0-9.]+)\s*MB", output)
    result["max_allocated_mb"] = parse_optional_float(r"Max allocated:\s*([0-9.]+)\s*MB", output)

    if include_bert_score:
        bert_match = re.search(
            r"Mean semantic BERT score \(([^)]+)\):\s*(nan|[0-9.]+)",
            output,
        )
        if bert_match:
            result["bert_score_mode"] = bert_match.group(1)
            bert_value = bert_match.group(2)
            result["mean_bert_score"] = float(bert_value) if bert_value != "nan" else float("nan")

    return result


def collect_numeric_values(results, key):
    values = []
    for row in results:
        if row.get("status") != "ok":
            continue
        value = row.get(key)
        if not isinstance(value, (int, float)):
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        values.append(float(value))
    return values


def summarize_numeric_column(results, key):
    values = collect_numeric_values(results, key)
    if not values:
        return "", ""
    avg = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return avg, std


def format_value(value):
    if value == "":
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.5f}"
    return str(value)


def main():
    args = parse_args()

    run_dirs = sorted(
        path for path in glob.glob(os.path.join(args.runs_dir, "*")) if os.path.isdir(path)
    )
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in {args.runs_dir}")

    results = []

    for i, run_dir in enumerate(run_dirs):
        split_number = SPLITS[i % len(SPLITS)]
        model_path = os.path.join(run_dir, "checkpoints")

        cmd = [
            sys.executable,
            args.eval_script,
            "--model_path",
            model_path,
            "--split_number",
            str(split_number),
        ]

        if args.split_dir is not None:
            cmd.extend(["--split_dir", args.split_dir])
        if args.seed is not None:
            cmd.extend(["--seed", str(args.seed)])
        if args.gpu_number is not None:
            cmd.extend(["--gpu_number", str(args.gpu_number)])
        if args.bert_score:
            cmd.append("--bert-score")

        print(f"\n[{i + 1}/{len(run_dirs)}] Running: {' '.join(cmd)}")
        proc = subprocess.run(cmd, capture_output=True, text=True)

        output = proc.stdout + "\n" + proc.stderr
        print(output)

        parsed = parse_output(output, include_bert_score=args.bert_score)
        status = "ok" if proc.returncode == 0 else f"error_{proc.returncode}"

        results.append(
            {
                "run_idx": i,
                "folder": os.path.basename(run_dir),
                "split_number": split_number,
                "status": status,
                **parsed,
            }
        )

    csv_path = os.path.join(args.runs_dir, "daywise_eval_summary.csv")
    fieldnames = [
        "run_idx",
        "folder",
        "split_number",
        "status",
        "average_per",
        "average_ctc_loss",
        "latency_mean_ms",
        "latency_median_ms",
        "latency_p95_ms",
        "current_allocated_mb",
        "max_allocated_mb",
        "mean_bert_score",
        "bert_score_mode",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("\nFinal table:\n")
    if args.bert_score:
        avg_per, std_per = summarize_numeric_column(results, "average_per")
        avg_latency, std_latency = summarize_numeric_column(results, "latency_mean_ms")
        avg_bert, std_bert = summarize_numeric_column(results, "mean_bert_score")
        avg_max_mb, std_max_mb = summarize_numeric_column(results, "max_allocated_mb")

        print(
            f"{'idx':<4} {'split':<6} {'PER':<10} {'lat_ms':<10} {'BERT':<10} {'max_MB':<10} {'status':<12} folder"
        )
        print("-" * 120)
        for row in results:
            print(
                f"{row['run_idx']:<4} "
                f"{row['split_number']:<6} "
                f"{format_value(row['average_per']):<10} "
                f"{format_value(row['latency_mean_ms']):<10} "
                f"{format_value(row['mean_bert_score']):<10} "
                f"{format_value(row['max_allocated_mb']):<10} "
                f"{row['status']:<12} "
                f"{row['folder']}"
            )
        print("-" * 120)
        print(
            f"{'avg':<4} "
            f"{'-':<6} "
            f"{format_value(avg_per):<10} "
            f"{format_value(avg_latency):<10} "
            f"{format_value(avg_bert):<10} "
            f"{format_value(avg_max_mb):<10} "
            f"{'summary':<12} "
            f"across successful splits"
        )
        print(
            f"{'std':<4} "
            f"{'-':<6} "
            f"{format_value(std_per):<10} "
            f"{format_value(std_latency):<10} "
            f"{format_value(std_bert):<10} "
            f"{format_value(std_max_mb):<10} "
            f"{'summary':<12} "
            f"across successful splits"
        )
    else:
        avg_per, std_per = summarize_numeric_column(results, "average_per")
        avg_latency, std_latency = summarize_numeric_column(results, "latency_mean_ms")
        avg_max_mb, std_max_mb = summarize_numeric_column(results, "max_allocated_mb")

        print(
            f"{'idx':<4} {'split':<6} {'PER':<10} {'lat_ms':<10} {'max_MB':<10} {'status':<12} folder"
        )
        print("-" * 110)
        for row in results:
            print(
                f"{row['run_idx']:<4} "
                f"{row['split_number']:<6} "
                f"{format_value(row['average_per']):<10} "
                f"{format_value(row['latency_mean_ms']):<10} "
                f"{format_value(row['max_allocated_mb']):<10} "
                f"{row['status']:<12} "
                f"{row['folder']}"
            )
        print("-" * 110)
        print(
            f"{'avg':<4} "
            f"{'-':<6} "
            f"{format_value(avg_per):<10} "
            f"{format_value(avg_latency):<10} "
            f"{format_value(avg_max_mb):<10} "
            f"{'summary':<12} "
            f"across successful splits"
        )
        print(
            f"{'std':<4} "
            f"{'-':<6} "
            f"{format_value(std_per):<10} "
            f"{format_value(std_latency):<10} "
            f"{format_value(std_max_mb):<10} "
            f"{'summary':<12} "
            f"across successful splits"
        )

    print(f"\nSaved CSV to: {csv_path}")


if __name__ == "__main__":
    main()
