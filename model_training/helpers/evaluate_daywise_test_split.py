import argparse
from contextlib import ExitStack
import json
import os
import re
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader


MODEL_TRAINING_ROOT = Path(__file__).resolve().parents[1]
if str(MODEL_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_TRAINING_ROOT))

from helpers.dataset import BrainToTextDataset, load_trial_indices_from_split_csv
from helpers.evaluate_model_helpers import LOGIT_TO_PHONEME
from trainer import BrainToTextDecoder_Trainer
from utils.seeding import make_dataloader_generator, seed_worker, set_global_seed

# python model_training/helpers/evaluate_daywise_test_split.py --model_path model_training/logs/snn/2026-05-21_09-58-21_717396_snn_pid7933/ --split_number 0 --gpu_number 0

GLOBAL_SEED = 20
SPLIT_METRICS_PATTERN = re.compile(r"daywise_test_split_(\d+)_metrics\.json$")
DISPLAY_SKIP_PHONEMES = {"BLANK", " | "}

def reset_memory_tracking(device):
    if getattr(device, "type", None) != "cuda":
        return

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)


def get_memory_stats(device):
    if getattr(device, "type", None) != "cuda":
        return None, None

    torch.cuda.synchronize(device)
    current_allocated_mb = torch.cuda.memory_allocated(device) / 1024**2
    max_allocated_mb = torch.cuda.max_memory_allocated(device) / 1024**2
    return current_allocated_mb, max_allocated_mb


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on the custom daywise test split and report PER."
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help=(
            "Path to either a run directory or a checkpoints directory containing "
            "args.yaml and best_checkpoint."
        ),
    )
    parser.add_argument(
        "--split_number",
        type=int,
        default=None,
        help=(
            "Split iteration number to evaluate, e.g. 0..4. If omitted, the script "
            "will infer it from an existing daywise_test_split_XX_metrics.json file "
            "in the checkpoints directory."
        ),
    )
    parser.add_argument(
        "--split_dir",
        default=str(Path("data") / "daywise_sentence_splits"),
        help="Directory containing iteration_XX_splits.csv files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=GLOBAL_SEED,
        help=f"Global evaluation seed. Default: {GLOBAL_SEED}.",
    )
    parser.add_argument(
        "--gpu_number",
        default="0",
        help="GPU number to use. Set to -1 to force CPU.",
    )
    parser.add_argument(
        "--latency-warmup",
        type=int,
        default=10,
        help="Number of initial utterances to run before latency measurement.",
    )
    parser.add_argument(
        "--export-pt",
        nargs="?",
        const="",
        default=None,
        help=(
            "Optionally save a per-trial validation payload as a .pt file. "
            "If provided without a value, the default path is "
            "checkpoints/daywise_test_split_XX_validation_export.pt."
        ),
    )
    parser.add_argument(
        "--include-alignment",
        action="store_true",
        help=(
            "When exporting PT data, compute CTC forced alignment plus boundary/interior masks."
        ),
    )
    return parser.parse_args()


def resolve_model_dirs(model_path_arg: str):
    path = Path(model_path_arg).resolve()
    if path.is_dir() and (path / "args.yaml").exists():
        checkpoints_dir = path
        run_dir = path.parent
        return run_dir, checkpoints_dir

    if path.is_dir() and (path / "checkpoints" / "args.yaml").exists():
        run_dir = path
        checkpoints_dir = path / "checkpoints"
        return run_dir, checkpoints_dir

    raise FileNotFoundError(
        f"Could not resolve model path from {path}. Expected either a checkpoints dir "
        f"with args.yaml or a run dir with checkpoints/args.yaml."
    )


def resolve_split_number(split_number, checkpoints_dir: Path):
    if split_number is not None:
        return int(split_number)

    inferred = []
    for candidate in checkpoints_dir.glob("daywise_test_split_*_metrics.json"):
        match = SPLIT_METRICS_PATTERN.fullmatch(candidate.name)
        if match is None:
            continue
        inferred.append(int(match.group(1)))

    inferred = sorted(set(inferred))
    if len(inferred) == 1:
        return inferred[0]
    if not inferred:
        raise ValueError(
            "Could not infer split_number because no daywise_test_split_XX_metrics.json "
            f"file was found in {checkpoints_dir}. Pass --split_number explicitly."
        )
    raise ValueError(
        "Could not infer split_number because multiple split metrics files were found in "
        f"{checkpoints_dir}: {inferred}. Pass --split_number explicitly."
    )


def resolve_export_path(export_pt_arg, checkpoints_dir: Path, split_number: int):
    if export_pt_arg is None:
        return None

    if export_pt_arg == "":
        return checkpoints_dir / f"daywise_test_split_{split_number:02d}_validation_export.pt"

    export_path = Path(export_pt_arg)
    if not export_path.is_absolute():
        export_path = (Path.cwd() / export_path).resolve()
    return export_path


def build_test_loader(args_cfg, split_csv_path: Path):
    source_filenames = args_cfg["dataset"].get(
        "custom_source_filenames",
        ["data_train.hdf5", "data_val.hdf5"],
    )
    _, _, test_trials = load_trial_indices_from_split_csv(
        split_csv_path=split_csv_path,
        dataset_dir=args_cfg["dataset"]["dataset_dir"],
        sessions=args_cfg["dataset"]["sessions"],
        source_filenames=source_filenames,
    )

    feature_subset = args_cfg["dataset"].get("feature_subset")
    test_dataset = BrainToTextDataset(
        trial_indicies=test_trials,
        split="test",
        days_per_batch=None,
        n_batches=None,
        batch_size=1,
        must_include_days=None,
        random_seed=int(args_cfg["seed"]),
        feature_subset=feature_subset,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=None,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=make_dataloader_generator(int(args_cfg["seed"])),
    )

    return test_trials, test_loader


def compute_effective_length(raw_length, patch_size, patch_stride):
    raw_length = max(int(raw_length), 0)
    patch_size = int(patch_size)
    patch_stride = int(patch_stride)

    if patch_size <= 0 or patch_stride <= 0:
        return raw_length
    if raw_length < patch_size:
        return 0

    return ((raw_length - patch_size) // patch_stride) + 1


def summarize_length_stats(raw_lengths, patch_size, patch_stride, include_global_keys=False):
    raw_lengths = [int(length) for length in raw_lengths]
    effective_lengths = [
        compute_effective_length(length, patch_size, patch_stride) for length in raw_lengths
    ]

    num_test_utterances = len(raw_lengths)
    total_raw_length = int(sum(raw_lengths))
    total_effective_length = int(sum(effective_lengths))

    stats = {
        "num_test_utterances": num_test_utterances,
        "raw_lengths_per_utterance": raw_lengths,
        "mean_raw_length": (
            float(total_raw_length / num_test_utterances) if num_test_utterances > 0 else 0.0
        ),
        "total_raw_length": total_raw_length,
        "effective_lengths_per_utterance": effective_lengths,
        "mean_effective_length": (
            float(total_effective_length / num_test_utterances)
            if num_test_utterances > 0
            else 0.0
        ),
        "total_effective_length": total_effective_length,
    }

    if include_global_keys:
        stats["all_raw_lengths"] = list(raw_lengths)
        stats["all_effective_lengths"] = list(effective_lengths)

    return stats


def collect_test_split_length_stats(args_cfg, test_trials):
    patch_size = int(args_cfg["model"]["patch_size"])
    patch_stride = int(args_cfg["model"]["patch_stride"])
    sessions = list(args_cfg["dataset"]["sessions"])

    per_session_length_stats = {}
    all_raw_lengths = []

    for day_idx, session_name in enumerate(sessions):
        trial_refs = test_trials.get(day_idx, {}).get("trial_refs", [])
        raw_lengths = []

        with ExitStack() as stack:
            open_handles = {}
            for ref in trial_refs:
                session_path = str(ref["session_path"])
                if session_path not in open_handles:
                    open_handles[session_path] = stack.enter_context(h5py.File(session_path, "r"))

                trial_idx = int(ref["trial_idx"])
                raw_lengths.append(
                    int(open_handles[session_path][f"trial_{trial_idx:04d}"].attrs["n_time_steps"])
                )

        per_session_length_stats[session_name] = summarize_length_stats(
            raw_lengths,
            patch_size=patch_size,
            patch_stride=patch_stride,
        )
        all_raw_lengths.extend(raw_lengths)

    global_length_stats = summarize_length_stats(
        all_raw_lengths,
        patch_size=patch_size,
        patch_stride=patch_stride,
        include_global_keys=True,
    )

    return per_session_length_stats, global_length_stats


def phoneme_ids_to_display_string(sequence_ids):
    phonemes = []
    for token_id in sequence_ids:
        phoneme = LOGIT_TO_PHONEME[int(token_id)]
        if phoneme in DISPLAY_SKIP_PHONEMES:
            continue
        phonemes.append(phoneme)
    return " ".join(phonemes)


def compute_edit_distance(reference_ids, hypothesis_ids):
    dp = list(range(len(hypothesis_ids) + 1))
    for ref_index, ref_token in enumerate(reference_ids, start=1):
        previous = dp[0]
        dp[0] = ref_index
        for hyp_index, hyp_token in enumerate(hypothesis_ids, start=1):
            current = dp[hyp_index]
            if ref_token == hyp_token:
                dp[hyp_index] = previous
            else:
                dp[hyp_index] = 1 + min(previous, dp[hyp_index], dp[hyp_index - 1])
            previous = current
    return dp[-1]


def collect_random_phoneme_examples(metrics, sample_count=10, seed=GLOBAL_SEED):
    utterance_records = []
    for true_batch, decoded_batch, length_batch, day_batch, trial_batch in zip(
        metrics["true_seq"],
        metrics["decoded_seqs"],
        metrics["phone_seq_lens"],
        metrics["day_indicies"],
        metrics["trial_nums"],
    ):
        for true_seq, decoded_seq, true_len, day_idx, trial_num in zip(
            true_batch,
            decoded_batch,
            length_batch,
            day_batch,
            trial_batch,
        ):
            target_ids = np.asarray(true_seq[: int(true_len)], dtype=np.int32)
            decoded_ids = np.asarray(decoded_seq, dtype=np.int32)
            edit_distance = compute_edit_distance(target_ids.tolist(), decoded_ids.tolist())
            utterance_per = float(edit_distance / max(1, len(target_ids)))
            utterance_records.append(
                {
                    "day": int(day_idx),
                    "trial": int(trial_num),
                    "per": utterance_per,
                    "target_phonemes": phoneme_ids_to_display_string(target_ids),
                    "decoded_phonemes": phoneme_ids_to_display_string(decoded_ids),
                }
            )

    if not utterance_records:
        return []

    sample_size = min(int(sample_count), len(utterance_records))
    rng = np.random.default_rng(int(seed))
    sampled_indices = rng.choice(len(utterance_records), size=sample_size, replace=False)
    return [utterance_records[int(index)] for index in sampled_indices]


def _prepare_forward_inputs(trainer, batch):
    features = batch["input_features"].to(trainer.device)
    n_time_steps = batch["n_time_steps"].to(trainer.device)
    day_indicies = batch["day_indicies"].to(trainer.device)
    session_ids = batch.get("session_ids", batch["day_indicies"]).to(trainer.device)

    features, _ = trainer.transform_data(features, n_time_steps, "val")
    session_summaries = trainer._lookup_session_summaries(session_ids)
    effective_length = int(features.shape[1])

    return features, day_indicies, session_summaries, effective_length


def measure_per_sample_latency_ms(trainer, loader, warmup_batches=10):
    latencies = []
    effective_lengths = []

    trainer.model.eval()
    use_amp = bool(trainer.args.get("use_amp", False)) and trainer.device.type == "cuda"

    with torch.inference_mode():
        for warmup_idx, batch in enumerate(loader):
            if warmup_idx >= int(warmup_batches):
                break

            features, day_indicies, session_summaries, _ = _prepare_forward_inputs(trainer, batch)

            with torch.autocast(
                device_type=trainer.device.type,
                enabled=use_amp,
                dtype=torch.float16,
            ):
                trainer._forward_model(features, day_indicies, session_summaries)

        if trainer.device.type == "cuda":
            torch.cuda.synchronize(trainer.device)

        for batch in loader:
            features, day_indicies, session_summaries, effective_length = _prepare_forward_inputs(
                trainer,
                batch,
            )

            if trainer.device.type == "cuda":
                torch.cuda.synchronize(trainer.device)
            start = time.perf_counter()

            with torch.autocast(
                device_type=trainer.device.type,
                enabled=use_amp,
                dtype=torch.float16,
            ):
                trainer._forward_model(features, day_indicies, session_summaries)

            if trainer.device.type == "cuda":
                torch.cuda.synchronize(trainer.device)
            end = time.perf_counter()

            latencies.append((end - start) * 1000.0)
            effective_lengths.append(effective_length)

    return (
        np.asarray(latencies, dtype=np.float64),
        np.asarray(effective_lengths, dtype=np.float64),
    )

def main():
    cli_args = parse_args()
    run_dir, checkpoints_dir = resolve_model_dirs(cli_args.model_path)
    split_number = resolve_split_number(cli_args.split_number, checkpoints_dir)
    export_path = resolve_export_path(cli_args.export_pt, checkpoints_dir, split_number)

    if cli_args.include_alignment and export_path is None:
        raise ValueError("--include-alignment requires --export-pt.")

    split_dir = Path(cli_args.split_dir)
    if not split_dir.is_absolute():
        split_dir = (MODEL_TRAINING_ROOT.parent / split_dir).resolve()
    split_csv_path = split_dir / f"iteration_{split_number:02d}_splits.csv"
    if not split_csv_path.exists():
        raise FileNotFoundError(f"Split CSV not found: {split_csv_path}")

    args_path = checkpoints_dir / "args.yaml"
    checkpoint_path = checkpoints_dir / "best_checkpoint"
    if not args_path.exists():
        raise FileNotFoundError(f"Missing args.yaml in {checkpoints_dir}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing best_checkpoint in {checkpoints_dir}")

    args_cfg = OmegaConf.load(args_path)
    args_cfg.mode = "train"
    args_cfg.init_from_checkpoint = True
    args_cfg.init_checkpoint_path = str(checkpoint_path)
    args_cfg.seed = int(cli_args.seed)
    args_cfg.gpu_number = str(cli_args.gpu_number)
    args_cfg.dataset.custom_split_csv = str(split_csv_path)
    if "custom_source_filenames" not in args_cfg.dataset:
        args_cfg.dataset.custom_source_filenames = ["data_train.hdf5", "data_val.hdf5"]
    dataset_dir = Path(str(args_cfg.dataset.dataset_dir))
    if not dataset_dir.is_absolute():
        dataset_dir = (MODEL_TRAINING_ROOT / dataset_dir).resolve()
    args_cfg.dataset.dataset_dir = str(dataset_dir)

    os_gpu = str(cli_args.gpu_number)
    if os_gpu == "-1":
        pass
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = os_gpu

    set_global_seed(int(args_cfg.seed))

    test_trials, test_loader = build_test_loader(args_cfg, split_csv_path)
    test_day_mask = [
        1 if len(test_trials[day_idx].get("trial_refs", [])) > 0 else 0
        for day_idx in range(len(args_cfg["dataset"]["sessions"]))
    ]
    args_cfg.dataset.dataset_probability_val = test_day_mask

    trainer = BrainToTextDecoder_Trainer(args_cfg)
    trainer.val_loader = test_loader

    current_allocated_mb = None
    max_allocated_mb = None
    try:
        reset_memory_tracking(trainer.device)
        metrics = trainer.validation(
            loader=test_loader,
            return_logits=False,
            return_data=False,
            export_frame_data=export_path is not None,
            include_alignment=cli_args.include_alignment,
        )
        latencies, latency_effective_lengths = measure_per_sample_latency_ms(
            trainer,
            test_loader,
            warmup_batches=cli_args.latency_warmup,
        )
        current_allocated_mb, max_allocated_mb = get_memory_stats(trainer.device)
    finally:
        trainer.close()

    valid_latency_mask = latency_effective_lengths > 0
    latency_per_effective_step = (
        latencies[valid_latency_mask] / latency_effective_lengths[valid_latency_mask]
        if latencies.size > 0
        else np.asarray([], dtype=np.float64)
    )

    per_session_counts = {
        args_cfg["dataset"]["sessions"][day_idx]: len(test_trials[day_idx].get("trial_refs", []))
        for day_idx in test_trials.keys()
    }
    per_session_length_stats, global_length_stats = collect_test_split_length_stats(
        args_cfg,
        test_trials,
    )

    result = {
        "model_path": str(run_dir),
        "checkpoints_dir": str(checkpoints_dir),
        "split_number": int(split_number),
        "seed": int(cli_args.seed),
        "split_csv_path": str(split_csv_path),
        "avg_PER": float(metrics["avg_PER"]),
        "avg_loss": float(metrics["avg_loss"]),
        "day_PERs": metrics["day_PERs"],
        "per_session_test_counts": per_session_counts,
        "per_session_length_stats": per_session_length_stats,
        "global_length_stats": global_length_stats,
        "ctc_evidence_path": metrics.get("ctc_evidence_path"),
        "latency_mean_ms": float(np.mean(latencies)) if latencies.size > 0 else None,
        "latency_median_ms": float(np.median(latencies)) if latencies.size > 0 else None,
        "latency_p95_ms": float(np.percentile(latencies, 95)) if latencies.size > 0 else None,
        "latency_warmup_utterances": int(cli_args.latency_warmup),
        "latency_effective_lengths": latency_effective_lengths.astype(int).tolist(),
        "latency_per_effective_step_mean_ms": (
            float(np.mean(latency_per_effective_step))
            if latency_per_effective_step.size > 0
            else None
        ),
        "latency_per_effective_step_median_ms": (
            float(np.median(latency_per_effective_step))
            if latency_per_effective_step.size > 0
            else None
        ),
        "latency_per_effective_step_p95_ms": (
            float(np.percentile(latency_per_effective_step, 95))
            if latency_per_effective_step.size > 0
            else None
        ),
        "current_allocated_mb": (
            float(current_allocated_mb) if current_allocated_mb is not None else None
        ),
        "max_allocated_mb": (
            float(max_allocated_mb) if max_allocated_mb is not None else None
        ),
    }
    if export_path is not None:
        export_payload = {
            "trials": metrics["trial_exports"],
            "phoneme_classes": metrics["phoneme_classes"],
        }
        export_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(export_payload, export_path)
        result["validation_export_pt_path"] = str(export_path)
        result["validation_export_includes_alignment"] = bool(cli_args.include_alignment)

    output_path = checkpoints_dir / f"daywise_test_split_{split_number:02d}_metrics.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print(f"Model path: {run_dir}")
    print(f"Split number: {split_number}")
    print(f"Seed: {cli_args.seed}")
    print(f"Split CSV: {split_csv_path}")
    print(f"Average PER: {metrics['avg_PER']:.5f}")
    print(f"Average CTC loss: {metrics['avg_loss']:.5f}")
    if latencies.size > 0:
        print(f"Average latency: {np.mean(latencies):.3f} ms")
        print(f"Latency mean: {np.mean(latencies):.3f} ms")
        print(f"Latency median: {np.median(latencies):.3f} ms")
        print(f"Latency p95: {np.percentile(latencies, 95):.3f} ms")
        print(f"Latency warmup utterances: {cli_args.latency_warmup}")
        if latency_per_effective_step.size > 0:
            print(f"Latency per effective step mean: {np.mean(latency_per_effective_step):.6f} ms")
            print(f"Latency per effective step median: {np.median(latency_per_effective_step):.6f} ms")
            print(f"Latency per effective step p95: {np.percentile(latency_per_effective_step, 95):.6f} ms")
    if current_allocated_mb is not None:
        print(f"Current allocated: {current_allocated_mb:.2f} MB")
        print(f"Max allocated: {max_allocated_mb:.2f} MB")
    else:
        print("Current allocated: N/A (CPU)")
        print("Max allocated: N/A (CPU)")
    if export_path is not None:
        print(f"Saved validation export PT to: {export_path}")
        print(f"Included CTC forced alignment: {cli_args.include_alignment}")
    print(f"Saved metrics to: {output_path}")
    print(f"Total test utterances: {global_length_stats['num_test_utterances']}")
    print(f"Mean raw length: {global_length_stats['mean_raw_length']:.2f}")
    print(f"Mean effective length: {global_length_stats['mean_effective_length']:.2f}")

    sampled_examples = collect_random_phoneme_examples(
        metrics,
        sample_count=10,
        seed=cli_args.seed,
    )
    if sampled_examples:
        print()
        print(
            f"Random phoneme examples ({len(sampled_examples)} utterances, seeded with {cli_args.seed}):"
        )
        for example_index, example in enumerate(sampled_examples, start=1):
            print(
                f"[{example_index}] day={example['day']} trial={example['trial']} "
                f"PER={example['per']:.5f}"
            )
            print(f"  ground truth: {example['target_phonemes']}")
            print(f"  decoded     : {example['decoded_phonemes']}")
            print()
    else:
        print()
        print("Random phoneme examples: none available.")


if __name__ == "__main__":
    main()
