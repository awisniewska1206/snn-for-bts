import argparse
import os
import sys

import torch
from omegaconf import OmegaConf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_TRAINING_DIR = os.path.dirname(SCRIPT_DIR)
if MODEL_TRAINING_DIR not in sys.path:
    sys.path.insert(0, MODEL_TRAINING_DIR)

from trainer import BrainToTextDecoder_Trainer
from utils.seeding import set_global_seed


def main():
    parser = argparse.ArgumentParser(description="Run inference-only lag ablation at detected CTC evidence peaks.")
    parser.add_argument("--trainer_config", required=True, help="Path to the trainer config yaml for the trained run.")
    parser.add_argument("--checkpoint_path", required=True, help="Path to the trained checkpoint to load.")
    parser.add_argument("--evidence_path", required=True, help="Path to the saved CTC evidence .pt/.pkl/.npz file.")
    parser.add_argument("--output_path", default=None, help="Where to save the lag ablation analysis .pt file.")
    parser.add_argument("--snn_config", default=None, help="Optional SNN config yaml if not embedded in trainer config.")
    parser.add_argument("--gpu", default="0", help="GPU index to use. Set to -1 for CPU.")
    parser.add_argument("--block_idx", type=int, default=0, help="Which lagged block to ablate.")
    parser.add_argument("--window_radius", type=int, default=2, help="Local window radius around each peak.")
    parser.add_argument("--top_peak_min_prob", type=float, default=0.35, help="Minimum top non-blank probability for peak detection.")
    parser.add_argument("--blank_probability_max", type=float, default=None, help="Optional maximum blank probability allowed at a detected peak.")
    parser.add_argument("--min_peak_distance", type=int, default=3, help="Minimum spacing between detected peaks.")
    parser.add_argument("--max_trials", type=int, default=None, help="Optional cap on number of trials to analyze.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for deterministic analysis setup.")
    parser.add_argument(
        "--ignore_saved_peak_indices",
        action="store_true",
        help="If set, always redetect peaks from the saved timestep probabilities instead of using peak_indices from the evidence file.",
    )
    args = parser.parse_args()

    print("[lag-ablation] starting analysis")
    print(f"[lag-ablation] trainer_config={args.trainer_config}")
    print(f"[lag-ablation] checkpoint_path={args.checkpoint_path}")
    print(f"[lag-ablation] evidence_path={args.evidence_path}")
    print(
        "[lag-ablation] settings: "
        f"block_idx={args.block_idx} window_radius={args.window_radius} "
        f"top_peak_min_prob={args.top_peak_min_prob} blank_probability_max={args.blank_probability_max} "
        f"min_peak_distance={args.min_peak_distance} max_trials={args.max_trials} "
        "include_only_lag0=True"
    )

    set_global_seed(args.seed)

    if args.gpu == "-1":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    trainer_cfg = OmegaConf.load(args.trainer_config)
    trainer_cfg.mode = "eval"
    trainer_cfg.model_type = "snn"
    trainer_cfg.init_from_checkpoint = True
    trainer_cfg.init_checkpoint_path = args.checkpoint_path
    trainer_cfg.save_best_checkpoint = False
    trainer_cfg.save_all_val_steps = False
    trainer_cfg.save_final_model = False
    trainer_cfg.save_val_metrics = False
    trainer_cfg.save_val_logits = False
    trainer_cfg.save_val_data = False
    trainer_cfg.use_amp = False
    trainer_cfg.seed = int(args.seed)
    trainer_cfg.gpu_number = 0 if args.gpu != "-1" else -1
    if "snn_config" in trainer_cfg and trainer_cfg.snn_config is not None:
        trainer_cfg.snn_config.seed = int(args.seed)

    if args.output_path:
        output_root = os.path.dirname(args.output_path) or "."
    else:
        output_root = os.path.join(
            os.path.dirname(args.evidence_path),
            "lag_peak_ablation",
        )
    os.makedirs(output_root, exist_ok=True)

    trainer_cfg.output_dir = output_root
    trainer_cfg.checkpoint_dir = os.path.join(output_root, "checkpoints")
    trainer_cfg.log_dir = os.path.join(output_root, "logs")
    os.makedirs(trainer_cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(trainer_cfg.log_dir, exist_ok=True)

    if args.snn_config is not None:
        trainer_cfg.snn_config = OmegaConf.load(args.snn_config)
        trainer_cfg.snn_config.seed = int(args.seed)
        print(f"[lag-ablation] loaded snn_config={args.snn_config}")

    print(f"[lag-ablation] output_root={output_root}")
    print("[lag-ablation] building trainer and loading checkpoint")

    trainer = BrainToTextDecoder_Trainer(trainer_cfg)
    try:
        print("[lag-ablation] trainer ready, running lag peak ablation")
        output_path = trainer.run_lag_peak_ablation(
            evidence_path=args.evidence_path,
            output_path=args.output_path,
            block_idx=args.block_idx,
            window_radius=args.window_radius,
            top_peak_min_prob=args.top_peak_min_prob,
            blank_probability_max=args.blank_probability_max,
            min_peak_distance=args.min_peak_distance,
            use_saved_peak_indices=not args.ignore_saved_peak_indices,
            max_trials=args.max_trials,
        )
        print(f"Saved lag peak ablation analysis to: {output_path}")
        results = torch.load(output_path, map_location="cpu", weights_only=False)
        metadata = results.get("metadata", {})
        print(
            "[lag-ablation] result summary: "
            f"selected_trials={metadata.get('num_selected_trials')} "
            f"matched_trials={metadata.get('num_matched_trials')} "
            f"matched_batches={metadata.get('num_batches_with_targets')} "
            f"detected_peaks={metadata.get('num_detected_peaks')} "
            f"in_range_peaks={metadata.get('num_in_range_peaks')} "
            f"saved_peak_records={metadata.get('num_peak_records')}"
        )
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
