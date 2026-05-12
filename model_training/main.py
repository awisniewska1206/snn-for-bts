import os
import json
import time
import shutil
import logging
import argparse
from datetime import datetime

import torch
from omegaconf import OmegaConf

from trainer import BrainToTextDecoder_Trainer
from utils.model_config import canonicalize_model_type, ensure_model_config_schema
from utils.seeding import set_global_seed


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_text(path: str, text: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def remove_dir_if_exists(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)


def cleanup_tmp_root(tmp_root="logs/.tmp"):
    if not os.path.exists(tmp_root):
        return
    for name in os.listdir(tmp_root):
        path = os.path.join(tmp_root, name)
        if os.path.isdir(path):
            print(f"[cleanup] removing stale temp run: {path}")
            shutil.rmtree(path)


def move_dir_with_retry(src: str, dst: str, retries: int = 5, delay: float = 1.0):
    last_err = None
    for _ in range(retries):
        try:
            shutil.move(src, dst)
            return
        except PermissionError as e:
            last_err = e
            time.sleep(delay)
    raise last_err


def remove_dir_with_retry(path: str, retries: int = 5, delay: float = 1.0):
    if not os.path.exists(path):
        return
    last_err = None
    for _ in range(retries):
        try:
            shutil.rmtree(path)
            return
        except PermissionError as e:
            last_err = e
            time.sleep(delay)
    raise last_err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        help="Model family to train. Use 'gru', 'snn', 'mlp', 'lstm', 'tcn', or 'lssca_snn'. 'rnn' is accepted as a legacy alias for 'gru'.",
    )
    parser.add_argument("--trainer_config", required=True)
    parser.add_argument("--snn_config", default=None)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--seed", type=int, default=None, help="Optional override for the run seed.")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--tag", default="", help="Experiment tag/comment to write into the training log")
    parser.add_argument(
        "--cleanup_tmp_on_start",
        action="store_true",
        help="Remove runs under logs/.tmp before starting. Disabled by default to avoid deleting active runs in other terminals.",
    )
    args_cli = parser.parse_args()

    if args_cli.cleanup_tmp_on_start:
        cleanup_tmp_root()

    os.environ["CUDA_VISIBLE_DEVICES"] = args_cli.gpu

    model_name = canonicalize_model_type(args_cli.model)
    valid_models = {"gru", "snn", "mlp", "lstm", "tcn", "lssca_snn"}
    if model_name not in valid_models:
        raise ValueError(f"Unsupported model '{args_cli.model}'. Expected one of: {sorted(valid_models)}")

    trainer_cfg = OmegaConf.load(args_cli.trainer_config)
    trainer_cfg.mode = "train"
    trainer_cfg.gpu_number = args_cli.gpu
    trainer_cfg.model_type = model_name
    trainer_cfg.tag = args_cli.tag
    ensure_model_config_schema(trainer_cfg.model)
    if args_cli.seed is not None:
        trainer_cfg.seed = int(args_cli.seed)

    snn_cfg = None
    if model_name == "snn":
        if args_cli.snn_config is None:
            raise ValueError("--snn_config is required when --model snn")
        snn_cfg = OmegaConf.load(args_cli.snn_config)
        snn_cfg.seed = int(trainer_cfg.seed)
        trainer_cfg.snn_config = snn_cfg

    run_seed = set_global_seed(int(trainer_cfg.seed))

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.9, device=0)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    suffix = args_cli.run_name if args_cli.run_name else model_name
    run_name = f"{ts}_{suffix}_pid{os.getpid()}"

    final_root = f"logs/{model_name}"
    tmp_root = "logs/.tmp"

    final_output_dir = os.path.join(final_root, run_name)
    tmp_output_dir = os.path.join(tmp_root, run_name)

    checkpoint_dir = os.path.join(tmp_output_dir, "checkpoints")
    log_dir = os.path.join(tmp_output_dir, "logs")
    config_dir = os.path.join(tmp_output_dir, "configs")

    remove_dir_if_exists(tmp_output_dir)

    ensure_dir(checkpoint_dir)
    ensure_dir(log_dir)
    ensure_dir(config_dir)

    trainer_cfg.output_dir = tmp_output_dir
    trainer_cfg.checkpoint_dir = checkpoint_dir
    trainer_cfg.log_dir = log_dir

    OmegaConf.save(trainer_cfg, os.path.join(config_dir, "trainer_config_resolved.yaml"))
    shutil.copy2(args_cli.trainer_config, os.path.join(config_dir, "trainer_config_input.yaml"))

    if snn_cfg is not None:
        OmegaConf.save(snn_cfg, os.path.join(config_dir, "snn_config_resolved.yaml"))
        shutil.copy2(args_cli.snn_config, os.path.join(config_dir, "snn_config_input.yaml"))

    launch_info = {
        "timestamp": ts,
        "model": model_name,
        "requested_model": args_cli.model,
        "gpu": args_cli.gpu,
        "seed": run_seed,
        "trainer_config": args_cli.trainer_config,
        "snn_config": args_cli.snn_config,
        "tag": args_cli.tag,
        "tmp_output_dir": tmp_output_dir,
        "final_output_dir": final_output_dir,
        "run_name": run_name,
    }
    save_text(os.path.join(tmp_output_dir, "launch_info.json"), json.dumps(launch_info, indent=2))

    trainer = None
    try:
        trainer = BrainToTextDecoder_Trainer(trainer_cfg)
        trainer.train()

        if trainer is not None:
            trainer.close()

        logging.shutdown()
        time.sleep(1.0)

        save_text(os.path.join(tmp_output_dir, "SUCCESS"), "ok\n")

        ensure_dir(final_root)
        remove_dir_if_exists(final_output_dir)
        move_dir_with_retry(tmp_output_dir, final_output_dir)

        print(f"[SUCCESS] saved to: {final_output_dir}")

    except KeyboardInterrupt:
        print("[INTERRUPTED] removing temp run")
        if trainer is not None:
            trainer.close()
        logging.shutdown()
        time.sleep(1.0)
        remove_dir_with_retry(tmp_output_dir)
        raise

    except Exception:
        print("[FAILED] removing temp run")
        if trainer is not None:
            trainer.close()
        logging.shutdown()
        time.sleep(1.0)
        remove_dir_with_retry(tmp_output_dir)
        raise


if __name__ == "__main__":
    main()
