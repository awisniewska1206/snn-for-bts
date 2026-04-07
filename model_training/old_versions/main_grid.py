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

# run it as python main_grid.py --model snn --trainer_config configs/trainer_args_snn.yaml --snn_config configs/snn_args_alternative_core.yaml --patch_sizes "4,6,8" --patch_strides "2,4"


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


def parse_int_list(x: str):
    return [int(v.strip()) for v in x.split(",") if v.strip() != ""]


def make_run_name(model_name: str, run_name: str | None, ps: int, st: int):
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    suffix = run_name if run_name else model_name
    return f"{ts}_{suffix}_ps{ps}_st{st}_pid{os.getpid()}"


def run_one_config(args_cli, patch_size: int, patch_stride: int):
    trainer_cfg = OmegaConf.load(args_cli.trainer_config)
    trainer_cfg.mode = "train"
    trainer_cfg.gpu_number = args_cli.gpu
    trainer_cfg.model_type = args_cli.model

    snn_cfg = None
    if args_cli.model == "snn":
        if args_cli.snn_config is None:
            raise ValueError("--snn_config is required when --model snn")
        snn_cfg = OmegaConf.load(args_cli.snn_config)
        trainer_cfg.snn_config = snn_cfg

    # Apply grid values
    trainer_cfg.model.patch_size = patch_size
    trainer_cfg.model.patch_stride = patch_stride

    run_name = make_run_name(args_cli.model, args_cli.run_name, patch_size, patch_stride)

    final_root = f"logs/{args_cli.model}"
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
        "timestamp": datetime.now().isoformat(),
        "model": args_cli.model,
        "gpu": args_cli.gpu,
        "trainer_config": args_cli.trainer_config,
        "snn_config": args_cli.snn_config,
        "tmp_output_dir": tmp_output_dir,
        "final_output_dir": final_output_dir,
        "run_name": run_name,
        "patch_size": patch_size,
        "patch_stride": patch_stride,
    }
    save_text(os.path.join(tmp_output_dir, "launch_info.json"), json.dumps(launch_info, indent=2))

    trainer = None
    try:
        print(f"\n[RUN] patch_size={patch_size}, patch_stride={patch_stride}")
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
        return {"patch_size": patch_size, "patch_stride": patch_stride, "status": "success", "dir": final_output_dir}

    except KeyboardInterrupt:
        print("[INTERRUPTED] removing temp run")
        if trainer is not None:
            trainer.close()
        logging.shutdown()
        time.sleep(1.0)
        remove_dir_with_retry(tmp_output_dir)
        raise

    except Exception as e:
        print(f"[FAILED] patch_size={patch_size}, patch_stride={patch_stride}")
        if trainer is not None:
            trainer.close()
        logging.shutdown()
        time.sleep(1.0)
        remove_dir_with_retry(tmp_output_dir)
        return {"patch_size": patch_size, "patch_stride": patch_stride, "status": "failed", "error": repr(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["rnn", "snn"], required=True)
    parser.add_argument("--trainer_config", required=True)
    parser.add_argument("--snn_config", default=None)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--run_name", default=None)
    parser.add_argument(
        "--cleanup_tmp_on_start",
        action="store_true",
        help="Remove runs under logs/.tmp before starting.",
    )

    # New: small grid search args
    parser.add_argument("--patch_sizes", default=None, help='Comma-separated list, e.g. "4,6,8"')
    parser.add_argument("--patch_strides", default=None, help='Comma-separated list, e.g. "2,4"')

    args_cli = parser.parse_args()

    if args_cli.cleanup_tmp_on_start:
        cleanup_tmp_root()

    os.environ["CUDA_VISIBLE_DEVICES"] = args_cli.gpu
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.9, device=0)

    # Default: single run from config
    if args_cli.patch_sizes is None or args_cli.patch_strides is None:
        base_cfg = OmegaConf.load(args_cli.trainer_config)
        patch_size = int(base_cfg.model.patch_size)
        patch_stride = int(base_cfg.model.patch_stride)
        results = [run_one_config(args_cli, patch_size, patch_stride)]
    else:
        patch_sizes = parse_int_list(args_cli.patch_sizes)
        patch_strides = parse_int_list(args_cli.patch_strides)

        results = []
        for ps in patch_sizes:
            for st in patch_strides:
                results.append(run_one_config(args_cli, ps, st))

    print("\n=== GRID RESULTS ===")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()