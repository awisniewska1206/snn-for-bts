import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from omegaconf import OmegaConf


SEEDS = [0, 1, 2, 3, 4]
DELAY_SECONDS = 5

SCRIPT_DIR = Path(__file__).resolve().parent
MAIN_PATH = SCRIPT_DIR / "main.py"
LOG_PATH = SCRIPT_DIR / "experiment_log.csv"

BASE_CONFIGS = {
    "mlp": SCRIPT_DIR / "configs" / "trainer_args_mlp.yaml",
    "lstm": SCRIPT_DIR / "configs" / "trainer_args_lstm.yaml",
    "rnn": SCRIPT_DIR / "configs" / "trainer_args_rnn.yaml",
    "snn": SCRIPT_DIR / "configs" / "trainer_args_snn_new.yaml",
}

# Edit these overrides to match how you want to define small/medium/large.
SIZE_OVERRIDES = {
    "mlp": {
        "large": {
            "model.mlp_hidden_dim": 512,
            "model.n_layers": 2,
            "model.patch_size": 8,
        },
    },
    "rnn": {
        "small": {
            "model.n_units": 256,
            "model.n_layers": 3,
        },
        "medium": {
            "model.n_units": 128,
            "model.n_layers": 3,
        },
        "large": {
            "model.n_units": 768,
            "model.n_layers": 5,
        },
    },
    "lstm": {
        "small": {
            "model.n_units": 256,
            "model.n_layers": 3,
        },
        "medium": {
            "model.n_units": 128,
            "model.n_layers": 3,
        },
        "large": {
            "model.n_units": 768,
            "model.n_layers": 5,
        },
    },
    "snn": {
        "small": {
            "model.n_units": 256,
            "model.n_layers": 2,
        },
        "medium": {
            "model.n_units": 384,
            "model.n_layers": 2,
        },
        "large": {
            "model.n_units": 512,
            "model.n_layers": 3,
        },
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run one model family sequentially across size presets and seeds.")
    parser.add_argument("--model", choices=sorted(BASE_CONFIGS.keys()), required=True)
    parser.add_argument("--gpu", required=True)
    return parser.parse_args()


def ensure_log_header(path: Path):
    if path.exists():
        return

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model_family", "size_config", "seed", "tag", "status"])


def append_log_row(path: Path, model_family: str, size_config: str, seed: int, tag: str, status: str):
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([model_family, size_config, seed, tag, status])


def build_command(model: str, gpu: str, config_path: Path, tag: str, seed: int):
    return [
        sys.executable,
        str(MAIN_PATH),
        "--model",
        model,
        "--trainer_config",
        str(config_path),
        "--gpu",
        gpu,
        "--tag",
        tag,
        "--seed",
        str(seed),
    ]


def make_temp_config(model: str, size_name: str, output_path: Path):
    base_config_path = BASE_CONFIGS[model]
    config = OmegaConf.load(base_config_path)

    for key, value in SIZE_OVERRIDES[model][size_name].items():
        OmegaConf.update(config, key, value, merge=False)

    OmegaConf.save(config, output_path)
    return output_path


def main():
    args = parse_args()
    base_config_path = BASE_CONFIGS[args.model]
    size_names = list(SIZE_OVERRIDES[args.model].keys())
    total_runs = len(size_names) * len(SEEDS)

    ensure_log_header(LOG_PATH)

    print(f"Running {total_runs} experiment(s) for model family '{args.model}' on GPU {args.gpu}")
    print(f"Base config: {base_config_path}")
    print(f"Logging to {LOG_PATH}")

    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_config_path}")

    with TemporaryDirectory(prefix=f"{args.model}_family_runs_", dir=SCRIPT_DIR) as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        run_index = 0

        for size_name in size_names:
            temp_config_path = temp_dir / f"{args.model}_{size_name}.yaml"
            make_temp_config(args.model, size_name, temp_config_path)

            for seed in SEEDS:
                run_index += 1
                tag = f"{args.model}_{size_name}_seed{seed}"

                print(f"\n[{run_index}/{total_runs}] Starting {tag}")
                print(f"Generated config: {temp_config_path}")

                command = build_command(
                    model=args.model,
                    gpu=args.gpu,
                    config_path=temp_config_path,
                    tag=tag,
                    seed=seed,
                )

                try:
                    result = subprocess.run(command, cwd=SCRIPT_DIR, check=False)
                    status = "success" if result.returncode == 0 else "fail"
                    print(f"[{run_index}/{total_runs}] Finished {tag} with status={status}")
                except Exception as exc:
                    status = "fail"
                    print(f"[{run_index}/{total_runs}] Crash while running {tag}: {exc}")

                append_log_row(LOG_PATH, args.model, size_name, seed, tag, status)

                if run_index < total_runs:
                    print(f"Sleeping {DELAY_SECONDS}s before next run...")
                    time.sleep(DELAY_SECONDS)


if __name__ == "__main__":
    main()
