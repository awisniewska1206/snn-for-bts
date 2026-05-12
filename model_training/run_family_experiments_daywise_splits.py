import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from omegaconf import OmegaConf

from utils.model_config import canonicalize_model_type

# python run_family_experiments_daywise_splits.py   --model snn   --gpu 0   --seed 20   --split_dir data/daywise_sentence_splits --split_iterations 4
GLOBAL_SEED = 20
DEFAULT_SPLIT_ITERATIONS = [0]
DELAY_SECONDS = 5

SCRIPT_DIR = Path(__file__).resolve().parent
MAIN_PATH = SCRIPT_DIR / "main.py"
LOG_PATH = SCRIPT_DIR / "experiment_log_daywise_splits.csv"
SPLIT_GENERATOR_PATH = SCRIPT_DIR / "create_daywise_sentence_splits.py"

BASE_CONFIGS = {
    "mlp": SCRIPT_DIR / "configs" / "trainer_args_mlp.yaml",
    "lstm": SCRIPT_DIR / "configs" / "trainer_args_lstm.yaml",
    "tcn": SCRIPT_DIR / "configs" / "trainer_args_tcn.yaml",
    "lssca_snn": SCRIPT_DIR / "configs" / "trainer_args_lssca_snn.yaml",
    "gru": SCRIPT_DIR / "configs" / "trainer_args_gru.yaml",
    "snn": SCRIPT_DIR / "configs" / "trainer_args_snn_best.yaml",
}

SNN_CONFIGS = {
    "snn": SCRIPT_DIR / "configs" / "spikachu_alternative.yaml",
}

SIZE_OVERRIDES = {
    "snn": {
        "lag_01": {
            "model.hidden_size": 256,
            "model.n_layers": 4,
            "model.dropout": 0.06387528313160899,
            "model.patch_size": 6,
            "weight_decay": 0.0005391942570602405,
            "grad_norm_clip_value": 5.967032507444588,
            "lr_max": 0.011110330581942046,
            "lr_min": 6.583797432981374e-05,
            "lr_max_day": 0.011110330581942046,
            "lr_min_day": 6.583797432981374e-05,
        },

    },
    "mlp": {
        "small_budget": {
            "model.hidden_size": 384,
            "model.n_layers": 5,
            "model.dropout": 0.001090402834570424,
            "model.patch_size": 8,
            "trainer.weight_decay": 0.0008094100979307726,
            "trainer.grad_norm_clip": 4.897937298822798,
            "trainer.lr_max": 0.005654100087911855,
            "trainer.lr_min": 0.0017262190995382806,
        },
    },
    "gru": {
        "small_budget": {
            "model.hidden_size": 256,
            "model.n_layers": 3,
            "model.dropout": 0.2552548631132045,
            "model.patch_size": 10,
            "trainer.weight_decay": 0.0007173991343375181,
            "trainer.grad_norm_clip": 7.227485741403328,
            "trainer.lr_max": 0.00888453833491444,
            "trainer.lr_min": 0.00018387437167787326,
        },
    },
    "lstm": {
        "small_budget": {
            "model.hidden_size": 256,
            "model.n_layers": 4,
            "model.dropout": 0.21102527497158424,
            "model.patch_size": 6,
            "trainer.weight_decay": 0.000984986985451823,
            "trainer.grad_norm_clip": 8.661375999391652,
            "trainer.lr_max": 0.002355335301417049,
            "trainer.lr_min": 0.002111578582092775,
        },
    },
    "tcn": {
        "small_budget": {
            "model.hidden_size": 64,
            "model.n_layers": 4,
            "model.dropout": 0.08401562756298117,
            "model.patch_size": 10,
            "model.tcn_kernel_size": 7,
            "model.tcn_dilation_base": 1,
            "trainer.weight_decay": 0.0005274056383191394,
            "trainer.grad_norm_clip": 7.3424066473083816,
            "trainer.lr_max": 0.004613735707427554,
            "trainer.lr_min": 0.00020715672084508095,
        },
    },
}

SNN_CONFIG_OVERRIDES = {
    "lag_01" : {
        "model.setup.hidden_size": 256,
        "model.setup.patch_size": 6,
        "model.setup.patch_stride": 4,
        "model.setup.snn_core.tau": 6.471620099063689,
        "model.setup.snn_core.v_threshold": 0.8203519640943726,
        "model.setup.snn_core.n_blocks": 4,
        "model.setup.snn_core.p_drop": 0.06387528313160899,
        #"model.setup.snn_core.lagged_lags": [0, 1],
        #"model.setup.snn_core.lagged_one_random_lag_per_pair": True,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one model family sequentially across size presets and five day-wise split iterations."
    )
    parser.add_argument("--model", choices=sorted(list(BASE_CONFIGS.keys()) + ["rnn"]), required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument(
        "--seed",
        type=int,
        default=GLOBAL_SEED,
        help="Global seed used both for training and for generating day-wise split files.",
    )
    parser.add_argument(
        "--split_dir",
        default=str(Path("data") / "daywise_sentence_splits"),
        help="Directory containing iteration_XX_splits.csv files.",
    )
    parser.add_argument(
        "--csv_path",
        default=str(Path("data") / "t15_copyTaskData_description.csv"),
        help="Source CSV used to generate split files if they are missing.",
    )
    parser.add_argument(
        "--split_iterations",
        nargs="+",
        type=int,
        default=DEFAULT_SPLIT_ITERATIONS,
        help="One or more split indices to run, e.g. --split_iterations 0 or --split_iterations 0 2 4.",
    )
    parser.add_argument(
        "--snn_config",
        default=None,
        help="Optional explicit SNN config YAML. If omitted for --model snn, a default config is used.",
    )
    return parser.parse_args()


def ensure_log_header(path: Path):
    if path.exists():
        return

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model_family", "size_config", "split_iteration", "seed", "tag", "status"])


def append_log_row(path: Path, model_family: str, size_config: str, split_iteration: int, seed: int, tag: str, status: str):
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([model_family, size_config, split_iteration, seed, tag, status])


def build_command(model: str, gpu: str, config_path: Path, tag: str, seed: int, snn_config_path: Path | None = None):
    command = [
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
    if model == "snn":
        if snn_config_path is None:
            raise ValueError("snn_config_path is required when model='snn'")
        command.extend(["--snn_config", str(snn_config_path)])
    return command


def session_name_to_date(session_name: str) -> str:
    parts = session_name.split(".")
    if len(parts) < 4:
        raise ValueError(f"Session name does not match expected pattern: {session_name}")
    return f"{parts[1]}-{parts[2]}-{parts[3]}"


def normalize_split_iterations(split_iterations):
    normalized = []
    seen = set()
    for split_iteration in split_iterations:
        split_iteration = int(split_iteration)
        if split_iteration < 0:
            raise ValueError(f"Split iterations must be non-negative, got {split_iteration}")
        if split_iteration in seen:
            continue
        seen.add(split_iteration)
        normalized.append(split_iteration)

    if not normalized:
        raise ValueError("At least one split iteration must be provided")

    return normalized


def ensure_split_files(split_dir: Path, csv_path: Path, seed: int, split_iterations):
    expected = [split_dir / f"iteration_{i:02d}_splits.csv" for i in split_iterations]
    if all(path.exists() for path in expected):
        return

    split_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(SPLIT_GENERATOR_PATH),
        "--csv_path",
        str(csv_path),
        "--output_dir",
        str(split_dir),
        "--seed",
        str(seed),
    ]
    result = subprocess.run(command, cwd=SCRIPT_DIR, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to generate split files in {split_dir}")
    missing = [path for path in expected if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Requested split files were not generated: "
            + ", ".join(str(path.name) for path in missing)
        )


def compute_dataset_probability_val(split_csv_path: Path, sessions):
    counts_by_date = {}
    with split_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["split"] != "val":
                continue
            counts_by_date[row["date"]] = counts_by_date.get(row["date"], 0) + 1

    return [
        1 if counts_by_date.get(session_name_to_date(session_name), 0) > 0 else 0
        for session_name in sessions
    ]


def apply_overrides(config, overrides):
    for key, value in overrides.items():
        OmegaConf.update(config, key, value, merge=False)


def sync_snn_trainer_patch_config(config, size_name: str):
    snn_overrides = SNN_CONFIG_OVERRIDES.get(size_name, {})
    patch_size = snn_overrides.get("model.setup.patch_size")
    patch_stride = snn_overrides.get("model.setup.patch_stride")

    if patch_size is not None:
        OmegaConf.update(config, "model.patch_size", int(patch_size), merge=False)
    if patch_stride is not None:
        OmegaConf.update(config, "model.patch_stride", int(patch_stride), merge=False)


def make_temp_config(model: str, size_name: str, split_csv_path: Path, seed: int, output_path: Path):
    base_config_path = BASE_CONFIGS[model]
    config = OmegaConf.load(base_config_path)

    apply_overrides(config, SIZE_OVERRIDES.get(model, {}).get(size_name, {}))
    if model == "snn":
        sync_snn_trainer_patch_config(config, size_name)

    OmegaConf.update(config, "seed", int(seed), merge=False)
    OmegaConf.update(config, "dataset.custom_split_csv", str(split_csv_path), merge=False)
    OmegaConf.update(
        config,
        "dataset.custom_source_filenames",
        ["data_train.hdf5", "data_val.hdf5"],
        merge=False,
    )
    dataset_probability_val = compute_dataset_probability_val(
        split_csv_path=split_csv_path,
        sessions=list(config.dataset.sessions),
    )
    OmegaConf.update(config, "dataset.dataset_probability_val", dataset_probability_val, merge=False)

    OmegaConf.save(config, output_path)
    return output_path


def make_temp_snn_config(size_name: str, base_snn_config_path: Path, seed: int, output_path: Path):
    config = OmegaConf.load(base_snn_config_path)
    apply_overrides(config, SNN_CONFIG_OVERRIDES[size_name])
    OmegaConf.update(config, "seed", int(seed), merge=False)
    OmegaConf.save(config, output_path)
    return output_path


def main():
    args = parse_args()
    model_name = canonicalize_model_type(args.model)
    base_config_path = BASE_CONFIGS[model_name]
    size_names = list(SNN_CONFIG_OVERRIDES.keys()) if model_name == "snn" else list(SIZE_OVERRIDES[model_name].keys())
    split_iterations = normalize_split_iterations(args.split_iterations)
    split_dir = (SCRIPT_DIR.parent / args.split_dir).resolve() if not Path(args.split_dir).is_absolute() else Path(args.split_dir)
    csv_path = (SCRIPT_DIR.parent / args.csv_path).resolve() if not Path(args.csv_path).is_absolute() else Path(args.csv_path)

    total_runs = len(size_names) * len(split_iterations)

    ensure_log_header(LOG_PATH)

    print(f"Running {total_runs} experiment(s) for model family '{model_name}' on GPU {args.gpu}")
    print(f"Base config: {base_config_path}")
    print(f"Global seed: {args.seed}")
    print(f"Split iterations: {split_iterations}")
    print(f"Split directory: {split_dir}")
    print(f"Logging to {LOG_PATH}")

    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_config_path}")

    snn_config_path = None
    if model_name == "snn":
        snn_config_path = Path(args.snn_config) if args.snn_config else SNN_CONFIGS["snn"]
        if not snn_config_path.is_absolute():
            snn_config_path = (SCRIPT_DIR / snn_config_path).resolve()
        if not snn_config_path.exists():
            raise FileNotFoundError(f"SNN config not found: {snn_config_path}")
        print(f"SNN config: {snn_config_path}")

    ensure_split_files(
        split_dir=split_dir,
        csv_path=csv_path,
        seed=args.seed,
        split_iterations=split_iterations,
    )

    with TemporaryDirectory(prefix=f"{model_name}_family_daywise_", dir=SCRIPT_DIR) as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        run_index = 0

        for size_name in size_names:
            for split_iteration in split_iterations:
                split_csv_path = split_dir / f"iteration_{split_iteration:02d}_splits.csv"
                temp_config_path = temp_dir / f"{model_name}_{size_name}_split{split_iteration}.yaml"
                make_temp_config(
                    model=model_name,
                    size_name=size_name,
                    split_csv_path=split_csv_path,
                    seed=args.seed,
                    output_path=temp_config_path,
                )
                temp_snn_config_path = snn_config_path
                if model_name == "snn":
                    temp_snn_config_path = temp_dir / f"{model_name}_{size_name}_split{split_iteration}_snn.yaml"
                    make_temp_snn_config(
                        size_name=size_name,
                        base_snn_config_path=snn_config_path,
                        seed=args.seed,
                        output_path=temp_snn_config_path,
                    )

                run_index += 1
                tag = f"{model_name}_{size_name}_split{split_iteration}_seed{args.seed}"

                print(f"\n[{run_index}/{total_runs}] Starting {tag}")
                print(f"Generated config: {temp_config_path}")
                print(f"Using split CSV: {split_csv_path}")
                if model_name == "snn":
                    print(f"Generated SNN config: {temp_snn_config_path}")

                command = build_command(
                    model=model_name,
                    gpu=args.gpu,
                    config_path=temp_config_path,
                    tag=tag,
                    seed=args.seed,
                    snn_config_path=temp_snn_config_path,
                )

                try:
                    result = subprocess.run(command, cwd=SCRIPT_DIR, check=False)
                    status = "success" if result.returncode == 0 else "fail"
                    print(f"[{run_index}/{total_runs}] Finished {tag} with status={status}")
                except Exception as exc:
                    status = "fail"
                    print(f"[{run_index}/{total_runs}] Crash while running {tag}: {exc}")

                append_log_row(LOG_PATH, model_name, size_name, split_iteration, args.seed, tag, status)

                if run_index < total_runs:
                    print(f"Sleeping {DELAY_SECONDS}s before next run...")
                    time.sleep(DELAY_SECONDS)


if __name__ == "__main__":
    main()
