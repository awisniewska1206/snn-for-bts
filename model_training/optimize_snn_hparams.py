#!/usr/bin/env python3
import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

try:
    from omegaconf import OmegaConf
except ImportError:
    OmegaConf = None

from utils.model_config import canonicalize_model_type, ensure_model_config_schema

# python model_training/optimize_snn_hparams.py --model_type snn --gpu 0

DAILY_PARAMS = 11_819_520

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DAYWISE_SPLIT_CSV = (
    SCRIPT_DIR.parent / "data" / "daywise_sentence_splits" / "iteration_00_splits.csv"
)
DEFAULT_TRAINER_CONFIGS = {
    "snn": SCRIPT_DIR / "configs" / "trainer_args_snn.yaml",
    "gru": SCRIPT_DIR / "configs" / "trainer_args_gru.yaml",
    "lstm": SCRIPT_DIR / "configs" / "trainer_args_lstm.yaml",
    "mlp": SCRIPT_DIR / "configs" / "trainer_args_mlp.yaml",
    "rnn": SCRIPT_DIR / "configs" / "trainer_args_gru.yaml",
    "tcn": SCRIPT_DIR / "configs" / "trainer_args_tcn.yaml",
}
DEFAULT_SNN_CONFIG = SCRIPT_DIR / "configs" / "snn_specific_args.yaml"

TARGET_CORE_PARAMS = 500_000
TARGET_PARAMS = DAILY_PARAMS + TARGET_CORE_PARAMS
PARAM_TOLERANCE = 0.20
MIN_TARGET_PARAMS = int(DAILY_PARAMS + TARGET_CORE_PARAMS * (1.0 - PARAM_TOLERANCE))
MAX_TARGET_PARAMS = int(DAILY_PARAMS + TARGET_CORE_PARAMS * (1.0 + PARAM_TOLERANCE))

MODEL_TYPE_CHOICES = ("snn", "gru", "rnn", "lstm", "mlp", "tcn")
TRAINER_MODEL_TYPE_MAP = {
    "snn": "snn",
    "gru": "gru",
    "rnn": "gru",
    "lstm": "lstm",
    "mlp": "mlp",
    "tcn": "tcn",
}

SHARED_HIDDEN_SIZES = [32, 64, 128, 256] 
SHARED_PATCH_SIZES = [4, 6, 8, 10] 
TCN_KERNEL_SIZES = [3, 5, 7, 9] 
TCN_DILATION_BASES = [1, 2, 3, 4]

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run Optuna hyperparameter optimization for a fixed model family on the "
            "day-wise sentence split iteration 00."
        )
    )
    parser.add_argument(
        "--model_type",
        choices=MODEL_TYPE_CHOICES,
        default="snn",
        help="Model family to optimize. 'rnn' is accepted as a legacy alias for 'gru'.",
    )
    parser.add_argument(
        "--trainer_config",
        default=None,
        help="Optional explicit trainer config YAML. Defaults to the canonical config for --model_type.",
    )
    parser.add_argument(
        "--snn_config",
        default=str(DEFAULT_SNN_CONFIG),
        help="Only used when --model_type=snn.",
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--seed", type=int, default=None, help="Fixed seed for all trials.")
    parser.add_argument(
        "--n_trials",
        type=int,
        default=40,
        help="Number of budget-eligible trials to run. Samples outside the parameter budget do not count.",
    )
    parser.add_argument("--trial_steps", type=int, default=4000)
    parser.add_argument("--val_interval", type=int, default=1000)
    parser.add_argument("--train_log_interval", type=int, default=200)
    parser.add_argument(
        "--pruner_warmup_steps",
        type=int,
        default=10000,
        help="Do not allow Optuna pruning before this many training batches have elapsed.",
    )
    parser.add_argument(
        "--study_name",
        default=None,
        help="Optional study name. Defaults to '<model_type>_optuna_daywise_split00'.",
    )
    parser.add_argument("--output_root", default=str(SCRIPT_DIR / "logs" / "optuna"))
    parser.add_argument(
        "--storage",
        default=None,
        help="Optional Optuna storage URL. Defaults to sqlite in the study directory.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Optional experiment tag. Defaults to 'optuna_<model_type>_daywise_split00'.",
    )
    parser.add_argument(
        "--custom_split_csv",
        default=str(DEFAULT_DAYWISE_SPLIT_CSV),
        help="CSV split definition to use. Defaults to the day-wise sentence split iteration 00 file.",
    )
    parser.add_argument(
        "--sampler_seed",
        type=int,
        default=None,
        help="Optional Optuna sampler seed. Defaults to --seed.",
    )
    return parser.parse_args()


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(path, payload):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def append_jsonl(path, payload):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def clone_config(cfg):
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def resolve_config_path(base_dir, value):
    if value in (None, "", "None", "null"):
        return value
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def normalize_trainer_paths(trainer_cfg):
    if "dataset" in trainer_cfg and "dataset_dir" in trainer_cfg.dataset:
        trainer_cfg.dataset.dataset_dir = resolve_config_path(SCRIPT_DIR, trainer_cfg.dataset.dataset_dir)

    if "dataset" in trainer_cfg and trainer_cfg.dataset.get("custom_split_csv", None):
        trainer_cfg.dataset.custom_split_csv = resolve_config_path(
            SCRIPT_DIR,
            trainer_cfg.dataset.custom_split_csv,
        )

    if trainer_cfg.get("init_from_checkpoint", False) and trainer_cfg.get("init_checkpoint_path", None):
        trainer_cfg.init_checkpoint_path = resolve_config_path(SCRIPT_DIR, trainer_cfg.init_checkpoint_path)


def trainer_model_type(model_type):
    return TRAINER_MODEL_TYPE_MAP[str(model_type)]


def default_study_name(model_type):
    return f"{model_type}_optuna_daywise_split00"


def default_tag(model_type):
    return f"optuna_{model_type}_daywise_split00"


def default_trainer_config_path(model_type):
    trainer_config = DEFAULT_TRAINER_CONFIGS.get(model_type)
    if trainer_config is None:
        raise ValueError(
            f"No default trainer config is registered for model_type='{model_type}'. "
            "Pass --trainer_config explicitly."
        )
    return trainer_config


def resize_block_lag_configs(snn_cfg, num_blocks):
    snn_core = snn_cfg.model.setup.snn_core
    existing = snn_core.get("block_lag_configs", None)
    default_block = {
        "lags": list(snn_core.lagged_lags),
        "num_lags_per_connection": int(snn_core.get("lagged_num_lags_per_connection", 1)),
        "distribution_type": str(snn_core.get("lagged_distribution_type", "uniform")),
        "decay_strength": float(snn_core.get("lagged_decay_strength", 1.0)),
    }

    block_lag_configs = []
    for block_idx in range(int(num_blocks)):
        if existing is not None and block_idx < len(existing):
            block_cfg = OmegaConf.to_container(existing[block_idx], resolve=False)
        else:
            block_cfg = copy.deepcopy(default_block)
        block_lag_configs.append(block_cfg)

    snn_core.block_lag_configs = block_lag_configs


def resolved_patch_stride(trainer_cfg, patch_size):
    configured_stride = int(trainer_cfg.model.get("patch_stride", 4) or 4)
    return min(configured_stride, int(patch_size))


def sample_trial_hparams(trial, model_type):
    lr_max = trial.suggest_float("lr_max", 0.002, 0.05, log=True)
    lr_min_upper = min(0.005, lr_max * 0.999)
    lr_min = trial.suggest_float("lr_min", 5e-5, lr_min_upper, log=True)

    sampled_params = {
        "hidden_size": trial.suggest_categorical("hidden_size", SHARED_HIDDEN_SIZES),
        "dropout": trial.suggest_float("dropout", 0.0, 0.4),
        "patch_size": trial.suggest_categorical("patch_size", SHARED_PATCH_SIZES),
        "weight_decay": trial.suggest_float("weight_decay", 5e-4, 1e-3, log=True),
        "grad_norm_clip": trial.suggest_float("grad_norm_clip", 1.0, 10.0),
        "lr_max": lr_max,
        "lr_min": lr_min,
    }

    if model_type == "snn":
        sampled_params["n_blocks"] = trial.suggest_int("n_blocks", 1, 4)
        sampled_params["tau"] = trial.suggest_float("tau", 1.0, 10.0)
        sampled_params["v_threshold"] = trial.suggest_float("v_threshold", 0.7, 0.95)
    elif model_type == "tcn":
        sampled_params["n_layers"] = trial.suggest_int("n_layers", 1, 4)
        sampled_params["tcn_kernel_size"] = trial.suggest_categorical(
            "tcn_kernel_size",
            TCN_KERNEL_SIZES,
        )
        sampled_params["tcn_dilation_base"] = trial.suggest_categorical(
            "tcn_dilation_base",
            TCN_DILATION_BASES,
        )
    else:

        sampled_params["n_layers"] = trial.suggest_int("n_layers", 1, 4)

    return sampled_params


def apply_trial_hparams(
    trainer_cfg,
    snn_cfg,
    model_type,
    sampled_params,
    trial_steps,
    val_interval,
    train_log_interval,
):
    lr_max = float(sampled_params["lr_max"])
    lr_min = float(sampled_params["lr_min"])
    if lr_min > lr_max:
        raise ValueError(f"Expected lr_min <= lr_max, got lr_min={lr_min} and lr_max={lr_max}")
    weight_decay = float(sampled_params["weight_decay"])
    hidden_size = int(sampled_params["hidden_size"])
    dropout = float(sampled_params["dropout"])
    patch_size = int(sampled_params["patch_size"])
    patch_stride = resolved_patch_stride(trainer_cfg, patch_size)
    grad_norm_clip_value = float(sampled_params["grad_norm_clip"])

    trainer_cfg.num_training_batches = int(trial_steps)
    trainer_cfg.lr_decay_steps = int(trial_steps)
    trainer_cfg.lr_decay_steps_day = int(trial_steps)
    trainer_cfg.batches_per_val_step = int(min(val_interval, trial_steps))
    trainer_cfg.batches_per_train_log = int(min(train_log_interval, trial_steps))
    trainer_cfg.save_best_checkpoint = False
    trainer_cfg.save_all_val_steps = False
    trainer_cfg.save_final_model = False
    trainer_cfg.save_val_metrics = False
    trainer_cfg.save_val_logits = False
    trainer_cfg.save_val_data = False
    trainer_cfg.early_stopping = False

    trainer_cfg.lr_max = lr_max
    trainer_cfg.lr_max_day = lr_max
    trainer_cfg.lr_min = lr_min
    trainer_cfg.lr_min_day = lr_min
    trainer_cfg.weight_decay = weight_decay
    trainer_cfg.grad_norm_clip_value = grad_norm_clip_value

    trainer_cfg.model.hidden_size = hidden_size
    trainer_cfg.model.dropout = dropout
    trainer_cfg.model.patch_size = patch_size
    trainer_cfg.model.patch_stride = patch_stride

    if model_type == "snn":
        num_blocks = int(sampled_params["n_blocks"])
        tau = float(sampled_params["tau"])
        v_threshold = float(sampled_params["v_threshold"])

        snn_cfg.model.setup.hidden_size = hidden_size
        snn_cfg.model.setup.patch_size = patch_size
        snn_cfg.model.setup.patch_stride = patch_stride
        snn_cfg.model.setup.snn_core.n_blocks = num_blocks
        snn_cfg.model.setup.snn_core.p_drop = dropout
        snn_cfg.model.setup.snn_core.v_threshold = v_threshold
        snn_cfg.model.setup.snn_core.tau = tau
        trainer_cfg.model.n_layers = num_blocks
        resize_block_lag_configs(snn_cfg, num_blocks)
    else:
        trainer_cfg.model.n_layers = int(sampled_params["n_layers"])
        if model_type == "tcn":
            trainer_cfg.model.tcn_kernel_size = int(sampled_params["tcn_kernel_size"])
            trainer_cfg.model.tcn_dilation_base = int(sampled_params["tcn_dilation_base"])


def build_trial_record(trial_number, state, sampled_params, trial_dir, record=None, message=None):
    payload = {
        "trial_number": int(trial_number),
        "state": str(state),
        "params": sampled_params,
        "trial_dir": str(trial_dir),
    }
    if message is not None:
        payload["message"] = str(message)
    if record is not None:
        payload.update(record)
    return payload


def is_budget_eligible_trial(trial):
    return bool(trial.user_attrs.get("budget_eligible", False))


def is_budget_rejected_trial(trial):
    return bool(trial.user_attrs.get("budget_rejected", False))


def count_budget_eligible_trials(trials):
    return sum(1 for trial in trials if is_budget_eligible_trial(trial))


def count_budget_rejected_trials(trials):
    return sum(1 for trial in trials if is_budget_rejected_trial(trial))


def main():
    args = parse_args()
    args.model_type = canonicalize_model_type(args.model_type)

    if OmegaConf is None:
        raise SystemExit(
            "OmegaConf is not installed in the active Python environment. "
            "Install the training dependencies and rerun."
        )

    args.trainer_config = args.trainer_config or str(default_trainer_config_path(args.model_type))
    args.study_name = args.study_name or default_study_name(args.model_type)
    args.tag = args.tag or default_tag(args.model_type)
    trainer_runtime_model_type = trainer_model_type(args.model_type)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    try:
        import optuna
        from optuna.pruners import MedianPruner
        from optuna.samplers import TPESampler
        from optuna.trial import TrialState
    except ImportError as exc:
        raise SystemExit(
            "Optuna is not installed in the active Python environment. "
            "Install `optuna` in the same environment used for training and rerun."
        ) from exc

    import torch

    from trainer import BrainToTextDecoder_Trainer
    from utils.seeding import set_global_seed

    trainer_config_path = Path(args.trainer_config).resolve()
    if not trainer_config_path.exists():
        raise FileNotFoundError(f"Trainer config not found: {trainer_config_path}")

    args.custom_split_csv = resolve_config_path(SCRIPT_DIR, args.custom_split_csv)
    custom_split_csv_path = Path(args.custom_split_csv).resolve()
    if not custom_split_csv_path.exists():
        raise FileNotFoundError(f"Custom split CSV not found: {custom_split_csv_path}")

    snn_config_path = None
    if args.model_type == "snn":
        snn_config_path = Path(args.snn_config).resolve()
        if not snn_config_path.exists():
            raise FileNotFoundError(f"SNN config not found: {snn_config_path}")

    base_trainer_cfg = OmegaConf.load(trainer_config_path)
    ensure_model_config_schema(base_trainer_cfg.model)
    base_snn_cfg = OmegaConf.load(snn_config_path) if snn_config_path is not None else None
    if "dataset" in base_trainer_cfg:
        base_trainer_cfg.dataset.custom_split_csv = str(custom_split_csv_path)

    base_seed = int(args.seed if args.seed is not None else base_trainer_cfg.get("seed", 0))
    sampler_seed = int(args.sampler_seed if args.sampler_seed is not None else base_seed)
    set_global_seed(base_seed)

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.9, device=0)

    study_dir = Path(args.output_root).resolve() / args.study_name
    trials_dir = study_dir / "trials"
    ensure_dir(trials_dir)

    storage = args.storage or f"sqlite:///{(study_dir / 'study.db').resolve()}"

    save_json(
        study_dir / "study_setup.json",
        {
            "model_type": args.model_type,
            "trainer_model_type": trainer_runtime_model_type,
            "study_name": args.study_name,
            "trainer_config": str(trainer_config_path),
            "snn_config": None if snn_config_path is None else str(snn_config_path),
            "custom_split_csv": str(custom_split_csv_path),
            "gpu": str(args.gpu),
            "seed": base_seed,
            "sampler_seed": sampler_seed,
            "n_trials": int(args.n_trials),
            "trial_steps": int(args.trial_steps),
            "val_interval": int(args.val_interval),
            "train_log_interval": int(args.train_log_interval),
            "pruner_warmup_steps": int(args.pruner_warmup_steps),
            "target_params": int(TARGET_PARAMS),
            "param_tolerance": float(PARAM_TOLERANCE),
            "param_budget_min": int(MIN_TARGET_PARAMS),
            "param_budget_max": int(MAX_TARGET_PARAMS),
            "storage": storage,
            "tag": args.tag,
        },
    )

    OmegaConf.save(base_trainer_cfg, study_dir / "base_trainer_config_resolved.yaml")
    if base_snn_cfg is not None:
        OmegaConf.save(base_snn_cfg, study_dir / "base_snn_config_resolved.yaml")

    sampler = TPESampler(seed=sampler_seed)
    pruner = MedianPruner(
        n_startup_trials=min(5, max(1, args.n_trials)),
        n_warmup_steps=int(args.pruner_warmup_steps),
        interval_steps=int(args.val_interval),
    )
    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )

    trial_log_path = study_dir / "trial_results.jsonl"
    target_budget_eligible_trials = int(args.n_trials)

    def objective(trial):
        sampled_params = sample_trial_hparams(trial, args.model_type)
        trial_record_context = {
            "model_type": args.model_type,
            "budget_eligible": False,
            "budget_rejected": False,
        }

        set_global_seed(base_seed)

        trainer_cfg = clone_config(base_trainer_cfg)
        snn_cfg = clone_config(base_snn_cfg) if base_snn_cfg is not None else None
        normalize_trainer_paths(trainer_cfg)

        trainer_cfg.mode = "train"
        trainer_cfg.model_type = trainer_runtime_model_type
        trainer_cfg.gpu_number = str(args.gpu)
        trainer_cfg.seed = base_seed
        trainer_cfg.tag = f"{args.tag}_trial{trial.number:04d}"
        if "dataset" in trainer_cfg:
            trainer_cfg.dataset.seed = base_seed

        if snn_cfg is not None:
            snn_cfg.seed = base_seed

        apply_trial_hparams(
            trainer_cfg=trainer_cfg,
            snn_cfg=snn_cfg,
            model_type=args.model_type,
            sampled_params=sampled_params,
            trial_steps=args.trial_steps,
            val_interval=args.val_interval,
            train_log_interval=args.train_log_interval,
        )

        if snn_cfg is not None:
            trainer_cfg.snn_config = snn_cfg
        else:
            trainer_cfg.snn_config = None

        trial_dir = trials_dir / f"trial_{trial.number:04d}"
        checkpoint_dir = trial_dir / "checkpoints"
        log_dir = trial_dir / "logs"
        config_dir = trial_dir / "configs"
        ensure_dir(checkpoint_dir)
        ensure_dir(log_dir)
        ensure_dir(config_dir)

        trainer_cfg.output_dir = str(trial_dir)
        trainer_cfg.checkpoint_dir = str(checkpoint_dir)
        trainer_cfg.log_dir = str(log_dir)

        OmegaConf.save(trainer_cfg, config_dir / "trainer_config_resolved.yaml")
        if snn_cfg is not None:
            OmegaConf.save(snn_cfg, config_dir / "snn_config_resolved.yaml")

        trainer = None
        start_time = time.time()
        try:
            trainer = BrainToTextDecoder_Trainer(trainer_cfg)

            n_params = int(count_params(trainer.model))
            trial_record_context["n_params"] = n_params
            trial_record_context["parameter_count"] = n_params
            trial_record_context["duration_sec"] = float(time.time() - start_time)

            trial.set_user_attr("model_type", args.model_type)
            trial.set_user_attr("n_params", n_params)
            trial.set_user_attr("parameter_count", n_params)
            trial.set_user_attr("trial_dir", str(trial_dir))

            if not (MIN_TARGET_PARAMS <= n_params <= MAX_TARGET_PARAMS):
                trial_record_context["budget_rejected"] = True
                trial.set_user_attr("budget_eligible", False)
                trial.set_user_attr("budget_rejected", True)
                raise optuna.exceptions.TrialPruned(
                    "Trial "
                    f"{trial.number} pruned before training because n_params={n_params} "
                    f"is outside the allowed budget [{MIN_TARGET_PARAMS}, {MAX_TARGET_PARAMS}]"
                )

            trial_record_context["budget_eligible"] = True
            trial.set_user_attr("budget_eligible", True)
            trial.set_user_attr("budget_rejected", False)

            def validation_callback(batch_index, epoch_index, val_metrics, trainer):
                del epoch_index, trainer
                current_per = float(val_metrics["avg_PER"])
                trial.report(current_per, step=int(batch_index))
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned(
                        f"Trial {trial.number} pruned at batch {batch_index} with val_PER={current_per:.6f}"
                    )

            trainer.register_validation_observer(validation_callback)
            trainer.train()

            val_metrics = trainer.validation(
                loader=trainer.val_loader,
                return_logits=False,
                return_data=False,
                collect_diagnostics=True,
            )

            result_record = {
                "model_type": args.model_type,
                "val_PER": float(val_metrics["avg_PER"]),
                "avg_loss": float(val_metrics["avg_loss"]),
                "spike_rate": (
                    None if val_metrics.get("spike_rate") is None else float(val_metrics["spike_rate"])
                ),
                "blank_ratio": float(val_metrics["blank_ratio"]),
                "n_params": n_params,
                "parameter_count": n_params,
                "duration_sec": float(time.time() - start_time),
                "budget_eligible": True,
                "budget_rejected": False,
                "spike_rates_by_layer": val_metrics.get("spike_rates_by_layer", {}),
                "ctc_evidence_path": val_metrics.get("ctc_evidence_path"),
            }

            trial.set_user_attr("val_PER", result_record["val_PER"])
            trial.set_user_attr("avg_loss", result_record["avg_loss"])
            trial.set_user_attr("spike_rate", result_record["spike_rate"])
            trial.set_user_attr("blank_ratio", result_record["blank_ratio"])

            payload = build_trial_record(
                trial_number=trial.number,
                state="COMPLETE",
                sampled_params=sampled_params,
                trial_dir=trial_dir,
                record=result_record,
            )
            save_json(trial_dir / "trial_summary.json", payload)
            append_jsonl(trial_log_path, payload)

            print(
                f"[trial {trial.number:04d}] "
                f"model_type={args.model_type} "
                f"val_PER={result_record['val_PER']:.6f} "
                f"blank_ratio={result_record['blank_ratio']:.6f} "
                f"n_params={result_record['n_params']}"
            )

            return result_record["val_PER"]

        except optuna.exceptions.TrialPruned as exc:
            payload = build_trial_record(
                trial_number=trial.number,
                state="PRUNED",
                sampled_params=sampled_params,
                trial_dir=trial_dir,
                record=trial_record_context,
                message=str(exc),
            )
            save_json(trial_dir / "trial_summary.json", payload)
            append_jsonl(trial_log_path, payload)
            raise
        except Exception as exc:
            trial_record_context["duration_sec"] = float(time.time() - start_time)
            payload = build_trial_record(
                trial_number=trial.number,
                state="FAILED",
                sampled_params=sampled_params,
                trial_dir=trial_dir,
                record=trial_record_context,
                message=str(exc),
            )
            save_json(trial_dir / "trial_summary.json", payload)
            append_jsonl(trial_log_path, payload)
            raise
        finally:
            if trainer is not None:
                trainer.close()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    while count_budget_eligible_trials(study.trials) < target_budget_eligible_trials:
        completed_budget_eligible_trials = count_budget_eligible_trials(study.trials)
        remaining_budget_eligible_trials = target_budget_eligible_trials - completed_budget_eligible_trials
        print(
            "Starting Optuna trial "
            f"for budget-eligible target {completed_budget_eligible_trials + 1}/{target_budget_eligible_trials} "
            f"({remaining_budget_eligible_trials} remaining)."
        )
        study.optimize(objective, n_trials=1, gc_after_trial=True)

    completed_trials = sorted(
        [trial for trial in study.trials if trial.state == TrialState.COMPLETE and trial.value is not None],
        key=lambda trial: float(trial.value),
    )
    if not completed_trials:
        raise RuntimeError(
            "The Optuna study finished without any completed trials. "
            "Check the per-trial summaries under the study directory for failures or pruning."
        )
    top_trials = []
    for trial in completed_trials[:5]:
        top_trials.append(
            {
                "trial_number": int(trial.number),
                "model_type": args.model_type,
                "val_PER": float(trial.value),
                "params": dict(trial.params),
                "user_attrs": dict(trial.user_attrs),
            }
        )

    save_json(study_dir / "top5_configs.json", top_trials)
    save_json(
        study_dir / "best_hyperparameters.json",
        {
            "trial_number": int(study.best_trial.number),
            "model_type": args.model_type,
            "val_PER": float(study.best_value),
            "params": dict(study.best_params),
            "user_attrs": dict(study.best_trial.user_attrs),
        },
    )
    save_json(
        study_dir / "study_summary.json",
        {
            "model_type": args.model_type,
            "best_trial_number": int(study.best_trial.number),
            "best_val_PER": float(study.best_value),
            "best_params": dict(study.best_params),
            "target_params": int(TARGET_PARAMS),
            "param_budget_min": int(MIN_TARGET_PARAMS),
            "param_budget_max": int(MAX_TARGET_PARAMS),
            "requested_budget_eligible_trials": target_budget_eligible_trials,
            "num_trials_total": len(study.trials),
            "num_trials_budget_eligible": count_budget_eligible_trials(study.trials),
            "num_trials_budget_rejected": count_budget_rejected_trials(study.trials),
            "num_trials_complete": len(completed_trials),
            "num_trials_pruned": sum(1 for trial in study.trials if trial.state == TrialState.PRUNED),
            "num_trials_pruned_runtime": sum(
                1
                for trial in study.trials
                if trial.state == TrialState.PRUNED and not is_budget_rejected_trial(trial)
            ),
            "num_trials_failed": sum(1 for trial in study.trials if trial.state == TrialState.FAIL),
            "storage": storage,
            "trial_log_path": str(trial_log_path),
        },
    )

    print("\nBest hyperparameters:")
    print(f"  model_type: {args.model_type}")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"  n_params: {study.best_trial.user_attrs.get('n_params')}")
    print(f"Best validation PER: {float(study.best_value):.6f}")
    print(f"Saved best hyperparameters to: {study_dir / 'best_hyperparameters.json'}")
    print(f"Saved top 5 configs to: {study_dir / 'top5_configs.json'}")


if __name__ == "__main__":
    sys.exit(main())
