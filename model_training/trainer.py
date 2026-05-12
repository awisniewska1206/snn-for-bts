import torch 
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import time
import os
import numpy as np
import math
import pathlib
import logging
import sys
import json
import yaml
from collections import defaultdict
import h5py
from model_training.models.snn_backbone import SpikingModel
import pickle

from helpers.dataset import (
    BrainToTextDataset,
    load_trial_indices_from_split_csv,
    train_test_split_indicies,
)
from helpers.data_augmentations import gauss_smooth
from utils.model_config import canonicalize_model_type, ensure_model_config_schema
from utils.seeding import make_dataloader_generator, seed_worker, set_global_seed

import torchaudio.functional as F # for edit distance
from omegaconf import OmegaConf
from helpers.evaluate_model_helpers import LOGIT_TO_PHONEME, _extract_transcription


torch.set_float32_matmul_precision('high') # makes float32 matmuls faster on some GPUs
torch._dynamo.config.cache_size_limit = 64

from models.gru_model import GRUDecoder
from models.mlp_model import MLPDecoder
from models.lstm_model import LSTMDecoder
from models.tcn_model import TCNDecoder

class BrainToTextDecoder_Trainer:
    """
    This class will initialize and train a brain-to-text phoneme decoder
    
    Written by Nick Card and Zachery Fogg with reference to Stanford NPTL's decoding function
    """

    def __init__(self, args):
        '''
        args : dictionary of training arguments
        '''

        # Trainer fields
        self.args = args
        self.args["model_type"] = canonicalize_model_type(self.args.get("model_type"))
        ensure_model_config_schema(self.args["model"])
        self.logger = None 
        self.device = None
        self.model = None
        self.optimizer = None
        self.learning_rate_scheduler = None
        self.ctc_loss = None 

        self.best_val_PER = torch.inf # track best PER for checkpointing
        self.best_val_loss = torch.inf # track best loss for checkpointing

        self.train_dataset = None 
        self.val_dataset = None 
        self.train_loader = None 
        self.val_loader = None 

        self.transform_args = self.args['dataset']['data_transforms']
        self.spiking_stats_dir = None
        self.spiking_batch_stats_path = None
        self.spiking_epoch_stats_path = None
        self.spiking_summary_path = None
        self.session_threshold_stats_path = None
        self.delay_stats_dir = None
        self.delay_stats_manifest_path = None
        self.lag_selection_dir = None
        self.delay_parameter_csv_path = None
        self.delay_parameter_export_interval = int(self.args.get("delay_parameter_export_interval", 6000))
        self.ctc_evidence_dir = None
        self._validation_run_index = 0
        self._validation_observers = []
        self._epoch_stat_accumulator = defaultdict(lambda: defaultdict(list))
        self._active_epoch_index = None
        self.session_summary_by_id = {}
        session_adapter_cfg = self.args.get("session_adaptation")
        if session_adapter_cfg is None:
            session_adapter_cfg = self.args.get("session_threshold_adapter", {})
        phoneme_transition_bias_cfg = self.args.get("phoneme_transition_bias", {})
        self.adaptation_mode = session_adapter_cfg.get("adaptation_mode", "day")
        self.session_threshold_adapter_enabled = self.adaptation_mode == "session_threshold"
        self.freeze_adapter_batch = session_adapter_cfg.get("freeze_adapter_batch", None)
        self._session_adapter_frozen = False
        self.seed = int(self.args.get("seed", 0))
        set_global_seed(self.seed)

        # Create output directory
        if args['mode'] == 'train':
            os.makedirs(self.args['output_dir'], exist_ok=True)
            self.spiking_stats_dir = os.path.join(self.args['output_dir'], 'spiking_stats')
            os.makedirs(self.spiking_stats_dir, exist_ok=True)
            self.spiking_batch_stats_path = os.path.join(self.spiking_stats_dir, 'per_batch.jsonl')
            self.spiking_epoch_stats_path = os.path.join(self.spiking_stats_dir, 'per_epoch.jsonl')
            self.spiking_summary_path = os.path.join(self.spiking_stats_dir, 'summary.json')
            self.session_threshold_stats_path = os.path.join(
                self.spiking_stats_dir,
                'session_threshold_adapter.jsonl',
            )
            self.delay_stats_dir = os.path.join(self.spiking_stats_dir, 'delay_dynamics')
            os.makedirs(self.delay_stats_dir, exist_ok=True)
            self.delay_stats_manifest_path = os.path.join(self.delay_stats_dir, 'manifest.jsonl')
            self.lag_selection_dir = os.path.join(self.args['output_dir'], 'lag_selections')
            os.makedirs(self.lag_selection_dir, exist_ok=True)
            self.delay_parameter_csv_path = os.path.join(
                self.args['output_dir'],
                'learned_delay_checkpoints.csv',
            )

        # Create checkpoint directory
        if args['save_best_checkpoint'] or args['save_all_val_steps'] or args['save_final_model']: 
            os.makedirs(self.args['checkpoint_dir'], exist_ok=True)

        artifact_root = self.args.get('output_dir') or self.args.get('checkpoint_dir') or '.'
        os.makedirs(artifact_root, exist_ok=True)
        self.ctc_evidence_dir = os.path.join(artifact_root, 'ctc_evidence')
        os.makedirs(self.ctc_evidence_dir, exist_ok=True)

        # Set up logging
        self.logger = logging.getLogger(__name__)
        for handler in self.logger.handlers[:]:  # make a copy of the list
            self.logger.removeHandler(handler)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        formatter = logging.Formatter(fmt='%(asctime)s: %(message)s')        

        if args['mode']=='train':
            # During training, save logs to file in output directory
            fh = logging.FileHandler(str(pathlib.Path(self.args['output_dir'],'training_log')))
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

        # Always print logs to stdout
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        self.logger.addHandler(sh)

        self.exp_tag = str(self.args.get("tag", "")).strip()
        if self.exp_tag:
            self.logger.info(f"EXPERIMENT_TAG: {self.exp_tag}")

        # Configure device pytorch will use 
        if torch.cuda.is_available():
            gpu_num = self.args.get('gpu_number', 0)
            try:
                gpu_num = int(gpu_num)
            except ValueError:
                self.logger.warning(f"Invalid gpu_number value: {gpu_num}. Using 0 instead.")
                gpu_num = 0

            max_gpu_index = torch.cuda.device_count() - 1
            if gpu_num > max_gpu_index:
                self.logger.warning(f"Requested GPU {gpu_num} not available. Using GPU 0 instead.")
                gpu_num = 0

            try:
                self.device = torch.device(f"cuda:{gpu_num}")
                test_tensor = torch.tensor([1.0]).to(self.device)
                test_tensor = test_tensor * 2
            except Exception as e:
                self.logger.error(f"Error initializing CUDA device {gpu_num}: {str(e)}")
                self.logger.info("Falling back to CPU")
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        self.logger.info(f'Using device: {self.device}')

        # Initialize the model
        model_type = self.args["model_type"]
        model_cfg = self.args["model"]
        input_dropout = float(model_cfg["input_network"].get("input_layer_dropout", 0.0))

        if model_type == "gru":
            self.model = GRUDecoder(
                neural_dim=model_cfg["n_input_features"],
                hidden_size=int(model_cfg["hidden_size"]),
                n_days=len(self.args["dataset"]["sessions"]),
                n_classes=self.args["dataset"]["n_classes"],
                dropout=float(model_cfg.get("dropout", 0.0)),
                input_dropout=input_dropout,
                n_layers=model_cfg["n_layers"],
                patch_size=model_cfg["patch_size"],
                patch_stride=model_cfg["patch_stride"],
                phoneme_transition_bias_cfg=phoneme_transition_bias_cfg,
            )
            self.logger.info("Initialized GRU decoding model")

        elif model_type == "mlp":
            self.model = MLPDecoder(
                neural_dim=model_cfg["n_input_features"],
                hidden_size=int(model_cfg["hidden_size"]),
                n_days=len(self.args["dataset"]["sessions"]),
                n_classes=self.args["dataset"]["n_classes"],
                dropout=float(model_cfg.get("dropout", 0.0)),
                input_dropout=input_dropout,
                n_layers=model_cfg["n_layers"],
                patch_size=model_cfg["patch_size"],
                patch_stride=model_cfg["patch_stride"],
                phoneme_transition_bias_cfg=phoneme_transition_bias_cfg,
                lagged_mlp_cfg=model_cfg.get("lagged_mlp", None),
            )
            self.logger.info("Initialized MLP decoding model")
            if getattr(self.model, "lagged_input_mixer", None) is not None:
                self.logger.info(
                    f"[MLP] input lagged mixer lags: {self.model.get_lag_values(0)}"
                )
            lag_selection_paths = self._save_lag_selection_exports()
            if lag_selection_paths:
                self.logger.info(
                    f"Saved lag selection exports for {len(lag_selection_paths)} block(s) to {self.lag_selection_dir}"
                )

        elif model_type == "lstm":
            self.model = LSTMDecoder(
                neural_dim=model_cfg["n_input_features"],
                hidden_size=int(model_cfg["hidden_size"]),
                n_days=len(self.args["dataset"]["sessions"]),
                n_classes=self.args["dataset"]["n_classes"],
                dropout=float(model_cfg.get("dropout", 0.0)),
                input_dropout=input_dropout,
                n_layers=model_cfg["n_layers"],
                patch_size=model_cfg["patch_size"],
                patch_stride=model_cfg["patch_stride"],
                phoneme_transition_bias_cfg=phoneme_transition_bias_cfg,
            )
            self.logger.info("Initialized LSTM decoding model")

        elif model_type == "tcn":
            self.model = TCNDecoder(
                neural_dim=model_cfg["n_input_features"],
                hidden_size=int(model_cfg["hidden_size"]),
                n_days=len(self.args["dataset"]["sessions"]),
                n_classes=self.args["dataset"]["n_classes"],
                dropout=float(model_cfg.get("dropout", 0.0)),
                input_dropout=input_dropout,
                n_layers=model_cfg["n_layers"],
                patch_size=model_cfg["patch_size"],
                patch_stride=model_cfg["patch_stride"],
                phoneme_transition_bias_cfg=phoneme_transition_bias_cfg,
                tcn_kernel_size=model_cfg.get("tcn_kernel_size", 3),
                tcn_dilation_base=model_cfg.get("tcn_dilation_base", 2),
            )
            self.logger.info("Initialized TCN decoding model")

        elif model_type == "lssca_snn":
            self.model = LSSCASNNDecoder(
                neural_dim=model_cfg["n_input_features"],
                n_days=len(self.args["dataset"]["sessions"]),
                n_classes=self.args["dataset"]["n_classes"],
                input_dropout=input_dropout,
                lss_window_size=model_cfg.get("lss_window_size", 100),
                lss_window_stride=model_cfg.get("lss_window_stride", 4),
                suppress_model_stdout=model_cfg.get("suppress_model_stdout", True),
            )
            self.logger.info("Initialized LSS-CA-SNN decoding model")

        elif model_type == "snn":
            if "snn_config" not in self.args or self.args["snn_config"] is None:
                raise ValueError("snn_config missing but model_type='snn'")

            cfg = self.args["snn_config"]

            self.model = SpikingModel(
                cfg=cfg,
                neural_dim=model_cfg["n_input_features"],
                n_days=len(self.args["dataset"]["sessions"]),
                n_classes=self.args["dataset"]["n_classes"],
                input_dropout=input_dropout,
                patch_size=model_cfg["patch_size"],
                patch_stride=model_cfg["patch_stride"],
                debug=False,
                session_adaptation_cfg=session_adapter_cfg,
                phoneme_transition_bias_cfg=phoneme_transition_bias_cfg,
                day_adaptation_mode=self.args.get("day_adaptation_mode", "matrix"),
            )
            self.logger.info("Initialized SNN decoding model")
            lag_selection_paths = self._save_lag_selection_exports()
            if lag_selection_paths:
                self.logger.info(
                    f"Saved lag selection exports for {len(lag_selection_paths)} block(s) to {self.lag_selection_dir}"
                )

        else:
            raise ValueError(f"Unknown model_type: {self.args['model_type']}")

        # Call torch.compile to speed up training
        #self.logger.info("Using torch.compile")
        #self.model = torch.compile(self.model)

        self.logger.info(f"Initialized decoding model")

        self.logger.info(self.model)

        # Log how many parameters are in the model
        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Model has {total_params:,} parameters")

        day_params = 0
        adapter_params = 0
        for name, param in self.model.named_parameters():
            if 'day' in name:
                day_params += param.numel()
            if 'low_rank_adapter' in name:
                adapter_params += param.numel()
        
        self.logger.info(f"Model has {day_params:,} day-specific parameters | {((day_params / total_params) * 100):.2f}% of total parameters")
        self.logger.info(f"Model has {adapter_params:,} low-rank adapter parameters | {((adapter_params / total_params) * 100):.2f}% of total parameters")
        self.logger.info(f"Adaptation mode: {self.adaptation_mode}")

        # Create datasets and dataloaders
        custom_split_csv = self.args["dataset"].get("custom_split_csv", None)
        if custom_split_csv:
            source_filenames = self.args["dataset"].get(
                "custom_source_filenames",
                ["data_train.hdf5", "data_val.hdf5"],
            )
            train_trials, val_trials, test_trials = load_trial_indices_from_split_csv(
                split_csv_path=custom_split_csv,
                dataset_dir=self.args["dataset"]["dataset_dir"],
                sessions=self.args["dataset"]["sessions"],
                source_filenames=source_filenames,
            )
            self.logger.info(
                f"Loaded custom split CSV from {custom_split_csv} using source files {source_filenames}"
            )
        else:
            train_file_paths = [os.path.join(self.args["dataset"]["dataset_dir"],s,'data_train.hdf5') for s in self.args['dataset']['sessions']]
            val_file_paths = [os.path.join(self.args["dataset"]["dataset_dir"],s,'data_val.hdf5') for s in self.args['dataset']['sessions']]

            # Ensure that there are no duplicate days
            if len(set(train_file_paths)) != len(train_file_paths):
                raise ValueError("There are duplicate sessions listed in the train dataset")
            if len(set(val_file_paths)) != len(val_file_paths):
                raise ValueError("There are duplicate sessions listed in the val dataset")

            # Split trials into train and test sets
            train_trials, _ = train_test_split_indicies(
                file_paths = train_file_paths, 
                test_percentage = 0,
                seed = self.seed,
                bad_trials_dict = None,
                )
            _, val_trials = train_test_split_indicies(
                file_paths = val_file_paths, 
                test_percentage = 1,
                seed = self.seed,
                bad_trials_dict = None,
                )
            test_trials = None

        # Save dictionaries to output directory to know which trials were train vs val 
        with open(os.path.join(self.args['output_dir'], 'train_val_trials.json'), 'w') as f: 
            payload = {'train' : train_trials, 'val': val_trials}
            if test_trials is not None:
                payload['test'] = test_trials
            json.dump(payload, f)

        train_counts = {
            self.args['dataset']['sessions'][day_idx]: len(train_trials[day_idx].get('trial_refs', train_trials[day_idx].get('trials', [])))
            for day_idx in train_trials.keys()
        }
        val_counts = {
            self.args['dataset']['sessions'][day_idx]: len(val_trials[day_idx].get('trial_refs', val_trials[day_idx].get('trials', [])))
            for day_idx in val_trials.keys()
        }
        self.logger.info(f"Per-session train counts: {train_counts}")
        self.logger.info(f"Per-session val counts: {val_counts}")

        # Determine if a only a subset of neural features should be used
        feature_subset = None
        if ('feature_subset' in self.args['dataset']) and self.args['dataset']['feature_subset'] != None: 
            feature_subset = self.args['dataset']['feature_subset']
            self.logger.info(f'Using only a subset of features: {feature_subset}')
            
        # train dataset and dataloader
        self.train_dataset = BrainToTextDataset(
            trial_indicies = train_trials,
            split = 'train',
            days_per_batch = self.args['dataset']['days_per_batch'],
            n_batches = self.args['num_training_batches'],
            batch_size = self.args['dataset']['batch_size'],
            must_include_days = None,
            random_seed = self.seed,
            feature_subset = feature_subset
            )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size = None, # Dataset.__getitem__() already returns batches
            shuffle = self.args['dataset']['loader_shuffle'],
            num_workers = self.args['dataset']['num_dataloader_workers'],
            pin_memory = True,
            worker_init_fn = seed_worker,
            generator = make_dataloader_generator(self.seed),
        )

        # val dataset and dataloader
        self.val_dataset = BrainToTextDataset(
            trial_indicies = val_trials, 
            split = 'test',
            days_per_batch = None,
            n_batches = None,
            batch_size = self.args['dataset']['batch_size'],
            must_include_days = None,
            random_seed = self.seed,
            feature_subset = feature_subset   
            )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size = None, # Dataset.__getitem__() already returns batches
            shuffle = False, 
            num_workers = 0,
            pin_memory = True,
            worker_init_fn = seed_worker,
            generator = make_dataloader_generator(self.seed),
        )

        self.logger.info("Successfully initialized datasets")
        self._initialize_session_summaries(train_trials)

        # Create optimizer, learning rate scheduler, and loss
        self.optimizer = self.create_optimizer()

        if self.args['lr_scheduler_type'] == 'linear':
            self.learning_rate_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer = self.optimizer,
                start_factor = 1.0,
                end_factor = self.args['lr_min'] / self.args['lr_max'],
                total_iters = self.args['lr_decay_steps'],
            )
        elif self.args['lr_scheduler_type'] == 'cosine':
            self.learning_rate_scheduler = self.create_cosine_lr_scheduler(self.optimizer)
        
        else:
            raise ValueError(f"Invalid learning rate scheduler type: {self.args['lr_scheduler_type']}")
        
        self.ctc_loss = torch.nn.CTCLoss(blank = 0, reduction = 'none', zero_infinity = False)

        # If a checkpoint is provided, then load from checkpoint
        if self.args['init_from_checkpoint']:
            self.load_model_checkpoint(self.args['init_checkpoint_path'])

        # Freeze the shared model trunk and/or day-specific input path when requested.
        model_trainable = bool(model_cfg.get("trainable", True))
        for name, param in self.model.named_parameters():
            if not model_trainable and (
                'gru' in name
                or 'mlp' in name
                or 'lstm' in name
                or 'tcn' in name
                or (model_type == "snn" and 'day' not in name)
                or (model_type == "lssca_snn" and name.startswith("backbone."))
            ):
                param.requires_grad = False

            elif not self.args['model']['input_network']['input_trainable'] and 'day' in name:
                param.requires_grad = False

        # Send model to device 
        self.model.to(self.device)
        try:
            self.train_batches_per_epoch = len(self.train_loader)
        except TypeError:
            self.train_batches_per_epoch = int(self.args['num_training_batches'])

    def close(self):
        if self.logger is not None:
            handlers = self.logger.handlers[:]
            for h in handlers:
                try:
                    h.flush()
                except Exception:
                    pass
                try:
                    h.close()
                except Exception:
                    pass
                self.logger.removeHandler(h)

        logging.shutdown()

    def register_validation_observer(self, callback):
        if callback is None:
            return
        self._validation_observers.append(callback)

    def _notify_validation_observers(self, batch_index, epoch_index, val_metrics):
        for callback in self._validation_observers:
            callback(
                batch_index=int(batch_index),
                epoch_index=int(epoch_index),
                val_metrics=val_metrics,
                trainer=self,
            )

    def _to_serializable(self, value):
        if isinstance(value, dict):
            return {str(k): self._to_serializable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_serializable(v) for v in value]
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.detach().cpu().item()
            return value.detach().cpu().tolist()
        return value

    def _forward_model(self, features, day_indicies, session_summaries=None):
        if self.args.get("model_type") == "snn":
            return self.model(features, day_indicies, session_summary=session_summaries)
        return self.model(features, day_indicies)

    def _record_epoch_layer_stats(self, layer_name, layer_stats):
        for metric_name, metric_value in layer_stats.items():
            if metric_value is None:
                continue
            if isinstance(metric_value, (int, float, np.integer, np.floating)):
                self._epoch_stat_accumulator[layer_name][metric_name].append(float(metric_value))

    def _flush_epoch_spiking_stats(self, epoch_index):
        if epoch_index is None or not self.spiking_epoch_stats_path:
            return
        if not self._epoch_stat_accumulator:
            return

        epoch_layers = {}
        for layer_name, metrics in self._epoch_stat_accumulator.items():
            epoch_layers[layer_name] = {}
            for metric_name, values in metrics.items():
                if not values:
                    continue
                epoch_layers[layer_name][metric_name] = float(np.mean(values))

        epoch_record = {
            "epoch_index": int(epoch_index),
            "num_batches": int(
                max((len(metric_values) for metrics in self._epoch_stat_accumulator.values() for metric_values in metrics.values()), default=0)
            ),
            "layers": epoch_layers,
        }

        global_neuron_params = self._log_global_neuron_parameters(
            prefix=f"Epoch {int(epoch_index)}",
        )
        if global_neuron_params:
            epoch_record["global_neuron_parameters"] = global_neuron_params

        with open(self.spiking_epoch_stats_path, "a") as f:
            f.write(json.dumps(self._to_serializable(epoch_record)) + "\n")

        with open(self.spiking_summary_path, "w") as f:
            json.dump(self._to_serializable(epoch_record), f, indent=2)

        self._epoch_stat_accumulator = defaultdict(lambda: defaultdict(list))

    def _log_global_neuron_parameters(self, prefix):
        if self.args.get("model_type") != "snn":
            return {}
        if not hasattr(self.model, "get_global_neuron_parameters"):
            return {}

        global_neuron_params = self.model.get_global_neuron_parameters() or {}
        if not global_neuron_params:
            return {}

        log_line = (
            f"{prefix} global neuron threshold="
            f"{float(global_neuron_params['threshold']):.6f}"
        )
        if "tau" in global_neuron_params:
            log_line += f" tau={float(global_neuron_params['tau']):.6f}"
        self.logger.info(log_line)
        return global_neuron_params

    def _save_spiking_batch_stats(self, batch_index, epoch_index):
        if self.args.get("model_type") != "snn":
            return
        if not hasattr(self.model, "get_spiking_layer_statistics"):
            return

        layer_stats = self.model.get_spiking_layer_statistics()
        if not layer_stats:
            return

        grad_stats = {}
        if hasattr(self.model, "get_spiking_gradient_statistics"):
            grad_stats = self.model.get_spiking_gradient_statistics()

        batch_record = {
            "batch_index": int(batch_index),
            "epoch_index": int(epoch_index),
            "layers": {},
        }

        for layer_name, stats in layer_stats.items():
            layer_record = dict(stats)
            layer_record["gradient_norm"] = grad_stats.get(layer_name)
            batch_record["layers"][layer_name] = layer_record
            self._record_epoch_layer_stats(layer_name, layer_record)

        with open(self.spiking_batch_stats_path, "a") as f:
            f.write(json.dumps(self._to_serializable(batch_record)) + "\n")

    def _initialize_session_summaries(self, trial_indicies):
        if not self.session_threshold_adapter_enabled:
            return

        feature_subset = self.args["dataset"].get("feature_subset")
        summary_by_id = {}

        for session_id, session_info in trial_indicies.items():
            total_sum = None
            total_sq_sum = None
            total_count = 0

            with h5py.File(session_info["session_path"], "r") as handle:
                for trial_idx in session_info["trials"]:
                    trial_group = handle[f"trial_{trial_idx:04d}"]
                    input_features = trial_group["input_features"][:]
                    if feature_subset is not None:
                        input_features = input_features[:, feature_subset]

                    input_features = input_features.astype(np.float64, copy=False)
                    if total_sum is None:
                        n_channels = int(input_features.shape[1])
                        total_sum = np.zeros(n_channels, dtype=np.float64)
                        total_sq_sum = np.zeros(n_channels, dtype=np.float64)

                    total_sum += input_features.sum(axis=0)
                    total_sq_sum += np.square(input_features).sum(axis=0)
                    total_count += int(input_features.shape[0])

            if total_sum is None or total_count <= 0:
                raise ValueError(f"No usable input_features found for session_id={session_id}")

            mean = total_sum / total_count
            var = np.maximum((total_sq_sum / total_count) - np.square(mean), 0.0)
            std = np.sqrt(var)
            summary = np.concatenate([mean, std], axis=0).astype(np.float32, copy=False)
            summary_by_id[int(session_id)] = torch.from_numpy(summary)

        self.session_summary_by_id = summary_by_id
        summary_dim = next(iter(summary_by_id.values())).numel() if summary_by_id else 0
        self.logger.info(
            f"Initialized {len(self.session_summary_by_id)} fixed session summaries for threshold adaptation "
            f"(summary_dim={summary_dim})"
        )

    def _lookup_session_summaries(self, session_ids):
        if not self.session_threshold_adapter_enabled:
            return None

        summaries = []
        missing = []
        for session_id in session_ids.detach().cpu().tolist():
            summary = self.session_summary_by_id.get(int(session_id))
            if summary is None:
                missing.append(int(session_id))
                continue
            summaries.append(summary)

        if missing:
            raise KeyError(f"Missing session summaries for session_ids={sorted(set(missing))}")

        return torch.stack(summaries, dim=0).to(self.device)

    def _save_session_threshold_batch_stats(self, epoch_index, batch_index):
        if not self.session_threshold_adapter_enabled:
            return
        if batch_index is None:
            return
        if not hasattr(self.model, "compute_session_adaptation"):
            return
        if not self.session_summary_by_id:
            return

        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            for session_id in sorted(self.session_summary_by_id.keys()):
                session_summary = self.session_summary_by_id[session_id].unsqueeze(0).to(self.device)
                delta_v_th, delta_gain = self.model.compute_session_adaptation(session_summary)
                if delta_v_th is None or delta_gain is None:
                    continue
                delta_v_th = delta_v_th.detach().float()
                delta_gain = delta_gain.detach().float()
                record = {
                    "session_id": int(session_id),
                    "epoch": None if epoch_index is None else int(epoch_index),
                    "batch": int(batch_index),
                    "delta_v_th_mean": float(delta_v_th.mean().item()),
                    "delta_v_th_std": float(delta_v_th.std(unbiased=False).item()),
                    "delta_gain_mean": float(delta_gain.mean().item()),
                    "delta_gain_std": float(delta_gain.std(unbiased=False).item()),
                }
                if self.session_threshold_stats_path is not None:
                    with open(self.session_threshold_stats_path, "a") as f:
                        f.write(json.dumps(record) + "\n")
        if was_training:
            self.model.train()

    def _maybe_freeze_session_adapter(self, current_batch):
        if not self.session_threshold_adapter_enabled:
            return
        if self._session_adapter_frozen:
            return
        if self.freeze_adapter_batch is None:
            return
        if current_batch < int(self.freeze_adapter_batch):
            return
        if not hasattr(self.model, "freeze_session_adapter"):
            return
        did_freeze = self.model.freeze_session_adapter()
        if not did_freeze:
            return
        self._session_adapter_frozen = True
        self.logger.info(
            f"Freezing SessionThresholdAdapter at batch {int(current_batch)} "
            f"(freeze_adapter_batch={int(self.freeze_adapter_batch)})"
        )

    def _verify_optimizer_parameter_consistency(self):
        if self.optimizer is None:
            return

        model_params = {id(param): name for name, param in self.model.named_parameters()}
        stale_refs = []
        for group_idx, group in enumerate(self.optimizer.param_groups):
            for param_idx, param in enumerate(group["params"]):
                if id(param) not in model_params:
                    stale_refs.append(
                        {
                            "group_idx": int(group_idx),
                            "param_idx": int(param_idx),
                        }
                    )

        if stale_refs:
            raise RuntimeError(
                "Optimizer param_groups contain parameter references that are not present in the current model. "
                f"Found {len(stale_refs)} stale reference(s), sample={stale_refs[:5]}"
            )

    def _save_delay_statistics(self, batch_index, epoch_index, event_name, store_full=True):
        # Lag snapshot export is disabled because rebuilding/storing full delay
        # tensors can dominate memory and disk usage for large lag mixers.
        return []

        if self.args.get("model_type") != "snn":
            return []
        if not self.delay_stats_dir:
            return []

        backbone = getattr(self.model, "backbone", None)
        if backbone is None or not hasattr(backbone, "save_delay_snapshots"):
            return []

        event_dir = os.path.join(self.delay_stats_dir, event_name)
        saved_paths = backbone.save_delay_snapshots(
            event_dir,
            epoch=epoch_index,
            step=batch_index,
            store_full=store_full,
        )

        manifest_record = {
            "event_name": event_name,
            "batch_index": int(batch_index),
            "epoch_index": int(epoch_index),
            "store_full": bool(store_full),
            "snapshots": saved_paths,
        }
        with open(self.delay_stats_manifest_path, "a") as f:
            f.write(json.dumps(self._to_serializable(manifest_record)) + "\n")

        return saved_paths

    def _save_lag_selection_exports(self):
        if not self.lag_selection_dir:
            return []
        if not hasattr(self.model, "save_lag_selection_snapshots"):
            return []

        saved_paths = self.model.save_lag_selection_snapshots(self.lag_selection_dir)
        if saved_paths:
            manifest_path = os.path.join(self.lag_selection_dir, "manifest.json")
            with open(manifest_path, "w") as f:
                json.dump(self._to_serializable(saved_paths), f, indent=2)

        return saved_paths

    def _save_delay_parameter_checkpoint(self, batch_index, epoch_index, event_name):
        if self.args.get("model_type") != "snn":
            return None
        if not self.delay_parameter_csv_path:
            return None

        backbone = getattr(self.model, "backbone", None)
        if backbone is None or not hasattr(backbone, "append_delay_parameter_checkpoint_csv"):
            return None

        export_result = backbone.append_delay_parameter_checkpoint_csv(
            self.delay_parameter_csv_path,
            epoch=epoch_index,
            step=batch_index,
            event_name=event_name,
            discrete=False,
        )
        if not export_result or int(export_result.get("rows_written", 0)) == 0:
            return export_result

        self.logger.info(
            f"Saved learned delay parameters to {export_result['path']} "
            f"(event={event_name}, batch={int(batch_index)}, rows={int(export_result['rows_written'])})"
        )

        for summary in export_result.get("summaries", []):
            sample_values = summary.get("sample_values", [])
            sample_text = ", ".join(f"{float(value):.4f}" for value in sample_values[:5])
            self.logger.info(
                "[delay params] "
                f"event={summary.get('event_name')} "
                f"step={int(summary.get('step')) if summary.get('step') is not None else 'NA'} "
                f"block={int(summary.get('block_idx')) if summary.get('block_idx') is not None else 'NA'} "
                f"param={summary.get('parameter')} "
                f"shape={summary.get('shape')} "
                f"min={float(summary.get('min')):.6f} "
                f"max={float(summary.get('max')):.6f} "
                f"mean={float(summary.get('mean')):.6f} "
                f"std={float(summary.get('std')):.6f} "
                f"sample=[{sample_text}]"
            )

        return export_result

    def _scalar_to_int(self, value):
        if isinstance(value, torch.Tensor):
            return int(value.detach().cpu().item())
        return int(value)

    def _build_trial_id(self, day_idx, block_num, trial_num):
        return (
            f"day_{self._scalar_to_int(day_idx):02d}_"
            f"block_{self._scalar_to_int(block_num):04d}_"
            f"trial_{self._scalar_to_int(trial_num):04d}"
        )

    def _ctc_extended_target(self, target, blank_id=0):
        target = target.to(torch.long)
        if target.numel() == 0:
            return torch.tensor([blank_id], dtype=torch.long, device=target.device)

        extended = torch.full(
            (target.numel() * 2 + 1,),
            fill_value=blank_id,
            dtype=torch.long,
            device=target.device,
        )
        extended[1::2] = target
        return extended

    def _forced_align_ctc_viterbi(self, log_probs, target, blank_id=0):
        T = int(log_probs.shape[0])
        if T == 0:
            return torch.empty(0, dtype=torch.int32)

        target = target.to(device=log_probs.device, dtype=torch.long)
        if target.numel() == 0:
            return torch.full((T,), blank_id, dtype=torch.int32)

        ext = self._ctc_extended_target(target, blank_id=blank_id)
        S = int(ext.shape[0])
        neg_inf = torch.tensor(-1e30, device=log_probs.device, dtype=log_probs.dtype)

        dp = torch.full((T, S), neg_inf, dtype=log_probs.dtype, device=log_probs.device)
        backptr = torch.full((T, S), -1, dtype=torch.long, device=log_probs.device)

        dp[0, 0] = log_probs[0, ext[0]]
        if S > 1:
            dp[0, 1] = log_probs[0, ext[1]]

        for t in range(1, T):
            for s in range(S):
                best_score = dp[t - 1, s]
                best_prev = s

                if s - 1 >= 0 and dp[t - 1, s - 1] > best_score:
                    best_score = dp[t - 1, s - 1]
                    best_prev = s - 1

                if s - 2 >= 0 and ext[s] != blank_id and ext[s] != ext[s - 2]:
                    if dp[t - 1, s - 2] > best_score:
                        best_score = dp[t - 1, s - 2]
                        best_prev = s - 2

                dp[t, s] = best_score + log_probs[t, ext[s]]
                backptr[t, s] = best_prev

        if S == 1:
            end_state = 0
        else:
            end_state = S - 1 if dp[T - 1, S - 1] >= dp[T - 1, S - 2] else S - 2

        state_path = torch.empty(T, dtype=torch.long, device=log_probs.device)
        state_path[-1] = end_state
        for t in range(T - 1, 0, -1):
            state_path[t - 1] = backptr[t, state_path[t]]

        return ext[state_path].detach().cpu().to(torch.int32)

    def _boundary_and_interior_masks(self, aligned_target, boundary_radius=2, blank_id=0):
        T = int(aligned_target.numel())
        boundary = torch.zeros(T, dtype=torch.bool)

        for t in range(1, T):
            if int(aligned_target[t]) == int(aligned_target[t - 1]):
                continue
            start = max(0, t - boundary_radius)
            end = min(T, t + boundary_radius + 1)
            boundary[start:end] = True

        interior = (~boundary) & (aligned_target != blank_id)
        return boundary, interior

    def _build_validation_trial_export_record(
        self,
        logits,
        adjusted_len,
        day_idx,
        block_num,
        trial_num,
        labels,
        phone_seq_len,
        session_id=None,
        include_alignment=False,
        boundary_radius=2,
        blank_id=0,
    ):
        valid_len = self._scalar_to_int(adjusted_len)
        valid_logits = logits[:valid_len].detach().float().cpu()
        probs = torch.softmax(valid_logits, dim=-1)
        predictions = probs.argmax(dim=-1).to(torch.int32)

        target_len = self._scalar_to_int(phone_seq_len)
        target_sequence = labels[:target_len].detach().cpu().to(torch.int32)

        record = {
            "trial_id": self._build_trial_id(day_idx, block_num, trial_num),
            "input_length": valid_len,
            "target_sequence": target_sequence,
            "target_length": target_len,
            "logits": valid_logits.to(torch.float32),
            "probs": probs.to(torch.float32),
            "predictions": predictions,
        }

        if session_id is not None:
            record["session_id"] = self._scalar_to_int(session_id)

        if include_alignment:
            log_probs = torch.log_softmax(valid_logits, dim=-1)
            aligned_target = self._forced_align_ctc_viterbi(
                log_probs=log_probs,
                target=target_sequence,
                blank_id=blank_id,
            )
            boundary_mask, interior_mask = self._boundary_and_interior_masks(
                aligned_target=aligned_target,
                boundary_radius=boundary_radius,
                blank_id=blank_id,
            )
            record["aligned_target"] = aligned_target
            record["boundary_mask"] = boundary_mask
            record["interior_mask"] = interior_mask

        return record

    def _compute_ctc_evidence_record(
        self,
        logits,
        adjusted_len,
        day_idx,
        block_num,
        trial_num,
        labels,
        phone_seq_len,
        transcription,
        decoded_seq,
    ):
        valid_len = self._scalar_to_int(adjusted_len)
        valid_logits = logits[:valid_len].detach().float()
        probs = torch.softmax(valid_logits, dim=-1)
        log_probs = torch.log_softmax(valid_logits, dim=-1)

        blank_probability = probs[:, 0]
        top_non_blank_probability, top_non_blank_class = probs[:, 1:].max(dim=-1)
        top_non_blank_class = top_non_blank_class + 1
        argmax_class = probs.argmax(dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)

        target_seq = None
        if labels is not None and phone_seq_len is not None:
            target_seq = labels[: self._scalar_to_int(phone_seq_len)].detach().cpu().to(torch.int32).numpy()

        decoded_seq_np = None
        if decoded_seq is not None:
            decoded_seq_np = np.asarray(decoded_seq, dtype=np.int32)

        record = {
            'trial_id': self._build_trial_id(day_idx, block_num, trial_num),
            'day_index': self._scalar_to_int(day_idx),
            'block_num': self._scalar_to_int(block_num),
            'trial_num': self._scalar_to_int(trial_num),
            'output_length': valid_len,
            'blank_probability': blank_probability.cpu().numpy().astype(np.float32),
            'top_non_blank_probability': top_non_blank_probability.cpu().numpy().astype(np.float32),
            'top_non_blank_class': top_non_blank_class.cpu().numpy().astype(np.int32),
            'argmax_class': argmax_class.cpu().numpy().astype(np.int32),
            'entropy': entropy.cpu().numpy().astype(np.float32),
            'target_sequence': target_seq,
            'decoded_sequence': decoded_seq_np,
            'transcription': _extract_transcription(transcription.detach().cpu().numpy()) if transcription is not None else None,
        }

        return record

    def _save_ctc_evidence_records(self, records, split_name='validation'):
        if not records or not self.ctc_evidence_dir:
            return None

        save_path = os.path.join(
            self.ctc_evidence_dir,
            f'{split_name}_run_{self._validation_run_index:04d}.pt',
        )

        payload = {
            'split': split_name,
            'run_index': int(self._validation_run_index),
            'phoneme_classes': list(LOGIT_TO_PHONEME),
            'trials': records,
        }
        torch.save(payload, save_path)
        self._validation_run_index += 1
        return save_path

    def _load_analysis_artifact(self, path):
        suffix = os.path.splitext(path)[1].lower()
        if suffix == ".pt":
            return torch.load(path, map_location="cpu", weights_only=False)
        if suffix == ".pkl":
            with open(path, "rb") as f:
                return pickle.load(f)
        if suffix == ".npz":
            npz = np.load(path, allow_pickle=True)
            return {key: npz[key] for key in npz.files}
        raise ValueError(f"Unsupported artifact type: {suffix}")

    def _compute_ctc_timestep_metrics(self, logits, adjusted_len, include_logits=False):
        valid_len = self._scalar_to_int(adjusted_len)
        valid_logits = logits[:valid_len].detach().float()
        probs = torch.softmax(valid_logits, dim=-1)
        log_probs = torch.log_softmax(valid_logits, dim=-1)

        top_non_blank_probability, top_non_blank_class = probs[:, 1:].max(dim=-1)
        metrics = {
            "output_length": valid_len,
            "blank_probability": probs[:, 0].cpu().numpy().astype(np.float32),
            "top_non_blank_probability": top_non_blank_probability.cpu().numpy().astype(np.float32),
            "top_non_blank_class": (top_non_blank_class + 1).cpu().numpy().astype(np.int32),
            "argmax_class": probs.argmax(dim=-1).cpu().numpy().astype(np.int32),
            "entropy": (-(probs * log_probs).sum(dim=-1)).cpu().numpy().astype(np.float32),
        }
        if include_logits:
            metrics["logits"] = valid_logits.cpu().numpy().astype(np.float32)
        return metrics

    def _local_maxima(self, values, threshold=0.0, min_distance=1):
        values = np.asarray(values, dtype=np.float32)
        if len(values) < 3:
            return np.array([], dtype=np.int32)

        candidates = []
        for idx in range(1, len(values) - 1):
            if values[idx] < threshold:
                continue
            if values[idx] >= values[idx - 1] and values[idx] >= values[idx + 1]:
                candidates.append(idx)

        if not candidates:
            return np.array([], dtype=np.int32)

        candidates = sorted(candidates, key=lambda idx: values[idx], reverse=True)
        selected = []
        for idx in candidates:
            if all(abs(idx - kept_idx) >= min_distance for kept_idx in selected):
                selected.append(idx)

        return np.array(sorted(selected), dtype=np.int32)

    def _detect_peak_indices(
        self,
        trial_record,
        top_peak_min_prob=0.35,
        blank_probability_max=None,
        min_peak_distance=3,
        use_saved_peak_indices=True,
    ):
        if use_saved_peak_indices and ("peak_indices" in trial_record) and trial_record["peak_indices"] is not None:
            return np.asarray(trial_record["peak_indices"], dtype=np.int32)

        top_non_blank_probability = np.asarray(trial_record["top_non_blank_probability"], dtype=np.float32)
        peak_indices = self._local_maxima(
            top_non_blank_probability,
            threshold=top_peak_min_prob,
            min_distance=min_peak_distance,
        )

        if blank_probability_max is not None and len(peak_indices) > 0:
            blank_probability = np.asarray(trial_record["blank_probability"], dtype=np.float32)
            keep_mask = blank_probability[peak_indices] <= float(blank_probability_max)
            peak_indices = peak_indices[keep_mask]

        return peak_indices.astype(np.int32)

    def _prepare_peak_ablation_inputs(self, evidence):
        if not isinstance(evidence, dict):
            raise ValueError("Evidence artifact must be a dict-like object")

        if "trials" in evidence and evidence["trials"]:
            selected_trial_records = []
            for trial_record in evidence["trials"]:
                if "trial_id" not in trial_record:
                    continue
                selected_trial_records.append(dict(trial_record))
            return selected_trial_records, "trials"

        if "peaks" in evidence and evidence["peaks"]:
            grouped_trials = {}
            for peak_record in evidence["peaks"]:
                trial_id = peak_record.get("trial_id")
                if trial_id is None:
                    continue

                trial_entry = grouped_trials.setdefault(
                    trial_id,
                    {
                        "trial_id": trial_id,
                        "peak_indices": [],
                    },
                )

                peak_timestep = peak_record.get("peak_timestep")
                if peak_timestep is not None:
                    trial_entry["peak_indices"].append(int(peak_timestep))

                for key in (
                    "blank_probability",
                    "top_non_blank_probability",
                    "argmax_class",
                    "entropy",
                    "target_sequence",
                    "decoded_sequence",
                    "transcription",
                    "day_index",
                    "block_num",
                    "trial_num",
                ):
                    if key in peak_record and key not in trial_entry:
                        trial_entry[key] = peak_record[key]

                baseline_summary = peak_record.get("baseline_metrics_at_peak", {})
                if ("argmax_class" in baseline_summary) and ("argmax_class" not in trial_entry):
                    trial_entry["argmax_class"] = np.asarray([baseline_summary["argmax_class"]], dtype=np.int32)

            selected_trial_records = list(grouped_trials.values())
            return selected_trial_records, "peaks"

        raise ValueError("Evidence artifact must contain non-empty 'trials' or 'peaks'")

    def _extract_window_metrics(self, metrics, center_timestep, window_radius):
        center_timestep = int(center_timestep)
        output_length = int(metrics["output_length"])

        relative_offsets = []
        blank_probability = []
        top_non_blank_probability = []
        entropy = []
        argmax_class = []
        top_non_blank_class = []
        logits = []

        for rel_offset in range(-window_radius, window_radius + 1):
            timestep = center_timestep + rel_offset
            if timestep < 0 or timestep >= output_length:
                continue

            relative_offsets.append(rel_offset)
            blank_probability.append(float(metrics["blank_probability"][timestep]))
            top_non_blank_probability.append(float(metrics["top_non_blank_probability"][timestep]))
            entropy.append(float(metrics["entropy"][timestep]))
            argmax_class.append(int(metrics["argmax_class"][timestep]))
            top_non_blank_class.append(int(metrics["top_non_blank_class"][timestep]))
            if "logits" in metrics:
                logits.append(np.asarray(metrics["logits"][timestep], dtype=np.float32))

        window = {
            "relative_offsets": np.asarray(relative_offsets, dtype=np.int32),
            "blank_probability": np.asarray(blank_probability, dtype=np.float32),
            "top_non_blank_probability": np.asarray(top_non_blank_probability, dtype=np.float32),
            "entropy": np.asarray(entropy, dtype=np.float32),
            "argmax_class": np.asarray(argmax_class, dtype=np.int32),
            "top_non_blank_class": np.asarray(top_non_blank_class, dtype=np.int32),
        }
        if logits:
            window["logits"] = np.stack(logits).astype(np.float32)
        return window

    def _summarize_window_importance(self, baseline_window, ablated_window):
        if not np.array_equal(baseline_window["relative_offsets"], ablated_window["relative_offsets"]):
            raise ValueError("Baseline and ablated windows must use the same relative offsets")

        delta_blank_probability = ablated_window["blank_probability"] - baseline_window["blank_probability"]
        delta_top_non_blank_probability = (
            ablated_window["top_non_blank_probability"] - baseline_window["top_non_blank_probability"]
        )
        delta_entropy = ablated_window["entropy"] - baseline_window["entropy"]
        logit_delta = ablated_window["logits"] - baseline_window["logits"]

        return {
            "relative_offsets": baseline_window["relative_offsets"].copy(),
            "delta_blank_probability": delta_blank_probability.astype(np.float32),
            "delta_top_non_blank_probability": delta_top_non_blank_probability.astype(np.float32),
            "delta_entropy": delta_entropy.astype(np.float32),
            "logit_l1_diff": np.abs(logit_delta).sum(axis=-1).astype(np.float32),
            "logit_l2_diff": np.sqrt(np.square(logit_delta).sum(axis=-1)).astype(np.float32),
        }

    def _build_peak_baseline_summary(self, metrics, peak_timestep, predicted_class):
        peak_timestep = int(peak_timestep)
        baseline_summary = {
            "blank_probability": float(metrics["blank_probability"][peak_timestep]),
            "top_non_blank_probability": float(metrics["top_non_blank_probability"][peak_timestep]),
            "entropy": float(metrics["entropy"][peak_timestep]),
            "argmax_class": int(predicted_class),
            "top_non_blank_class": int(metrics["top_non_blank_class"][peak_timestep]),
        }
        if 0 <= int(predicted_class) < len(LOGIT_TO_PHONEME):
            baseline_summary["predicted_phoneme"] = LOGIT_TO_PHONEME[int(predicted_class)]
        return baseline_summary

    def _phoneme_to_broad_category(self, phoneme_label):
        if phoneme_label is None:
            return None

        vowels = {
            "AA", "AE", "AH", "AO", "AW", "AY",
            "EH", "ER", "EY", "IH", "IY", "OW",
            "OY", "UH", "UW",
        }
        return "vowel" if str(phoneme_label).upper() in vowels else "consonant"

    def _recover_previous_phoneme_from_sequence(self, framewise_class_sequence, peak_timestep):
        # Approximate the previous decoded symbol by CTC-style collapsing of the
        # baseline framewise class trace up to the current peak timestep.
        if framewise_class_sequence is None:
            return None

        class_sequence = np.asarray(framewise_class_sequence, dtype=np.int32).reshape(-1)
        if len(class_sequence) == 0:
            return None

        peak_timestep = int(peak_timestep)
        peak_timestep = max(0, min(peak_timestep, len(class_sequence) - 1))
        collapsed_labels = []
        previous_class = None

        for class_idx in class_sequence[: peak_timestep + 1]:
            class_idx = int(class_idx)
            if class_idx == previous_class:
                continue
            previous_class = class_idx
            if class_idx == 0:
                continue
            collapsed_labels.append(class_idx)

        if len(collapsed_labels) < 2:
            return None

        previous_class = int(collapsed_labels[-2])
        if previous_class < 0 or previous_class >= len(LOGIT_TO_PHONEME):
            return {
                "previous_class": previous_class,
                "previous_phoneme": None,
                "previous_category": None,
                "source": "heuristic_ctc_collapse",
            }

        previous_phoneme = LOGIT_TO_PHONEME[previous_class]
        return {
            "previous_class": previous_class,
            "previous_phoneme": previous_phoneme,
            "previous_category": self._phoneme_to_broad_category(previous_phoneme),
            "source": "heuristic_ctc_collapse",
        }

    def run_lag_peak_ablation(
        self,
        evidence_path,
        output_path=None,
        block_idx=0,
        window_radius=2,
        top_peak_min_prob=0.35,
        blank_probability_max=None,
        min_peak_distance=3,
        use_saved_peak_indices=True,
        max_trials=None,
        debug_single=False,
        debug_lag=0,
        debug_peak_idx=0,
        debug_stop_after_peak_detection=False,
    ):
        if self.args.get("model_type") != "snn":
            raise ValueError("Lag peak ablation is only supported for model_type='snn'")
        if not hasattr(self.model, "set_inference_lag_mask"):
            raise AttributeError("Model does not support inference lag masking")

        evidence = self._load_analysis_artifact(evidence_path)
        selected_trial_records, evidence_source = self._prepare_peak_ablation_inputs(evidence)
        self.logger.info(
            f"Lag peak ablation: loaded evidence from {evidence_path} using '{evidence_source}' records "
            f"({len(selected_trial_records)} trial entries before max_trials filtering)"
        )

        filtered_trial_records = []
        for trial_record in selected_trial_records:
            filtered_trial_records.append(trial_record)
            if max_trials is not None and len(filtered_trial_records) >= int(max_trials):
                break
        selected_trial_records = filtered_trial_records

        if not selected_trial_records:
            raise ValueError("No trial records with trial_id were available for ablation analysis")

        evidence_by_trial_id = {record["trial_id"]: record for record in selected_trial_records}
        selected_trial_ids = set(evidence_by_trial_id.keys())
        self.logger.info(
            f"Lag peak ablation: analyzing {len(selected_trial_ids)} selected trial ids | "
            f"block_idx={block_idx} | window_radius={window_radius} | "
            f"use_saved_peak_indices={use_saved_peak_indices}"
        )

        lag_values = list(self.model.get_lag_values(block_idx))
        num_lags = len(lag_values)
        self.logger.info(f"Lag peak ablation: lag values for block {block_idx}: {lag_values}")

        ablation_specs = []
        for lag_idx, lag_value in enumerate(lag_values):
            lag_mask = np.ones(num_lags, dtype=np.float32)
            lag_mask[lag_idx] = 0.0
            ablation_specs.append(
                {
                    "condition_index": len(ablation_specs),
                    "condition_name": "single_lag_ablation",
                    "lag_index": int(lag_idx),
                    "lag_value": int(lag_value),
                    "lag_mask": lag_mask,
                }
            )

        if 0 not in lag_values:
            raise ValueError(
                f"run_lag_peak_ablation expects lag_values for block {block_idx} to contain 0 so it can collect the only_lag_0 condition. Got: {lag_values}"
            )
        zero_lag_idx = int(lag_values.index(0))
        lag_mask = np.zeros(num_lags, dtype=np.float32)
        lag_mask[zero_lag_idx] = 1.0
        ablation_specs.append(
            {
                "condition_index": len(ablation_specs),
                "condition_name": "only_lag_0",
                "lag_index": int(zero_lag_idx),
                "lag_value": 0,
                "lag_mask": lag_mask,
            }
        )

        num_ablation_conditions = len(ablation_specs)
        self.logger.info(
            "Lag peak ablation: evaluating conditions: "
            f"{[(spec['condition_name'], spec['lag_index'], spec['lag_value']) for spec in ablation_specs]}"
        )

        if output_path is None:
            evidence_stem = os.path.splitext(os.path.basename(evidence_path))[0]
            output_path = os.path.join(
                self.ctc_evidence_dir,
                f"{evidence_stem}_lag_peak_ablation_block_{int(block_idx):02d}.pt",
            )

        aggregate_at_peak = {
            condition_idx: {
                "delta_blank_probability": [],
                "delta_top_non_blank_probability": [],
                "delta_entropy": [],
                "logit_l1_diff": [],
                "logit_l2_diff": [],
            }
            for condition_idx in range(num_ablation_conditions)
        }
        aggregate_by_relative_time = {
            condition_idx: {
                rel_offset: {
                    "delta_blank_probability": [],
                    "delta_top_non_blank_probability": [],
                    "delta_entropy": [],
                    "logit_l1_diff": [],
                    "logit_l2_diff": [],
                }
                for rel_offset in range(-int(window_radius), int(window_radius) + 1)
            }
            for condition_idx in range(num_ablation_conditions)
        }
        class_breakdown = defaultdict(
            lambda: {
                condition_idx: {
                    "delta_blank_probability": [],
                    "delta_top_non_blank_probability": [],
                    "delta_entropy": [],
                    "logit_l1_diff": [],
                    "logit_l2_diff": [],
                }
                for condition_idx in range(num_ablation_conditions)
            }
        )

        analysis_records = []
        num_batches_with_targets = 0
        num_matched_trials = 0
        num_peaks_detected = 0
        num_peaks_in_range = 0
        unmatched_trial_ids = set(selected_trial_ids)
        self.model.eval()
        self.model.clear_inference_lag_masks()

        with torch.inference_mode():
            for batch_idx, batch in enumerate(self.val_loader):
                batch_trial_ids = [
                    self._build_trial_id(
                        batch["day_indicies"][idx],
                        batch["block_nums"][idx],
                        batch["trial_nums"][idx],
                    )
                    for idx in range(len(batch["trial_nums"]))
                ]
                selected_positions = [
                    idx for idx, trial_id in enumerate(batch_trial_ids) if trial_id in selected_trial_ids
                ]
                if not selected_positions:
                    continue

                num_batches_with_targets += 1
                num_matched_trials += len(selected_positions)
                for idx in selected_positions:
                    unmatched_trial_ids.discard(batch_trial_ids[idx])

                self.logger.info(
                    f"Lag peak ablation: matched {len(selected_positions)} trial(s) in val batch {batch_idx}"
                )

                features = batch["input_features"].to(self.device)
                n_time_steps = batch["n_time_steps"].to(self.device)
                day_indicies = batch["day_indicies"].to(self.device)
                session_ids = batch.get("session_ids", batch["day_indicies"]).to(self.device)

                features, n_time_steps = self.transform_data(features, n_time_steps, "val")
                adjusted_lens = self.compute_output_lengths(n_time_steps)
                session_summaries = self._lookup_session_summaries(session_ids)

                baseline_logits = self._forward_model(features, day_indicies, session_summaries)
                adjusted_lens = torch.clamp(adjusted_lens, max=baseline_logits.shape[1])

                baseline_metrics = {}
                for idx in selected_positions:
                    trial_id = batch_trial_ids[idx]
                    baseline_metrics[trial_id] = self._compute_ctc_timestep_metrics(
                        baseline_logits[idx],
                        adjusted_lens[idx],
                        include_logits=True,
                    )

                ablated_metrics_by_lag = {}
                for spec in ablation_specs:
                    condition_idx = spec["condition_index"]
                    self.model.set_inference_lag_mask(block_idx, spec["lag_mask"])
                    ablated_logits = self._forward_model(features, day_indicies, session_summaries)
                    self.model.clear_inference_lag_masks()

                    ablated_metrics_by_lag[condition_idx] = {}
                    for idx in selected_positions:
                        trial_id = batch_trial_ids[idx]
                        ablated_metrics_by_lag[condition_idx][trial_id] = self._compute_ctc_timestep_metrics(
                            ablated_logits[idx],
                            adjusted_lens[idx],
                            include_logits=True,
                        )

                for idx in selected_positions:
                    trial_id = batch_trial_ids[idx]
                    trial_record = evidence_by_trial_id[trial_id]
                    baseline_trial_metrics = baseline_metrics[trial_id]
                    peak_indices = self._detect_peak_indices(
                        trial_record,
                        top_peak_min_prob=top_peak_min_prob,
                        blank_probability_max=blank_probability_max,
                        min_peak_distance=min_peak_distance,
                        use_saved_peak_indices=use_saved_peak_indices,
                    )
                    num_peaks_detected += int(len(peak_indices))
                    self.logger.info(
                        f"Lag peak ablation: trial {trial_id} yielded {len(peak_indices)} candidate peak(s)"
                    )

                    for peak_timestep in peak_indices.tolist():
                        if peak_timestep < 0 or peak_timestep >= baseline_trial_metrics["output_length"]:
                            self.logger.info(
                                f"Lag peak ablation: skipping out-of-range peak t={peak_timestep} "
                                f"for trial {trial_id} with output_length={baseline_trial_metrics['output_length']}"
                            )
                            continue

                        num_peaks_in_range += 1
                        predicted_class = int(baseline_trial_metrics["argmax_class"][peak_timestep])
                        baseline_window = self._extract_window_metrics(
                            baseline_trial_metrics,
                            center_timestep=peak_timestep,
                            window_radius=window_radius,
                        )

                        peak_result = {
                            "trial_id": trial_id,
                            "peak_timestep": int(peak_timestep),
                            "predicted_class_at_peak": predicted_class,
                            "predicted_phoneme_at_peak": (
                                LOGIT_TO_PHONEME[predicted_class]
                                if 0 <= predicted_class < len(LOGIT_TO_PHONEME)
                                else None
                            ),
                            "baseline_metrics_at_peak": self._build_peak_baseline_summary(
                                baseline_trial_metrics,
                                peak_timestep=peak_timestep,
                                predicted_class=predicted_class,
                            ),
                            # Save the full baseline class traces so later notebook analyses
                            # can recover local sequence context around each detected peak.
                            "argmax_class": baseline_trial_metrics["argmax_class"].copy(),
                            "top_non_blank_class": baseline_trial_metrics["top_non_blank_class"].copy(),
                            "blank_probability": baseline_trial_metrics["blank_probability"].copy(),
                            "top_non_blank_probability": baseline_trial_metrics["top_non_blank_probability"].copy(),
                            "entropy": baseline_trial_metrics["entropy"].copy(),
                            "baseline_window": {
                                "relative_offsets": baseline_window["relative_offsets"],
                                "blank_probability": baseline_window["blank_probability"],
                                "top_non_blank_probability": baseline_window["top_non_blank_probability"],
                                "entropy": baseline_window["entropy"],
                                "argmax_class": baseline_window["argmax_class"],
                                "top_non_blank_class": baseline_window["top_non_blank_class"],
                            },
                            "lag_results": [],
                        }

                        previous_phoneme_info = self._recover_previous_phoneme_from_sequence(
                            baseline_trial_metrics["argmax_class"],
                            peak_timestep=peak_timestep,
                        )
                        if previous_phoneme_info is not None:
                            peak_result["previous_class_at_peak"] = int(previous_phoneme_info["previous_class"])
                            peak_result["previous_phoneme_at_peak"] = previous_phoneme_info["previous_phoneme"]
                            peak_result["previous_phoneme_category"] = previous_phoneme_info["previous_category"]
                            peak_result["previous_phoneme_source"] = previous_phoneme_info["source"]

                        for spec in ablation_specs:
                            condition_idx = spec["condition_index"]
                            ablated_trial_metrics = ablated_metrics_by_lag[condition_idx][trial_id]
                            ablated_window = self._extract_window_metrics(
                                ablated_trial_metrics,
                                center_timestep=peak_timestep,
                                window_radius=window_radius,
                            )
                            importance_window = self._summarize_window_importance(
                                baseline_window,
                                ablated_window,
                            )

                            zero_offset_idx = np.where(importance_window["relative_offsets"] == 0)[0]
                            if len(zero_offset_idx) != 1:
                                continue
                            zero_offset_idx = int(zero_offset_idx[0])

                            lag_result = {
                                "condition_index": int(condition_idx),
                                "condition_name": spec["condition_name"],
                                "lag_index": int(spec["lag_index"]),
                                "lag_value": int(spec["lag_value"]),
                                "ablated_metrics_at_peak": {
                                    "blank_probability": float(ablated_trial_metrics["blank_probability"][peak_timestep]),
                                    "top_non_blank_probability": float(
                                        ablated_trial_metrics["top_non_blank_probability"][peak_timestep]
                                    ),
                                    "entropy": float(ablated_trial_metrics["entropy"][peak_timestep]),
                                    "argmax_class": int(ablated_trial_metrics["argmax_class"][peak_timestep]),
                                    "top_non_blank_class": int(ablated_trial_metrics["top_non_blank_class"][peak_timestep]),
                                },
                                "ablated_window": {
                                    "relative_offsets": ablated_window["relative_offsets"],
                                    "blank_probability": ablated_window["blank_probability"],
                                    "top_non_blank_probability": ablated_window["top_non_blank_probability"],
                                    "entropy": ablated_window["entropy"],
                                    "argmax_class": ablated_window["argmax_class"],
                                    "top_non_blank_class": ablated_window["top_non_blank_class"],
                                },
                                # Importance uses ablated - baseline so positive values mean the metric increased after masking this lag.
                                "importance_window": importance_window,
                                "importance_at_peak": {
                                    "delta_blank_probability": float(
                                        importance_window["delta_blank_probability"][zero_offset_idx]
                                    ),
                                    "delta_top_non_blank_probability": float(
                                        importance_window["delta_top_non_blank_probability"][zero_offset_idx]
                                    ),
                                    "delta_entropy": float(importance_window["delta_entropy"][zero_offset_idx]),
                                    "logit_l1_diff": float(importance_window["logit_l1_diff"][zero_offset_idx]),
                                    "logit_l2_diff": float(importance_window["logit_l2_diff"][zero_offset_idx]),
                                },
                            }
                            peak_result["lag_results"].append(lag_result)

                            for metric_name, values in aggregate_at_peak[condition_idx].items():
                                values.append(lag_result["importance_at_peak"][metric_name])

                            for rel_idx, rel_offset in enumerate(importance_window["relative_offsets"]):
                                rel_bucket = aggregate_by_relative_time[condition_idx][int(rel_offset)]
                                rel_bucket["delta_blank_probability"].append(
                                    float(importance_window["delta_blank_probability"][rel_idx])
                                )
                                rel_bucket["delta_top_non_blank_probability"].append(
                                    float(importance_window["delta_top_non_blank_probability"][rel_idx])
                                )
                                rel_bucket["delta_entropy"].append(float(importance_window["delta_entropy"][rel_idx]))
                                rel_bucket["logit_l1_diff"].append(float(importance_window["logit_l1_diff"][rel_idx]))
                                rel_bucket["logit_l2_diff"].append(float(importance_window["logit_l2_diff"][rel_idx]))

                            class_bucket = class_breakdown[predicted_class][condition_idx]
                            for metric_name, values in class_bucket.items():
                                values.append(lag_result["importance_at_peak"][metric_name])

                        analysis_records.append(peak_result)

        self.model.clear_inference_lag_masks()

        if unmatched_trial_ids:
            sample_unmatched = sorted(unmatched_trial_ids)[:5]
            self.logger.warning(
                f"Lag peak ablation: {len(unmatched_trial_ids)} selected trial id(s) were not found in the validation loader. "
                f"Sample: {sample_unmatched}"
            )

        self.logger.info(
            f"Lag peak ablation summary: matched_trials={num_matched_trials} | "
            f"matched_batches={num_batches_with_targets} | detected_peaks={num_peaks_detected} | "
            f"in_range_peaks={num_peaks_in_range} | saved_peak_records={len(analysis_records)}"
        )

        if num_batches_with_targets == 0:
            sample_selected = sorted(selected_trial_ids)[:5]
            sample_loader = []
            for batch in self.val_loader:
                for idx in range(min(len(batch["trial_nums"]), 5 - len(sample_loader))):
                    sample_loader.append(
                        self._build_trial_id(
                            batch["day_indicies"][idx],
                            batch["block_nums"][idx],
                            batch["trial_nums"][idx],
                        )
                    )
                if len(sample_loader) >= 5:
                    break
            self.logger.warning(
                "Lag peak ablation found no matching trials in the validation loader. "
                f"Selected sample ids: {sample_selected} | Loader sample ids: {sample_loader}"
            )
        elif len(analysis_records) == 0:
            self.logger.warning(
                "Lag peak ablation matched trials but produced no peak records. "
                "This usually means no peaks passed the current detection settings."
            )

        aggregate_mean_by_lag = []
        for spec in ablation_specs:
            condition_idx = spec["condition_index"]
            lag_summary = {
                "condition_index": int(condition_idx),
                "condition_name": spec["condition_name"],
                "lag_index": int(spec["lag_index"]),
                "lag_value": int(spec["lag_value"]),
                "num_peaks": len(aggregate_at_peak[condition_idx]["logit_l2_diff"]),
            }
            for metric_name, values in aggregate_at_peak[condition_idx].items():
                lag_summary[f"mean_{metric_name}"] = float(np.mean(values)) if values else None
            aggregate_mean_by_lag.append(lag_summary)

        aggregate_relative_time = []
        for spec in ablation_specs:
            condition_idx = spec["condition_index"]
            for rel_offset in range(-int(window_radius), int(window_radius) + 1):
                rel_bucket = aggregate_by_relative_time[condition_idx][rel_offset]
                row = {
                    "condition_index": int(condition_idx),
                    "condition_name": spec["condition_name"],
                    "lag_index": int(spec["lag_index"]),
                    "lag_value": int(spec["lag_value"]),
                    "relative_offset": int(rel_offset),
                    "num_samples": len(rel_bucket["logit_l2_diff"]),
                }
                for metric_name, values in rel_bucket.items():
                    row[f"mean_{metric_name}"] = float(np.mean(values)) if values else None
                aggregate_relative_time.append(row)

        aggregate_by_class = []
        for predicted_class, lag_dict in class_breakdown.items():
            for spec in ablation_specs:
                condition_idx = spec["condition_index"]
                metrics_dict = lag_dict[condition_idx]
                row = {
                    "predicted_class": int(predicted_class),
                    "predicted_phoneme": (
                        LOGIT_TO_PHONEME[int(predicted_class)]
                        if 0 <= int(predicted_class) < len(LOGIT_TO_PHONEME)
                        else None
                    ),
                    "condition_index": int(condition_idx),
                    "condition_name": spec["condition_name"],
                    "lag_index": int(spec["lag_index"]),
                    "lag_value": int(spec["lag_value"]),
                    "num_peaks": len(metrics_dict["logit_l2_diff"]),
                }
                for metric_name, values in metrics_dict.items():
                    row[f"mean_{metric_name}"] = float(np.mean(values)) if values else None
                aggregate_by_class.append(row)

        results = {
            "metadata": {
                "analysis_type": "lag_peak_ablation",
                "evidence_path": str(evidence_path),
                "block_idx": int(block_idx),
                "lag_values": [int(v) for v in lag_values],
                "include_only_lag0": True,
                "ablation_conditions": [
                    {
                        "condition_index": int(spec["condition_index"]),
                        "condition_name": spec["condition_name"],
                        "lag_index": int(spec["lag_index"]),
                        "lag_value": int(spec["lag_value"]),
                    }
                    for spec in ablation_specs
                ],
                "window_radius": int(window_radius),
                "top_peak_min_prob": float(top_peak_min_prob),
                "blank_probability_max": (
                    float(blank_probability_max) if blank_probability_max is not None else None
                ),
                "min_peak_distance": int(min_peak_distance),
                "use_saved_peak_indices": bool(use_saved_peak_indices),
                "num_selected_trials": int(len(selected_trial_ids)),
                "num_matched_trials": int(num_matched_trials),
                "num_peak_records": int(len(analysis_records)),
                "num_detected_peaks": int(num_peaks_detected),
                "num_in_range_peaks": int(num_peaks_in_range),
                "num_batches_with_targets": int(num_batches_with_targets),
                "evidence_source": evidence_source,
            },
            "aggregate": {
                "mean_by_lag": aggregate_mean_by_lag,
                "relative_time_by_lag": aggregate_relative_time,
                "by_predicted_class": aggregate_by_class,
            },
            "peaks": analysis_records,
        }

        output_parent = os.path.dirname(output_path)
        if output_parent:
            os.makedirs(output_parent, exist_ok=True)
        torch.save(results, output_path)
        self.logger.info(f"Saved lag peak ablation analysis to {output_path}")
        return output_path



    def create_optimizer(self):
        '''
        Create the optimizer with special param groups 
    
        Biases and day weights should not be decayed
    
        Day weights should have a separate learning rate
        '''
        day_params = [p for name, p in self.model.named_parameters() if 'day_' in name]
        bias_params = [p for name, p in self.model.named_parameters() if 'day_' not in name and name.endswith('.bias')]
        other_params = [p for name, p in self.model.named_parameters() if 'day_' not in name and not name.endswith('.bias')]
        bias_params = [p for p in bias_params if p.requires_grad]
        day_params = [p for p in day_params if p.requires_grad]
        other_params = [p for p in other_params if p.requires_grad]
    
        if len(day_params) != 0:
            param_groups = [
                    {'params' : bias_params, 'weight_decay' : 0, 'group_type' : 'bias'},
                    {'params' : day_params, 'lr' : self.args['lr_max_day'], 'weight_decay' : self.args['weight_decay_day'], 'group_type' : 'day_layer'},
                    {'params' : other_params, 'group_type' : 'other'}
                ]
        else: 
            param_groups = [
                    {'params' : bias_params, 'weight_decay' : 0, 'group_type' : 'bias'},
                    {'params' : other_params, 'group_type' : 'other'}
                ]
    
        optim = torch.optim.AdamW(
            param_groups,
            lr = self.args['lr_max'],
            betas = (self.args['beta0'], self.args['beta1']),
            eps = self.args['epsilon'],
            weight_decay = self.args['weight_decay'],
            fused = True
        )
    
        return optim 
    
    def create_cosine_lr_scheduler(self, optim):
        lr_max = self.args['lr_max']
        lr_min = self.args['lr_min']
        lr_decay_steps = self.args['lr_decay_steps']
    
        lr_max_day =  self.args['lr_max_day']
        lr_min_day = self.args['lr_min_day']
        lr_decay_steps_day = self.args['lr_decay_steps_day']
    
        lr_warmup_steps = self.args['lr_warmup_steps']
        lr_warmup_steps_day = self.args['lr_warmup_steps_day']
    
        def lr_lambda(current_step, min_lr_ratio, decay_steps, warmup_steps):
            '''
            Create lr lambdas for each param group that implement cosine decay
    
            Different lr lambda decaying for day params vs rest of the model
            '''
            # Warmup phase
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
    
            # Cosine decay phase
            if current_step < decay_steps:
                progress = float(current_step - warmup_steps) / float(
                    max(1, decay_steps - warmup_steps)
                )
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                # Scale from 1.0 to min_lr_ratio
                return max(min_lr_ratio, min_lr_ratio + (1 - min_lr_ratio) * cosine_decay)
    
            # After cosine decay is complete, maintain min_lr_ratio
            return min_lr_ratio
    
        if len(optim.param_groups) == 3:
            lr_lambdas = [
                lambda step: lr_lambda(
                    step, 
                    lr_min / lr_max, 
                    lr_decay_steps, 
                    lr_warmup_steps), # biases 
                lambda step: lr_lambda(
                    step, 
                    lr_min_day / lr_max_day, 
                    lr_decay_steps_day,
                    lr_warmup_steps_day, 
                    ), # day params
                lambda step: lr_lambda(
                    step, 
                    lr_min / lr_max, 
                    lr_decay_steps, 
                    lr_warmup_steps), # rest of model weights
            ]
        elif len(optim.param_groups) == 2:
            lr_lambdas = [
                lambda step: lr_lambda(
                    step, 
                    lr_min / lr_max, 
                    lr_decay_steps, 
                    lr_warmup_steps), # biases 
                lambda step: lr_lambda(
                    step, 
                    lr_min / lr_max, 
                    lr_decay_steps, 
                    lr_warmup_steps), # rest of model weights
            ]
        else:
            raise ValueError(f"Invalid number of param groups in optimizer: {len(optim.param_groups)}")
    
        return LambdaLR(optim, lr_lambdas, -1)
    
    def load_model_checkpoint(self, load_path):
        ''' 
        Load a training checkpoint
        '''
        checkpoint = torch.load(
            load_path,
            map_location=self.device,
            weights_only=False,
        ) # checkpoint is just a dict
    
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.learning_rate_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_PER = checkpoint['val_PER'] # best phoneme error rate
        self.best_val_loss = checkpoint['val_loss'] if 'val_loss' in checkpoint.keys() else torch.inf
    
        self.model.to(self.device)
    
        # Send optimizer params back to GPU
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
    
        self.logger.info("Loaded model from checkpoint: " + load_path)
    
    def save_model_checkpoint(self, save_path, PER, loss):
        '''
        Save a training checkpoint
        '''
        checkpoint_parent = os.path.dirname(save_path)
        if checkpoint_parent:
            os.makedirs(checkpoint_parent, exist_ok=True)
        os.makedirs(self.args['checkpoint_dir'], exist_ok=True)
        self._verify_optimizer_parameter_consistency()
    
        checkpoint = {
            'model_state_dict' : self.model.state_dict(),
            'optimizer_state_dict' : self.optimizer.state_dict(),
            'scheduler_state_dict' : self.learning_rate_scheduler.state_dict(),
            'val_PER' : PER,
            'val_loss' : loss
        }
    
        torch.save(checkpoint, save_path)
    
        self.logger.info("Saved model to checkpoint: " + save_path)
    
        # Save the args file alongside the checkpoint
        with open(os.path.join(self.args['checkpoint_dir'], 'args.yaml'), 'w') as f:
            OmegaConf.save(config=self.args, f=f)
    
    def create_attention_mask(self, sequence_lengths):
    
        max_length = torch.max(sequence_lengths).item()
    
        batch_size = sequence_lengths.size(0)
    
        # Create a mask for valid key positions (columns)
        # Shape: [batch_size, max_length]
        key_mask = torch.arange(max_length, device=sequence_lengths.device).expand(batch_size, max_length)
        key_mask = key_mask < sequence_lengths.unsqueeze(1)
    
        # Expand key_mask to [batch_size, 1, 1, max_length]
        # This will be broadcast across all query positions
        key_mask = key_mask.unsqueeze(1).unsqueeze(1)
    
        # Create the attention mask of shape [batch_size, 1, max_length, max_length]
        # by broadcasting key_mask across all query positions
        attention_mask = key_mask.expand(batch_size, 1, max_length, max_length)
    
        # Convert boolean mask to float mask:
        # - True (valid key positions) -> 0.0 (no change to attention scores)
        # - False (padding key positions) -> -inf (will become 0 after softmax)
        attention_mask_float = torch.where(attention_mask, 
                                        True,
                                        False)
    
        return attention_mask_float
    
    def compute_output_lengths(self, n_time_steps):
        if self.args.get("model_type") == "lssca_snn":
            window_size = int(self.args["model"].get("lss_window_size", 100))
            window_stride = int(self.args["model"].get("lss_window_stride", 4))
            if window_size <= 0 or window_stride <= 0:
                raise ValueError(
                    "lssca_snn requires model.lss_window_size > 0 and model.lss_window_stride > 0"
                )

            output_lengths = torch.where(
                n_time_steps >= window_size,
                (n_time_steps - window_size) // window_stride + 1,
                torch.ones_like(n_time_steps),
            )
            return torch.clamp(output_lengths, min=1).to(torch.int32)

        patch_size = int(self.args['model']['patch_size'])
        patch_stride = int(self.args['model']['patch_stride'])
    
        if patch_size > 0 and patch_stride > 0:
            output_lengths = (n_time_steps - patch_size) // patch_stride + 1
        else:
            output_lengths = n_time_steps
    
        return torch.clamp(output_lengths, min=1).to(torch.int32)
    
    def transform_data(self, features, n_time_steps, mode = 'train'):
    
            '''
            Apply various augmentations and smoothing to data
            Performing augmentations is much faster on GPU than CPU
            '''
    
            data_shape = features.shape
            batch_size = data_shape[0]
            channels = data_shape[-1]
    
            # We only apply these augmentations in training
            if mode == 'train':
                # add static gain noise 
                if self.transform_args['static_gain_std'] > 0:
                    warp_mat = torch.tile(torch.unsqueeze(torch.eye(channels), dim = 0), (batch_size, 1, 1))
                    warp_mat += torch.randn_like(warp_mat, device=self.device) * self.transform_args['static_gain_std']
    
                    features = torch.matmul(features, warp_mat)
    
                # add white noise
                if self.transform_args['white_noise_std'] > 0:
                    features += torch.randn(data_shape, device=self.device) * self.transform_args['white_noise_std']
    
                # add constant offset noise 
                if self.transform_args['constant_offset_std'] > 0:
                    features += torch.randn((batch_size, 1, channels), device=self.device) * self.transform_args['constant_offset_std']
    
                # add random walk noise
                if self.transform_args['random_walk_std'] > 0:
                    features += torch.cumsum(torch.randn(data_shape, device=self.device) * self.transform_args['random_walk_std'], dim =self.transform_args['random_walk_axis'])
    
                # randomly cutoff part of the data timecourse
                if self.transform_args['random_cut'] > 0:
                    cut = np.random.randint(0, self.transform_args['random_cut'])
                    features = features[:, cut:, :]
                    n_time_steps = n_time_steps - cut
    
            # Apply Gaussian smoothing to data 
            # This is done in both training and validation
            if self.transform_args['smooth_data']:
                features = gauss_smooth(
                    inputs = features, 
                    device = self.device,
                    smooth_kernel_std = self.transform_args['smooth_kernel_std'],
                    smooth_kernel_size= self.transform_args['smooth_kernel_size'],
                    )
                
            
            return features, n_time_steps
    
    
    def train(self):
        """
        Train the model
        """

        self.model.train()

        train_losses, val_losses, val_PERs, val_results = [], [], [], []
        val_steps_since_improvement = 0

        save_best_checkpoint = self.args.get("save_best_checkpoint", True)
        early_stopping = self.args.get("early_stopping", True)
        early_stopping_val_steps = self.args["early_stopping_val_steps"]

        train_start_time = time.time()
        self._active_epoch_index = 0

        for i, batch in enumerate(self.train_loader):
            epoch_index = i // max(1, self.train_batches_per_epoch)
            self._maybe_freeze_session_adapter(i)
            if self._active_epoch_index is None:
                self._active_epoch_index = epoch_index
            elif epoch_index != self._active_epoch_index:
                self._flush_epoch_spiking_stats(self._active_epoch_index)
                self._active_epoch_index = epoch_index

            self.model.train()
            self.optimizer.zero_grad()
            start_time = time.time()

            # Move data to device
            features = batch["input_features"].to(self.device)
            labels = batch["seq_class_ids"].to(self.device)
            n_time_steps = batch["n_time_steps"].to(self.device)
            phone_seq_lens = batch["phone_seq_lens"].to(self.device)
            day_indicies = batch["day_indicies"].to(self.device)
            session_ids = batch.get("session_ids", batch["day_indicies"]).to(self.device)

            log_stats = (i % 200 == 0)

            with torch.autocast(
                device_type="cuda",
                enabled=self.args["use_amp"],
                dtype=torch.float16,
            ):
                # Apply augmentations
                features, n_time_steps = self.transform_data(features, n_time_steps, "train")

                adjusted_lens = self.compute_output_lengths(n_time_steps)
                session_summaries = self._lookup_session_summaries(session_ids)

                # Forward
                logits = self._forward_model(features, day_indicies, session_summaries)  # [B, T, C]

                # CTC expects input_lengths <= logits time dim
                T_out = logits.shape[1]
                adjusted_lens = torch.clamp(adjusted_lens, max=T_out)

                if i == 0:
                    self.logger.info(
                        f"logits T_out={T_out}, adjusted_lens max={adjusted_lens.max().item()}, min={adjusted_lens.min().item()}"
                    )

                # ---- KEY STATS BLOCK (single block, cheap, BEFORE backward) ----
                if log_stats:
                    with torch.no_grad():
                        B, T, C = logits.shape
                        lens = torch.clamp(adjusted_lens, min=1, max=T)

                        t_idx = torch.arange(T, device=logits.device).unsqueeze(0)  # [1, T]
                        mask = t_idx < lens.unsqueeze(1)                             # [B, T]
                        logits_v = logits[mask]                                      # [N, C]

                        preds_v = logits_v.argmax(dim=-1)
                        blank_ratio = (preds_v == 0).float().mean().item()

                        avg_logit_mag = logits_v.abs().mean().item()

                        logp_v = torch.log_softmax(logits_v, dim=-1)
                        p_v = logp_v.exp()

                        top1p = p_v.max(dim=-1).values.mean().item()
                        entropy = (-(p_v * logp_v).sum(dim=-1)).mean().item()
                # ---- END KEY STATS BLOCK ----

                # Loss
                loss = self.ctc_loss(
                    log_probs=torch.permute(logits.log_softmax(2), [1, 0, 2]),
                    targets=labels,
                    input_lengths=adjusted_lens,
                    target_lengths=phone_seq_lens,
                )
                loss = torch.mean(loss)

            # Backward (ONCE)
            loss.backward()
            self._save_spiking_batch_stats(batch_index=i, epoch_index=epoch_index)

            # Clip gradient (grad_norm returned is pre-clip total norm)
            grad_norm = None
            clip_frac = None
            if self.args["grad_norm_clip_value"] > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.args["grad_norm_clip_value"],
                    error_if_nonfinite=False,
                    foreach=True,
                )
                clip_frac = min(
                    1.0,
                    float(self.args["grad_norm_clip_value"]) / (float(grad_norm) + 1e-12),
                )

            # Step
            self.optimizer.step()
            self.learning_rate_scheduler.step()
            self._save_session_threshold_batch_stats(epoch_index=epoch_index, batch_index=i)
            if i % 2000 == 0:
                self._log_global_neuron_parameters(prefix=f"Batch {int(i)}")

            # Save training metrics
            train_step_duration = time.time() - start_time
            train_losses.append(loss.detach().item())

            # Unified stats logging (low frequency)
            if log_stats:
                lr = self.optimizer.param_groups[0]["lr"]
                self.logger.info(
                    f"step {i} "
                    f"loss={loss.detach().item():.3f} "
                    f"lr={lr:.4g} "
                    f"blank_ratio={blank_ratio:.4f} "
                    f"logit|x|={avg_logit_mag:.4f} "
                    f"top1p={top1p:.4f} "
                    f"entropy={entropy:.4f} "
                    f"grad_norm={float(grad_norm) if grad_norm is not None else float('nan'):.2f} "
                    f"clip_frac={clip_frac if clip_frac is not None else float('nan'):.3f} "
                    f"time={train_step_duration:.3f}"
                )

            # Your existing training progress log (optional; keep if you want)
            if i % self.args["batches_per_train_log"] == 0 and not log_stats:
                # avoid duplicate logging when log_stats=True
                self.logger.info(
                    f"Train batch {i}: "
                    f"loss: {loss.detach().item():.2f} "
                    f"grad norm: {float(grad_norm) if grad_norm is not None else float('nan'):.2f} "
                    f"time: {train_step_duration:.3f}"
                )

            if (
                self.delay_parameter_export_interval > 0
                and i > 0
                and (i % self.delay_parameter_export_interval == 0)
            ):
                self._save_delay_parameter_checkpoint(
                    batch_index=i,
                    epoch_index=epoch_index,
                    event_name=f"periodic_{i:07d}",
                )

            # Validation
            if i % self.args["batches_per_val_step"] == 0 or i == (self.args["num_training_batches"] - 1):
                self.logger.info(f"Running test after training batch: {i}")

                start_time = time.time()
                val_metrics = self.validation(
                    loader=self.val_loader,
                    return_logits=self.args["save_val_logits"],
                    return_data=self.args["save_val_data"],
                )
                val_step_duration = time.time() - start_time

                delay_event_name = f"validation/epoch_{epoch_index:04d}_batch_{i:07d}"
                saved_delay_paths = self._save_delay_statistics(
                    batch_index=i,
                    epoch_index=epoch_index,
                    event_name=delay_event_name,
                    store_full=True,
                )
                if saved_delay_paths:
                    self.logger.info(
                        f"Saved delay dynamics snapshots for {len(saved_delay_paths)} block(s) to {os.path.join(self.delay_stats_dir, delay_event_name)}"
                    )

                # Export lag data periodically during training
                #if i % (self.args["batches_per_val_step"] * 5) == 0:  # Every 5 validation steps
                #    for j, block in enumerate(self.model.backbone.snn_core):
                 #       mixer = block.block[0]
                 #       mixer.export_learned_lag_summary(f"{self.args['checkpoint_dir']}/learned_lag_summary_block_{j}_batch_{i}.csv")
                   #     mixer.export_weights(f"{self.args['checkpoint_dir']}/learned_lag_weights_block_{j}_batch_{i}.csv")

                self.logger.info(
                    f"Val batch {i}: "
                    f"PER (avg): {val_metrics['avg_PER']:.4f} "
                    f"CTC Loss (avg): {val_metrics['avg_loss']:.4f} "
                    f"time: {val_step_duration:.3f}"
                )

                if self.args["log_individual_day_val_PER"]:
                    for day in val_metrics["day_PERs"].keys():
                        self.logger.info(
                            f"{self.args['dataset']['sessions'][day]} val PER: "
                            f"{val_metrics['day_PERs'][day]['total_edit_distance'] / val_metrics['day_PERs'][day]['total_seq_length']:0.4f}"
                        )

                val_PERs.append(val_metrics["avg_PER"])
                val_losses.append(val_metrics["avg_loss"])
                val_results.append(val_metrics)
                self._notify_validation_observers(
                    batch_index=i,
                    epoch_index=epoch_index,
                    val_metrics=val_metrics,
                )

                new_best = False
                if val_metrics["avg_PER"] < self.best_val_PER:
                    self.logger.info(
                        f"New best test PER {self.best_val_PER:.4f} --> {val_metrics['avg_PER']:.4f}"
                    )
                    self.best_val_PER = val_metrics["avg_PER"]
                    self.best_val_loss = val_metrics["avg_loss"]
                    new_best = True
                elif val_metrics["avg_PER"] == self.best_val_PER and (val_metrics["avg_loss"] < self.best_val_loss):
                    self.logger.info(
                        f"New best test loss {self.best_val_loss:.4f} --> {val_metrics['avg_loss']:.4f}"
                    )
                    self.best_val_loss = val_metrics["avg_loss"]
                    new_best = True

                if new_best:
                    if save_best_checkpoint:
                        self.logger.info("Checkpointing model")
                        self.save_model_checkpoint(
                            f"{self.args['checkpoint_dir']}/best_checkpoint",
                            self.best_val_PER,
                            self.best_val_loss,
                        )

                    self._save_delay_statistics(
                        batch_index=i,
                        epoch_index=epoch_index,
                        event_name="best",
                        store_full=True,
                    )

                    if self.args["save_val_metrics"]:
                        with open(f"{self.args['checkpoint_dir']}/val_metrics.pkl", "wb") as f:
                            pickle.dump(val_metrics, f)

                    val_steps_since_improvement = 0
                else:
                    val_steps_since_improvement += 1

                if self.args["save_all_val_steps"]:
                    self.save_model_checkpoint(
                        f"{self.args['checkpoint_dir']}/checkpoint_batch_{i}",
                        val_metrics["avg_PER"],
                    )

                if early_stopping and (val_steps_since_improvement >= early_stopping_val_steps):
                    self.logger.info(
                        f"Overall validation PER has not improved in {early_stopping_val_steps} validation steps. "
                        f"Stopping training early at batch: {i}"
                    )
                    break

        self._flush_epoch_spiking_stats(self._active_epoch_index)

        training_duration = time.time() - train_start_time
        self.logger.info(f"Best avg val PER achieved: {self.best_val_PER:.5f}")
        self.logger.info(f"Total training time: {(training_duration / 60):.2f} minutes")
        if self.spiking_stats_dir is not None:
            self.logger.info(f"Saved spiking statistics to: {self.spiking_stats_dir}")

        if self.exp_tag:
            self.logger.info(f"EXPERIMENT_TAG_FINAL: {self.exp_tag}")

        #for i, block in enumerate(self.model.backbone.snn_core):
        #    mixer = block.block[0]
        #    mixer.export_learned_lag_summary(f"{self.args['checkpoint_dir']}/learned_lag_summary_block_{i}.csv")
        #    mixer.export_weights(f"{self.args['checkpoint_dir']}/learned_lag_weights_block_{i}.csv")

        if self.args["save_final_model"] and len(val_PERs) > 0:
            self.save_model_checkpoint(
                f"{self.args['checkpoint_dir']}/final_checkpoint_batch_{i}",
                val_PERs[-1],
            )

        self._save_delay_statistics(
            batch_index=i,
            epoch_index=self._active_epoch_index if self._active_epoch_index is not None else 0,
            event_name="final",
            store_full=True,
        )
        self._save_delay_parameter_checkpoint(
            batch_index=i,
            epoch_index=self._active_epoch_index if self._active_epoch_index is not None else 0,
            event_name="final",
        )

        self.model.eval()

        '''for i, block in enumerate(self.model.backbone.snn_core):
            mixer = block.block[0]
            mixer.export_weights(f"{self.args['checkpoint_dir']}/lagged_weights_block_{i}.csv")'''

        train_stats = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_PERs": val_PERs,
            "val_metrics": val_results,
        }
        return train_stats

    def validation(
        self,
        loader,
        return_logits = False,
        return_data = False,
        collect_diagnostics = False,
        export_frame_data = False,
        include_alignment = False,
        alignment_boundary_radius = 2,
    ):
        '''
        Calculate metrics on the validation dataset
        '''
        self.model.eval()

        metrics = {}
        
        # Record metrics
        if return_logits: 
            metrics['logits'] = []
            metrics['n_time_steps'] = []

        if return_data: 
            metrics['input_features'] = []

        metrics['decoded_seqs'] = []
        metrics['true_seq'] = []
        metrics['phone_seq_lens'] = []
        metrics['transcription'] = []
        metrics['losses'] = []
        metrics['block_nums'] = []
        metrics['trial_nums'] = []
        metrics['day_indicies'] = []
        metrics['ctc_evidence_path'] = None
        if export_frame_data:
            metrics['trial_exports'] = []
            metrics['phoneme_classes'] = list(LOGIT_TO_PHONEME)

        ctc_evidence_records = []

        total_edit_distance = 0
        total_seq_length = 0
        total_blank_predictions = 0
        total_valid_frames = 0
        spike_rate_sums = defaultdict(float)
        spike_rate_counts = defaultdict(int)

        # Calculate PER for each specific day
        day_per = {}
        for d in range(len(self.args['dataset']['sessions'])):
            if self.args['dataset']['dataset_probability_val'][d] == 1: 
                day_per[d] = {'total_edit_distance' : 0, 'total_seq_length' : 0}

        for i, batch in enumerate(loader):        

            features = batch['input_features'].to(self.device)
            labels = batch['seq_class_ids'].to(self.device)
            n_time_steps = batch['n_time_steps'].to(self.device)
            phone_seq_lens = batch['phone_seq_lens'].to(self.device)
            day_indicies = batch['day_indicies'].to(self.device)
            session_ids = batch.get("session_ids", batch["day_indicies"]).to(self.device)

            # Determine if we should perform validation on this batch
            day = day_indicies[0].item()
            if self.args['dataset']['dataset_probability_val'][day] == 0: 
                if self.args['log_val_skip_logs']:
                    self.logger.info(f"Skipping validation on day {day}")
                continue
            
            with torch.no_grad():

                with torch.autocast(device_type = "cuda", enabled = self.args['use_amp'], dtype = torch.float16):
                    features, n_time_steps = self.transform_data(features, n_time_steps, 'val')

                    adjusted_lens = self.compute_output_lengths(n_time_steps)
                    session_summaries = self._lookup_session_summaries(session_ids)

                    logits = self._forward_model(features, day_indicies, session_summaries)
                    
                    # CTC expects input_lengths <= logits time dim
                    T_out = logits.shape[1]
                    adjusted_lens = torch.clamp(adjusted_lens, max=T_out)

                    if collect_diagnostics:
                        time_index = torch.arange(T_out, device=logits.device).unsqueeze(0)
                        valid_mask = time_index < adjusted_lens.unsqueeze(1)
                        predicted_classes = logits.argmax(dim=-1)
                        total_blank_predictions += int((predicted_classes[valid_mask] == 0).sum().item())
                        total_valid_frames += int(valid_mask.sum().item())

                        if self.args.get("model_type") == "snn" and hasattr(self.model, "get_spiking_layer_statistics"):
                            layer_stats = self.model.get_spiking_layer_statistics() or {}
                            for layer_name, stats in layer_stats.items():
                                mean_spike_rate = stats.get("mean_spike_rate")
                                if mean_spike_rate is None:
                                    continue
                                spike_rate_sums[layer_name] += float(mean_spike_rate)
                                spike_rate_counts[layer_name] += 1

                    # sanity check (do once)
                    if i == 0:
                        self.logger.info(f"logits T_out={T_out}, adjusted_lens max={adjusted_lens.max().item()}, min={adjusted_lens.min().item()}")

                    # sanity check (do once)
                    if i == 0:
                        self.logger.info(f"logits T_out={T_out}, adjusted_lens max={adjusted_lens.max().item()}, min={adjusted_lens.min().item()}")

                    loss = self.ctc_loss(
                        torch.permute(logits.log_softmax(2), [1, 0, 2]),
                        labels,
                        adjusted_lens,
                        phone_seq_lens,
                    )
                    loss = torch.mean(loss)

                # Calculate PER per day and also avg over entire validation set
                batch_edit_distance = 0 
                decoded_seqs = []
                for iterIdx in range(logits.shape[0]):
                    decoded_seq = torch.argmax(logits[iterIdx, 0 : adjusted_lens[iterIdx], :].clone().detach(),dim=-1)
                    decoded_seq = torch.unique_consecutive(decoded_seq, dim=-1)
                    decoded_seq = decoded_seq.cpu().detach().numpy()
                    decoded_seq = np.array([i for i in decoded_seq if i != 0])

                    trueSeq = np.array(
                        labels[iterIdx][0 : phone_seq_lens[iterIdx]].cpu().detach()
                    )
            
                    batch_edit_distance += F.edit_distance(decoded_seq, trueSeq)

                    decoded_seqs.append(decoded_seq)

                    ctc_evidence_records.append(
                        self._compute_ctc_evidence_record(
                            logits=logits[iterIdx],
                            adjusted_len=adjusted_lens[iterIdx],
                            day_idx=batch['day_indicies'][iterIdx],
                            block_num=batch['block_nums'][iterIdx],
                            trial_num=batch['trial_nums'][iterIdx],
                            labels=labels[iterIdx],
                            phone_seq_len=phone_seq_lens[iterIdx],
                            transcription=batch['transcriptions'][iterIdx],
                            decoded_seq=decoded_seq,
                        )
                    )
                    if export_frame_data:
                        metrics['trial_exports'].append(
                            self._build_validation_trial_export_record(
                                logits=logits[iterIdx],
                                adjusted_len=adjusted_lens[iterIdx],
                                day_idx=batch['day_indicies'][iterIdx],
                                block_num=batch['block_nums'][iterIdx],
                                trial_num=batch['trial_nums'][iterIdx],
                                labels=labels[iterIdx],
                                phone_seq_len=phone_seq_lens[iterIdx],
                                session_id=batch.get("session_ids", None)[iterIdx] if "session_ids" in batch else None,
                                include_alignment=include_alignment,
                                boundary_radius=alignment_boundary_radius,
                            )
                        )

            day = batch['day_indicies'][0].item()
                
            day_per[day]['total_edit_distance'] += batch_edit_distance
            day_per[day]['total_seq_length'] += torch.sum(phone_seq_lens).item()


            total_edit_distance += batch_edit_distance
            total_seq_length += torch.sum(phone_seq_lens)

            # Record metrics
            if return_logits: 
                metrics['logits'].append(logits.cpu().float().numpy()) # Will be in float16 if AMP is enabled, so need to set back to float32
                metrics['n_time_steps'].append(adjusted_lens.cpu().numpy())

            if return_data: 
                metrics['input_features'].append(batch['input_features'].cpu().numpy()) 

            metrics['decoded_seqs'].append(decoded_seqs)
            metrics['true_seq'].append(batch['seq_class_ids'].cpu().numpy())
            metrics['phone_seq_lens'].append(batch['phone_seq_lens'].cpu().numpy())
            metrics['transcription'].append(batch['transcriptions'].cpu().numpy())
            metrics['losses'].append(loss.detach().item())
            metrics['block_nums'].append(batch['block_nums'].numpy())
            metrics['trial_nums'].append(batch['trial_nums'].numpy())
            metrics['day_indicies'].append(batch['day_indicies'].cpu().numpy())

        avg_PER = total_edit_distance / total_seq_length

        metrics['day_PERs'] = day_per
        metrics['avg_PER'] = avg_PER.item()
        metrics['avg_loss'] = np.mean(metrics['losses'])
        metrics['ctc_evidence_path'] = self._save_ctc_evidence_records(ctc_evidence_records, split_name='validation')

        if collect_diagnostics:
            metrics['blank_ratio'] = (
                float(total_blank_predictions / total_valid_frames)
                if total_valid_frames > 0
                else float("nan")
            )
            metrics['spike_rates_by_layer'] = {
                layer_name: float(spike_rate_sums[layer_name] / spike_rate_counts[layer_name])
                for layer_name in sorted(spike_rate_sums.keys())
                if spike_rate_counts[layer_name] > 0
            }
            if metrics['spike_rates_by_layer']:
                metrics['spike_rate'] = float(
                    np.mean(list(metrics['spike_rates_by_layer'].values()))
                )
            else:
                metrics['spike_rate'] = None

        if metrics['ctc_evidence_path'] is not None:
            self.logger.info(f"Saved CTC evidence traces to {metrics['ctc_evidence_path']}")

        return metrics
