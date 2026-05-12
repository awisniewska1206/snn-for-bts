from pathlib import Path
import csv

import torch
from torch import nn
import torch.nn.functional as F


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _write_rows_to_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class LaggedMixer(nn.Module):
    """
    MLP-side port of the SNN lagged mixer so lag handling can match the SNN
    setup without editing any SNN files.
    """

    def __init__(
        self,
        channels,
        lags=(0, 1, 2, 4),
        one_random_lag_per_pair=False,
        num_lags_per_connection=1,
        lag_distribution_type="uniform",
        lag_decay_strength=1.0,
        seed=0,
    ):
        super().__init__()
        self.lags = list(lags)
        self.channels = int(channels)
        self.K = len(self.lags)
        if self.K <= 0:
            raise ValueError("lags must contain at least one entry")
        if any(int(lag) < 0 for lag in self.lags):
            raise ValueError(f"lags must be non-negative, got {self.lags}")
        self.lag_distribution_type = str(lag_distribution_type)
        self.lag_decay_strength = float(lag_decay_strength)
        self.one_random_lag_per_pair = bool(one_random_lag_per_pair)
        self.num_lags_per_connection = max(1, int(num_lags_per_connection))
        self.max_export_rows = 1_000_000
        self.max_snapshot_tensor_elements = 10_000_000

        if self.one_random_lag_per_pair:
            lag_probs = self._normalized_lag_probabilities()
            self.num_weight_banks = min(self.num_lags_per_connection, self.K)
            with torch.random.fork_rng():
                torch.manual_seed(int(seed))
                sampled_lag_indices = torch.multinomial(
                    lag_probs.expand(self.channels * self.channels, -1),
                    num_samples=self.num_weight_banks,
                    replacement=False,
                ).view(self.channels, self.channels, self.num_weight_banks)
            self.register_buffer(
                "selected_lag_indices",
                sampled_lag_indices.permute(2, 0, 1).contiguous(),
            )
        else:
            self.num_weight_banks = self.K
            self.selected_lag_indices = None

        dense_parameter_count = self.channels * self.channels * self.num_weight_banks
        if (not self.one_random_lag_per_pair) and dense_parameter_count > 50_000_000:
            raise ValueError(
                "Dense lagged mixer is too large for this input shape. "
                f"channels={self.channels}, num_lags={self.K} would allocate "
                f"{dense_parameter_count:,} lag weights. "
                "Reduce patch_size or lags, or enable lagged_one_random_lag_per_pair."
            )

        self.weight_banks = nn.Parameter(
            torch.empty(self.num_weight_banks, self.channels, self.channels)
        )
        self.bias = nn.Parameter(torch.empty(self.channels))
        self.register_buffer("inference_lag_mask", torch.ones(self.K), persistent=False)
        self.use_inference_lag_mask = False

        for weight_bank in self.weight_banks:
            nn.init.xavier_uniform_(weight_bank)
        nn.init.zeros_(self.bias)

    def _normalized_lag_probabilities(self):
        lag_values = torch.tensor(self.lags, dtype=torch.float32)

        if self.lag_distribution_type == "uniform":
            probs = torch.ones_like(lag_values)
        elif self.lag_distribution_type == "short-biased":
            probs = lag_values.max() - lag_values + 1.0
        elif self.lag_distribution_type == "long-biased":
            probs = lag_values - lag_values.min() + 1.0
        elif self.lag_distribution_type == "center-biased":
            center = lag_values.mean()
            probs = 1.0 / (1.0 + (lag_values - center).abs())
        elif self.lag_distribution_type == "decaying":
            min_lag = lag_values.min()
            probs = torch.exp(-self.lag_decay_strength * (lag_values - min_lag))
        else:
            raise ValueError(
                f"Unsupported lag_distribution_type: {self.lag_distribution_type}. "
                "Expected one of {'uniform', 'short-biased', 'long-biased', "
                "'center-biased', 'decaying'}."
            )

        prob_sum = probs.sum()
        if prob_sum <= 0:
            raise ValueError("Lag sampling probabilities must sum to a positive value.")
        return probs / prob_sum

    def set_inference_lag_mask(self, lag_mask):
        lag_mask = torch.as_tensor(
            lag_mask,
            dtype=self.weight_banks.dtype,
            device=self.weight_banks.device,
        )
        if lag_mask.ndim != 1 or lag_mask.numel() != self.K:
            raise ValueError(f"lag_mask must have shape ({self.K},), got {tuple(lag_mask.shape)}")

        self.inference_lag_mask.copy_(lag_mask)
        self.use_inference_lag_mask = True

    def clear_inference_lag_mask(self):
        self.inference_lag_mask.fill_(1.0)
        self.use_inference_lag_mask = False

    def _row_budget_exceeded(self, estimated_rows):
        return int(estimated_rows) > int(self.max_export_rows)

    def _selection_row_count(self):
        return self.channels * self.channels

    def _weight_row_count(self):
        return self.channels * self.channels * self.K

    def _export_skip_row(self, export_name, estimated_rows):
        return [
            {
                "skipped": 1,
                "export_name": export_name,
                "reason": "export exceeds lagged mixer row budget",
                "estimated_rows": int(estimated_rows),
                "max_rows": int(self.max_export_rows),
                "channels": int(self.channels),
                "num_lags": int(self.K),
            }
        ]

    def _mask_by_lag(self):
        if self._row_budget_exceeded(self._weight_row_count()):
            return None

        if not self.one_random_lag_per_pair:
            return torch.ones(self.channels, self.channels, self.K, dtype=torch.float32)

        mask = torch.zeros(self.channels, self.channels, self.K, dtype=torch.float32)
        selected = self.selected_lag_indices.detach().cpu()
        for bank_idx in range(self.num_weight_banks):
            mask.scatter_(
                2,
                selected[bank_idx].unsqueeze(-1),
                torch.ones(self.channels, self.channels, 1, dtype=mask.dtype),
            )
        return mask

    def get_lag_selection_rows(self):
        estimated_rows = self._selection_row_count()
        if self._row_budget_exceeded(estimated_rows):
            return self._export_skip_row("lag_selection", estimated_rows)

        rows = []
        if self.one_random_lag_per_pair:
            selected = self.selected_lag_indices.detach().cpu()
        else:
            dense_indices = list(range(self.K))

        for out_ch in range(self.channels):
            for in_ch in range(self.channels):
                if self.one_random_lag_per_pair:
                    active_lag_indices = sorted(
                        int(idx) for idx in selected[:, out_ch, in_ch].tolist()
                    )
                else:
                    active_lag_indices = dense_indices
                active_lags = [self.lags[k_idx] for k_idx in active_lag_indices]
                rows.append(
                    {
                        "out_channel": out_ch,
                        "in_channel": in_ch,
                        "requested_num_lags_per_connection": int(self.num_lags_per_connection),
                        "num_selected_lags": len(active_lags),
                        "selected_lag_indices": "|".join(str(k_idx) for k_idx in active_lag_indices),
                        "selected_lags": "|".join(str(lag) for lag in active_lags),
                    }
                )

        return rows

    def get_lag_distribution_rows(self):
        counts = torch.zeros(self.K, dtype=torch.int64)
        if self.one_random_lag_per_pair:
            selected = self.selected_lag_indices.detach().cpu()
            for bank_idx in range(self.num_weight_banks):
                counts += torch.bincount(selected[bank_idx].reshape(-1), minlength=self.K)
        else:
            counts += self.channels * self.channels
        total = int(counts.sum().item())

        rows = []
        for k_idx, lag in enumerate(self.lags):
            count = int(counts[k_idx].item())
            rows.append(
                {
                    "lag_index": k_idx,
                    "lag": lag,
                    "count": count,
                    "fraction": (count / total) if total > 0 else 0.0,
                }
            )

        return rows

    def export_lag_selection(self, path):
        _write_rows_to_csv(path, self.get_lag_selection_rows())

    def export_lag_distribution(self, path):
        _write_rows_to_csv(path, self.get_lag_distribution_rows())

    def save_snapshot(self, path, epoch=None, step=None, layer_name=None, store_full=True):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        snapshot = {
            "epoch": epoch,
            "step": step,
            "layer_name": layer_name,
            "lags": list(self.lags),
            "channels": int(self.channels),
            "num_lags": int(self.K),
            "lag_distribution_type": self.lag_distribution_type,
            "lag_decay_strength": float(self.lag_decay_strength),
            "one_random_lag_per_pair": self.one_random_lag_per_pair,
            "num_lags_per_connection": int(self.num_lags_per_connection),
            "distribution_rows": self.get_lag_distribution_rows(),
        }

        selection_rows = self.get_lag_selection_rows()
        if selection_rows and "skipped" in selection_rows[0]:
            snapshot["selection_rows_summary"] = selection_rows
        else:
            snapshot["selection_rows"] = selection_rows

        if store_full:
            total_tensor_elements = int(self.weight_banks.numel())
            if self.selected_lag_indices is not None:
                total_tensor_elements += int(self.selected_lag_indices.numel())

            if total_tensor_elements <= self.max_snapshot_tensor_elements:
                snapshot["weight_banks"] = self.weight_banks.detach().cpu()
                if self.selected_lag_indices is not None:
                    snapshot["selected_lag_indices"] = self.selected_lag_indices.detach().cpu()
                snapshot["bias"] = self.bias.detach().cpu()
            else:
                snapshot["full_tensors_skipped"] = True
                snapshot["full_tensors_reason"] = "lagged mixer tensors exceed snapshot budget"
                snapshot["full_tensor_elements"] = total_tensor_elements

        torch.save(snapshot, path)
        return snapshot

    def export_weights(self, path):
        estimated_rows = self._weight_row_count()
        if self._row_budget_exceeded(estimated_rows):
            _write_rows_to_csv(path, self._export_skip_row("lag_weights", estimated_rows))
            return

        channels = self.channels
        num_lags = self.K
        weight = torch.zeros(channels, channels, num_lags, dtype=self.weight_banks.dtype)
        mask = torch.zeros(channels, channels, num_lags, dtype=torch.int64)

        if self.one_random_lag_per_pair:
            selected = self.selected_lag_indices.detach().cpu()
            compact_weights = self.weight_banks.detach().cpu()
            for bank_idx in range(self.num_weight_banks):
                slot_indices = selected[bank_idx]
                slot_weights = compact_weights[bank_idx]
                for lag_idx in range(num_lags):
                    lag_mask = slot_indices == lag_idx
                    if lag_mask.any():
                        weight[:, :, lag_idx][lag_mask] = slot_weights[lag_mask]
                        mask[:, :, lag_idx][lag_mask] = 1
        else:
            dense_weights = self.weight_banks.detach().cpu()
            for lag_idx in range(num_lags):
                weight[:, :, lag_idx] = dense_weights[lag_idx]
                mask[:, :, lag_idx] = 1

        rows = []
        for out_ch in range(channels):
            for in_ch in range(channels):
                for lag_idx, lag in enumerate(self.lags):
                    rows.append(
                        {
                            "out_channel": out_ch,
                            "in_channel": in_ch,
                            "lag_index": lag_idx,
                            "lag": lag,
                            "selected": int(mask[out_ch, in_ch, lag_idx]),
                            "weight": weight[out_ch, in_ch, lag_idx].item(),
                        }
                    )

        _write_rows_to_csv(path, rows)

    def forward(self, x):
        t_steps, batch_size, channels = x.shape
        if channels != self.channels:
            raise ValueError(
                f"LaggedMixer expected channel size {self.channels}, got {channels}"
            )

        xs = []
        for lag in self.lags:
            if lag == 0:
                xs.append(x)
            else:
                pad = x.new_zeros(lag, batch_size, channels)
                xs.append(torch.cat([pad, x[:-lag]], dim=0))

        out = x.new_zeros(t_steps, batch_size, self.channels)
        inference_mask = None
        if (not self.training) and self.use_inference_lag_mask:
            inference_mask = self.inference_lag_mask.to(
                device=x.device,
                dtype=self.weight_banks.dtype,
            )

        if self.one_random_lag_per_pair:
            selected = self.selected_lag_indices
            for bank_idx in range(self.num_weight_banks):
                slot_indices = selected[bank_idx]
                slot_weights = self.weight_banks[bank_idx]
                if inference_mask is not None:
                    slot_weights = slot_weights * inference_mask[slot_indices]

                for lag_idx in range(self.K):
                    lag_mask = slot_indices == lag_idx
                    if not torch.any(lag_mask):
                        continue
                    masked_weight = slot_weights * lag_mask.to(dtype=slot_weights.dtype)
                    out = out + F.linear(xs[lag_idx], masked_weight, bias=None)
        else:
            for lag_idx in range(self.K):
                weight_bank = self.weight_banks[lag_idx]
                if inference_mask is not None:
                    weight_bank = weight_bank * inference_mask[lag_idx]
                out = out + F.linear(xs[lag_idx], weight_bank, bias=None)

        return out + self.bias.view(1, 1, -1)


class LaggedMLPBlock(nn.Module):
    def __init__(self, channels, dropout=0.0, residual=True, **lagged_kwargs):
        super().__init__()
        self.lagged = LaggedMixer(channels=channels, **lagged_kwargs)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(float(dropout))
        self.residual = bool(residual)

    def forward(self, x):
        residual = x
        x = self.lagged(x.permute(1, 0, 2).contiguous())
        x = x.permute(1, 0, 2).contiguous()
        x = self.activation(x)
        x = self.dropout(x)
        if self.residual:
            x = x + residual
        return x


class LaggedMLPBackbone(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout=0.0, lagged_cfg=None):
        super().__init__()

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_layers = max(1, int(n_layers))
        self.dropout = float(dropout)
        self.lagged_cfg = lagged_cfg

        self.input_proj = nn.Linear(self.input_dim, self.hidden_dim)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

        self.input_activation = nn.GELU()
        self.input_dropout = nn.Dropout(self.dropout)

        block_cfgs = self._resolve_block_lag_configs(lagged_cfg)
        one_random = bool(
            _cfg_get(
                lagged_cfg,
                "lagged_one_random_lag_per_pair",
                _cfg_get(lagged_cfg, "one_random_lag_per_pair", False),
            )
        )
        seed = int(_cfg_get(lagged_cfg, "seed", 0))
        residual = bool(_cfg_get(lagged_cfg, "residual", True))

        self.blocks = nn.ModuleList(
            [
                LaggedMLPBlock(
                    channels=self.hidden_dim,
                    dropout=self.dropout,
                    residual=residual,
                    lags=block_cfg["lags"],
                    one_random_lag_per_pair=one_random,
                    num_lags_per_connection=block_cfg["num_lags_per_connection"],
                    lag_distribution_type=block_cfg["distribution_type"],
                    lag_decay_strength=block_cfg["decay_strength"],
                    seed=seed,
                )
                for block_cfg in block_cfgs
            ]
        )

    def _resolve_block_lag_configs(self, lagged_cfg):
        default_lags = tuple(
            int(lag)
            for lag in _cfg_get(
                lagged_cfg,
                "lagged_lags",
                _cfg_get(lagged_cfg, "lags", (0,)),
            )
        )
        default_distribution_type = str(
            _cfg_get(
                lagged_cfg,
                "lagged_distribution_type",
                _cfg_get(lagged_cfg, "distribution_type", "uniform"),
            )
        )
        default_decay_strength = float(
            _cfg_get(
                lagged_cfg,
                "lagged_decay_strength",
                _cfg_get(lagged_cfg, "decay_strength", 1.0),
            )
        )
        default_num_lags_per_connection = int(
            _cfg_get(
                lagged_cfg,
                "lagged_num_lags_per_connection",
                _cfg_get(lagged_cfg, "num_lags_per_connection", 1),
            )
        )
        per_block_cfgs = _cfg_get(lagged_cfg, "block_lag_configs", None)

        if per_block_cfgs is None:
            return [
                {
                    "lags": default_lags,
                    "distribution_type": default_distribution_type,
                    "decay_strength": default_decay_strength,
                    "num_lags_per_connection": default_num_lags_per_connection,
                }
                for _ in range(self.n_layers)
            ]

        if len(per_block_cfgs) != self.n_layers:
            raise ValueError(
                "block_lag_configs must have one entry per lagged MLP block: "
                f"expected {self.n_layers}, got {len(per_block_cfgs)}"
            )

        resolved = []
        for block_cfg in per_block_cfgs:
            lags = _cfg_get(block_cfg, "lags", default_lags)
            resolved.append(
                {
                    "lags": tuple(int(lag) for lag in lags),
                    "distribution_type": str(
                        _cfg_get(block_cfg, "distribution_type", default_distribution_type)
                    ),
                    "decay_strength": float(
                        _cfg_get(block_cfg, "decay_strength", default_decay_strength)
                    ),
                    "num_lags_per_connection": int(
                        _cfg_get(
                            block_cfg,
                            "num_lags_per_connection",
                            default_num_lags_per_connection,
                        )
                    ),
                }
            )

        return resolved

    def forward(self, x):
        x = self.input_proj(x)
        x = self.input_activation(x)
        x = self.input_dropout(x)

        for block in self.blocks:
            x = block(x)

        return x

    def iter_delay_mixers(self):
        for block_idx, block in enumerate(self.blocks):
            yield {
                "block_idx": block_idx,
                "layer_name": f"mlp_backbone.block_{block_idx:02d}.lagged",
                "mixer": block.lagged,
            }

    def get_delay_mixer(self, block_idx):
        for item in self.iter_delay_mixers():
            if item["block_idx"] == int(block_idx):
                return item["mixer"]
        raise IndexError(f"No lag mixer found for block_idx={block_idx}")

    def get_lag_values(self, block_idx):
        return list(self.get_delay_mixer(block_idx).lags)

    def set_inference_lag_mask(self, block_idx, lag_mask):
        self.get_delay_mixer(block_idx).set_inference_lag_mask(lag_mask)

    def clear_inference_lag_masks(self):
        for item in self.iter_delay_mixers():
            item["mixer"].clear_inference_lag_mask()

    def save_lag_selection_snapshots(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        for item in self.iter_delay_mixers():
            selection_path = output_dir / f"block_{item['block_idx']:02d}_lag_selections.csv"
            distribution_path = output_dir / f"block_{item['block_idx']:02d}_lag_distribution.csv"
            weights_path = output_dir / f"block_{item['block_idx']:02d}_lag_weights.csv"
            snapshot_path = output_dir / f"block_{item['block_idx']:02d}_lag_snapshot.pt"

            item["mixer"].export_lag_selection(selection_path)
            item["mixer"].export_lag_distribution(distribution_path)
            item["mixer"].export_weights(weights_path)
            item["mixer"].save_snapshot(snapshot_path, layer_name=item["layer_name"])

            saved_paths.append(
                {
                    "block_idx": item["block_idx"],
                    "layer_name": item["layer_name"],
                    "selection_csv": str(selection_path),
                    "distribution_csv": str(distribution_path),
                    "weights_csv": str(weights_path),
                    "snapshot_pt": str(snapshot_path),
                }
            )

        return saved_paths
