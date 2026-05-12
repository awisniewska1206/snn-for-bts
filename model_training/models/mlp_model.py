from pathlib import Path

import torch
from torch import nn

from models.mlp_backbone import LaggedMixer
from models.phoneme_transition_bias import build_phoneme_transition_bias


class MLPDecoder(nn.Module):
    """
    Feed-forward decoder with the same day-specific input adaptation used by the GRU
    baseline so it can be swapped into the trainer with minimal changes.
    """

    def __init__(
        self,
        neural_dim,
        hidden_size,
        n_days,
        n_classes,
        dropout=0.0,
        input_dropout=0.0,
        n_layers=5,
        patch_size=0,
        patch_stride=0,
        phoneme_transition_bias_cfg=None,
        lagged_mlp_cfg=None,
        rnn_dropout=None,
    ):
        super().__init__()

        if rnn_dropout is not None:
            dropout = rnn_dropout

        self.neural_dim = neural_dim
        self.n_classes = n_classes
        self.n_days = n_days
        self.input_dropout = input_dropout
        self.dropout = float(dropout)
        self.hidden_dropout = self.dropout
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.n_layers = max(1, int(n_layers))
        self.hidden_dim = int(hidden_size)
        self.lagged_mlp_cfg = lagged_mlp_cfg
        if self.hidden_dim <= 0:
            raise ValueError("MLPDecoder requires hidden_size > 0")

        self.day_layer_activation = nn.Softsign()
        self.day_weights = nn.ParameterList(
            [nn.Parameter(torch.eye(self.neural_dim)) for _ in range(self.n_days)]
        )
        self.day_biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.neural_dim)) for _ in range(self.n_days)]
        )
        self.day_layer_dropout = nn.Dropout(input_dropout)

        self.input_size = self.neural_dim
        if self.patch_size > 0:
            self.input_size *= self.patch_size

        self.activation = nn.GELU()
        self.hidden_dropout_layer = nn.Dropout(self.hidden_dropout)
        self.use_lagged_mixer = bool(
            lagged_mlp_cfg is not None and lagged_mlp_cfg.get("enabled", False)
        )

        self.lagged_input_mixer = None
        if self.use_lagged_mixer:
            self.lagged_input_mixer = LaggedMixer(
                channels=self.input_size,
                lags=tuple(int(lag) for lag in lagged_mlp_cfg.get("lagged_lags", lagged_mlp_cfg.get("lags", [0]))),
                one_random_lag_per_pair=bool(
                    lagged_mlp_cfg.get(
                        "lagged_one_random_lag_per_pair",
                        lagged_mlp_cfg.get("one_random_lag_per_pair", False),
                    )
                ),
                num_lags_per_connection=int(
                    lagged_mlp_cfg.get(
                        "lagged_num_lags_per_connection",
                        lagged_mlp_cfg.get("num_lags_per_connection", 1),
                    )
                ),
                lag_distribution_type=str(
                    lagged_mlp_cfg.get(
                        "lagged_distribution_type",
                        lagged_mlp_cfg.get("distribution_type", "uniform"),
                    )
                ),
                lag_decay_strength=float(
                    lagged_mlp_cfg.get(
                        "lagged_decay_strength",
                        lagged_mlp_cfg.get("decay_strength", 1.0),
                    )
                ),
                seed=int(lagged_mlp_cfg.get("seed", 0)),
            )

        self.mlp = nn.ModuleList()
        in_features = self.input_size
        for _ in range(self.n_layers):
            linear = nn.Linear(in_features, self.hidden_dim)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            self.mlp.append(linear)
            in_features = self.hidden_dim

        self.out = nn.Linear(self.hidden_dim, self.n_classes)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

        self.phoneme_transition_bias = build_phoneme_transition_bias(
            n_classes=self.n_classes,
            cfg=phoneme_transition_bias_cfg,
        )

    def _apply_day_specific_input(self, x, day_idx):
        day_weights = torch.stack([self.day_weights[i] for i in day_idx], dim=0)
        day_biases = torch.cat([self.day_biases[i] for i in day_idx], dim=0).unsqueeze(1)

        x = torch.einsum("btd,bdk->btk", x, day_weights) + day_biases
        x = self.day_layer_activation(x)

        if self.input_dropout > 0:
            x = self.day_layer_dropout(x)

        return x

    def _patch_inputs(self, x):
        if self.patch_size <= 0:
            return x

        x = x.unsqueeze(1)
        x = x.permute(0, 3, 1, 2)
        x_unfold = x.unfold(3, self.patch_size, self.patch_stride)
        x_unfold = x_unfold.squeeze(2)
        x_unfold = x_unfold.permute(0, 2, 3, 1)
        return x_unfold.reshape(x.size(0), x_unfold.size(1), -1)

    def forward(
        self,
        x,
        day_idx,
        states=None,
        return_state=False,
        session_summary=None,
    ):
        del states, session_summary

        x = self._apply_day_specific_input(x, day_idx)
        x = self._patch_inputs(x)

        if self.use_lagged_mixer:
            x = self.lagged_input_mixer(x.permute(1, 0, 2).contiguous())
            x = x.permute(1, 0, 2).contiguous()

        for linear in self.mlp:
            x = linear(x)
            x = self.activation(x)
            if self.hidden_dropout > 0:
                x = self.hidden_dropout_layer(x)

        logits = self.out(x)
        if self.phoneme_transition_bias is not None:
            logits = self.phoneme_transition_bias(logits)

        if return_state:
            return logits, None

        return logits

    def save_lag_selection_snapshots(self, output_dir):
        if self.lagged_input_mixer is None:
            return []

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        selection_path = output_dir / "block_00_lag_selections.csv"
        distribution_path = output_dir / "block_00_lag_distribution.csv"
        weights_path = output_dir / "block_00_lag_weights.csv"
        snapshot_path = output_dir / "block_00_lag_snapshot.pt"

        self.lagged_input_mixer.export_lag_selection(selection_path)
        self.lagged_input_mixer.export_lag_distribution(distribution_path)
        self.lagged_input_mixer.export_weights(weights_path)
        self.lagged_input_mixer.save_snapshot(snapshot_path, layer_name="mlp.input_lagged_mixer")

        return [
            {
                "block_idx": 0,
                "layer_name": "mlp.input_lagged_mixer",
                "selection_csv": str(selection_path),
                "distribution_csv": str(distribution_path),
                "weights_csv": str(weights_path),
                "snapshot_pt": str(snapshot_path),
            }
        ]

    def get_lag_values(self, block_idx):
        if self.lagged_input_mixer is None:
            raise AttributeError("MLP backbone does not expose lag values")
        if int(block_idx) != 0:
            raise IndexError(f"No lag mixer found for block_idx={block_idx}")
        return list(self.lagged_input_mixer.lags)

    def set_inference_lag_mask(self, block_idx, lag_mask):
        if self.lagged_input_mixer is None:
            raise AttributeError("MLP lagged mixer does not support inference lag masking")
        if int(block_idx) != 0:
            raise IndexError(f"No lag mixer found for block_idx={block_idx}")
        self.lagged_input_mixer.set_inference_lag_mask(lag_mask)

    def clear_inference_lag_masks(self):
        if self.lagged_input_mixer is None:
            return
        self.lagged_input_mixer.clear_inference_lag_mask()
