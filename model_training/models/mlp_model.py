import torch
from torch import nn

from models.phoneme_transition_bias import build_phoneme_transition_bias


class MLPDecoder(nn.Module):
    """
    Feed-forward decoder with the same day-specific input adaptation used by the GRU
    baseline so it can be swapped into the trainer with minimal changes.
    """

    def __init__(
        self,
        neural_dim,
        n_units,
        n_days,
        n_classes,
        rnn_dropout=0.0,
        input_dropout=0.0,
        n_layers=5,
        patch_size=0,
        patch_stride=0,
        phoneme_transition_bias_cfg=None,
    ):
        super().__init__()

        self.neural_dim = neural_dim
        self.n_classes = n_classes
        self.n_days = n_days
        self.input_dropout = input_dropout
        self.hidden_dropout = rnn_dropout
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.n_layers = max(1, int(n_layers))
        self.hidden_dim = int(n_units)
        if self.hidden_dim <= 0:
            raise ValueError("MLPDecoder requires n_units > 0 for the hidden width")

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
        self.mlp = nn.ModuleList()
        in_features = self.input_size
        for _ in range(self.n_layers):
            linear = nn.Linear(in_features, self.hidden_dim)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            self.mlp.append(linear)
            in_features = self.hidden_dim

        self.hidden_dropout_layer = nn.Dropout(self.hidden_dropout)
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
