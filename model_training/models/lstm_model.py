import torch
from torch import nn

from models.phoneme_transition_bias import build_phoneme_transition_bias


class LSTMDecoder(nn.Module):
    """
    LSTM decoder with the same day-specific input pathway used by the GRU
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
        self.n_units = n_units
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.n_days = n_days

        self.rnn_dropout = rnn_dropout
        self.input_dropout = input_dropout

        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.phoneme_transition_bias = build_phoneme_transition_bias(
            n_classes=self.n_classes,
            cfg=phoneme_transition_bias_cfg,
        )

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

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.n_units,
            num_layers=self.n_layers,
            dropout=self.rnn_dropout,
            batch_first=True,
            bidirectional=False,
        )

        for name, param in self.lstm.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

        self.out = nn.Linear(self.n_units, self.n_classes)
        nn.init.xavier_uniform_(self.out.weight)

        self.h0 = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(1, 1, self.n_units)))
        self.c0 = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(1, 1, self.n_units)))

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
        del session_summary

        x = self._apply_day_specific_input(x, day_idx)
        x = self._patch_inputs(x)

        if states is None:
            h0 = self.h0.expand(self.n_layers, x.shape[0], self.n_units).contiguous()
            c0 = self.c0.expand(self.n_layers, x.shape[0], self.n_units).contiguous()
            states = (h0, c0)

        output, hidden_states = self.lstm(x, states)

        logits = self.out(output)
        if self.phoneme_transition_bias is not None:
            logits = self.phoneme_transition_bias(logits)

        if return_state:
            return logits, hidden_states

        return logits
