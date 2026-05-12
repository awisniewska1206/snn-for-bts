import torch
import torch.nn.functional as F
from torch import nn

from models.phoneme_transition_bias import build_phoneme_transition_bias


class CausalTCNBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.0):
        super().__init__()

        if kernel_size < 2:
            raise ValueError("CausalTCNBlock requires kernel_size >= 2")

        self.pad = int((kernel_size - 1) * dilation)
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        residual = x

        x = x.transpose(1, 2).contiguous()
        x = F.pad(x, (self.pad, 0))
        x = self.conv(x)
        x = x.transpose(1, 2).contiguous()
        x = self.activation(x)
        x = self.dropout(x)

        return x + residual


class TCNDecoder(nn.Module):
    """
    Causal temporal convolution decoder with the same day-specific input path
    and trainer-facing interface as the recurrent baselines.
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
        tcn_kernel_size=3,
        tcn_dilation_base=2,
        rnn_dropout=None,
    ):
        super().__init__()

        if rnn_dropout is not None:
            dropout = rnn_dropout

        self.neural_dim = neural_dim
        self.hidden_size = int(hidden_size)
        self.n_classes = n_classes
        self.n_layers = max(1, int(n_layers))
        self.n_days = n_days
        self.dropout = float(dropout)
        self.rnn_dropout = self.dropout
        self.input_dropout = input_dropout
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.tcn_kernel_size = int(tcn_kernel_size)
        self.tcn_dilation_base = int(tcn_dilation_base)

        if self.hidden_size <= 0:
            raise ValueError("TCNDecoder requires hidden_size > 0")
        if self.tcn_dilation_base < 1:
            raise ValueError("TCNDecoder requires tcn_dilation_base >= 1")

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

        self.input_projection = nn.Linear(self.input_size, self.hidden_size)
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)

        self.input_activation = nn.GELU()
        self.input_projection_dropout = nn.Dropout(self.dropout)

        self.tcn_blocks = nn.ModuleList(
            [
                CausalTCNBlock(
                    channels=self.hidden_size,
                    kernel_size=self.tcn_kernel_size,
                    dilation=self.tcn_dilation_base ** block_idx,
                    dropout=self.dropout,
                )
                for block_idx in range(self.n_layers)
            ]
        )

        self.out = nn.Linear(self.hidden_size, self.n_classes)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

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

        x = self.input_projection(x)
        x = self.input_activation(x)
        if self.dropout > 0:
            x = self.input_projection_dropout(x)

        for block in self.tcn_blocks:
            x = block(x)

        logits = self.out(x)
        if self.phoneme_transition_bias is not None:
            logits = self.phoneme_transition_bias(logits)

        if return_state:
            return logits, None

        return logits
