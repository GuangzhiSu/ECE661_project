import torch
import torch.nn as nn

from mamba import Mamba, MambaConfig


# Parent class for models to ensure they all have the expected features and structure
class Model(nn.Module):
    def __init__(
        self, hidden_dim: int, n_layers: int, input_size: int, output_size: int
    ):
        super().__init__()
        self.in_size = input_size
        self.fc_in = nn.Linear(input_size, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_size)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor):
        raise NotImplementedError


# Modified MambaStock model for this kind of prediction
class MambaStock(Model):
    def __init__(
        self, hidden_dim: int, n_layers: int, input_size: int, output_size: int
    ):
        super().__init__(hidden_dim, n_layers, input_size, output_size)
        self.config = MambaConfig(d_model=hidden_dim, n_layers=n_layers)
        self.mamba = Mamba(self.config)
        self.name = "Mamba"

        print(
            f"Mamba Stock model created with hidden dim {hidden_dim}, {n_layers} layers, and in/out sizes of {input_size} / {output_size}",
            flush=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc_in(x)
        x = x.unsqueeze(0)
        mamba_out = self.mamba(x)
        x = self.fc_out(mamba_out)
        x = self.tanh(x)
        return x.flatten()


# LSTMs are quickk
class LSTM(Model):
    def __init__(
        self,
        hidden_dim: int,
        n_layers: int,
        input_size: int,
        output_size: int,
        dropout: float = 0.2,
    ):
        super().__init__(hidden_dim, n_layers, input_size, output_size)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, n_layers, dropout=dropout)
        self.name = "LSTM"
        print(
            f"LSTM model created with hidden dim {hidden_dim}, {n_layers} layers, and in/out sizes of {input_size} / {output_size}",
            flush=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc_in(x)
        lstm_out, (hn, cn) = self.lstm(x)
        x = self.fc_out(lstm_out)
        x = self.tanh(x)
        return x.flatten()


# What's not a transformer nowadays?
# NOTE: Transformers are much much more computationally expensive than the other model types
class Transformer(Model):
    def __init__(
        self,
        hidden_dim: int,
        n_layers: int,
        input_size: int,
        output_size: int,
        dropout: float = 0.2,
    ):
        super().__init__(hidden_dim, n_layers, input_size, output_size)
        self.t_former = nn.Transformer(
            d_model=hidden_dim,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
            dropout=dropout,
            dim_feedforward=512,
        )
        self.name = "Transformer"

        print(
            f"Transformer model created with hidden dim {hidden_dim}, {n_layers} layers, and in/out sizes of {input_size} / {output_size}",
            flush=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc_in(x)
        tf_out = self.t_former(x, x)
        x = self.fc_out(tf_out)
        x = self.tanh(x)
        return x.flatten()
