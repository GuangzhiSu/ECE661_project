import torch
import torch.nn.functional as F
import numpy as np

import argparse
import time
from typing import Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from data import Dataset
from models import MambaStock, Transformer, LSTM
from models import Model
from dp_optimizer import DPAdamGaussianOptimizer  


# A dataclass that specifies the parameters of the current run
@dataclass
class runConfig:
    # Data
    symbols: list[str]
    start_date: str
    train_split: float

    # Model
    architecture: Model
    window_size: int
    hidden_dim: int
    layers: int

    # Training
    epochs: int
    learning_rate: float
    weight_decay: float
    cuda: bool


# Hard coded config for an example run
def config() -> runConfig:
    # Data
    file_path = "data/symbols.txt"
    symbols = read_file(file_path)
    start_date = "2010-01-01"
    train_split = 0.8

    # Model
    architecture = LSTM
    # architecture = Transformer
    # architecture = MambaStock
    window_size = 20
    hidden_dim = 16
    layers = 2

    # Training
    epochs = 80
    learning_rate = 0.01
    weight_decay = 1e-5
    cuda = False

    return runConfig(
        symbols,
        start_date,
        train_split,
        architecture,
        window_size,
        hidden_dim,
        layers,
        epochs,
        learning_rate,
        weight_decay,
        cuda,
    )


# Hard coded config for a single stock to graph predcitions of
def graph_config(config) -> runConfig:
    # Data
    file_path = "data/graph.txt"
    symbols = read_file(file_path)
    start_date = "2010-01-01"
    train_split = 0.5

    return runConfig(
        symbols,
        start_date,
        train_split,
        config.architecture,
        config.window_size,
        config.hidden_dim,
        config.layers,
        config.epochs,
        config.learning_rate,
        config.weight_decay,
        config.cuda,
    )


# Function for creating a runConfig from command line arguments
def parse_args() -> runConfig:
    input_parser = argparse.ArgumentParser()
    input_parser.add_argument(
        "--cuda",
        default=True,
        type=bool,
        help="Boolean of if Cuda should be used or not (Cuda must be available)",
    )
    input_parser.add_argument(
        "--symbol", default="GOOG", type=str, help="Stock symbol of target company"
    )
    input_parser.add_argument(
        "--start_date", default="2012-01-01", type=str, help="Start date of dataset"
    )
    input_parser.add_argument(
        "--train_split",
        default=0.8,
        type=float,
        help="Percentage of data reserved for training",
    )
    input_parser.add_argument(
        "--architecture",
        default=MambaStock,
        type=Model,
        help="Target model architecture",
    )
    input_parser.add_argument(
        "--window_size",
        default=20,
        type=int,
        help="Amound of days for the sliding window (Industry norms are 20 and 52)",
    )
    input_parser.add_argument(
        "--hidden_dim",
        default=16,
        type=int,
        help="Hidden dimension size of specialty layers",
    )
    input_parser.add_argument(
        "--layers", default=2, type=int, help="Number of specialty layers"
    )
    input_parser.add_argument(
        "--epochs", default=100, type=int, help="Number of training epochs"
    )
    input_parser.add_argument(
        "--learning_rate", default=0.01, type=float, help="Training learning rate"
    )
    input_parser.add_argument(
        "--weight_decay", default=1e-4, type=float, help="Training weight decay"
    )
    args = input_parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    print(f"Using Cuda: {args.cuda}")
    run_config = runConfig(
        args.symbol,
        args.start_date,
        args.train_split,
        args.architecture,
        args.window_size,
        args.hidden_dim,
        args.layers,
        args.epochs,
        args.learning_rate,
        args.weight_decay,
        args.cuda,
    )
    return run_config


# Evaluate the results of predictions compared to the labels
def evaluation_metric(
    y_test: np.ndarray, y_hat: np.ndarray
) -> Tuple[float, float, float, float]:
    MSE = mean_squared_error(y_test, y_hat)
    RMSE = MSE**0.5
    MAE = mean_absolute_error(y_test, y_hat)
    R2 = r2_score(y_test, y_hat)
    print("    MSE      RMSE     MAE      R2")
    print("%.5f %.5f %.5f %.5f" % (MSE, RMSE, MAE, R2), flush=True)
    return (MSE, RMSE, MAE, R2)


# Unreproducible randomness is overrated
def set_seed(seed: int, cuda: bool = False) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)


# Create the output plot of for the predcitions
def make_a_plot(ds: Dataset, preds: np.ndarray, path: str, title: str) -> None:
    train = ds.data[: ds.training_data_len]
    valid = ds.data[ds.training_data_len :]
    valid["Predictions"] = preds
    plt.figure(figsize=(16, 6))
    plt.title(title)
    plt.xlabel("Date", fontsize=18)
    plt.ylabel("Close Price USD ($)", fontsize=18)
    plt.plot(train["Close"])
    plt.plot(valid[["Close", "Predictions"]])
    plt.legend(["Train", "Reality", "Predictions"], loc="upper left")
    plt.tight_layout()
    plt.savefig(path)
    print(f"Plot saved to {path}", flush=True)


# Training step
def train(
    model: Model,
    trainX: np.ndarray,
    trainY: np.ndarray,
    run_config: runConfig,
    v: int = 1,
) -> Model:
    
    #normal optimizer
    opt = torch.optim.Adam(
        model.parameters(),
        lr=run_config.learning_rate,
        weight_decay=run_config.weight_decay,
    )
    
    # # Initialize the DP optimizer
    # opt = DPAdamGaussianOptimizer(
    #     model.parameters(),
    #     l2_norm_clip=2.0,                  # Gradient clipping threshold
    #     noise_multiplier=0.5,             # Noise multiplier for DP
    #     num_microbatches=16,              # Number of microbatches
    #     lr=run_config.learning_rate       # Learning rate
    # )
    
    xt = torch.from_numpy(trainX).float()
    yt = torch.from_numpy(trainY).float()

    # if t:
    #     xt = torch.transpose(xt, 0, 1)

    if run_config.cuda:
        model = model.cuda()
        xt = xt.cuda()
        yt = yt.cuda()

    if v > 0:
        print(f"Training for {run_config.epochs} epochs", flush=True)

    start_time = time.time()
    for e in range(run_config.epochs):
        model.train()

        #normal training process
        z = model(xt).squeeze()
        loss = F.mse_loss(z, yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if v == 2 and e % 10 == 0 and e != 0:
            print("Epoch %d | Lossp: %.4f" % (e, loss.item()), flush=True)

        # # DP optimizer training process

        # # Compute DP gradients
        # final_grads = opt.compute_grads(F.mse_loss, model, xt, yt)

        # # Apply sanitized gradients
        # opt.apply_grads(model, final_grads)

        # # Optionally log progress
        # if v == 2 and e % 10 == 0 and e != 0:
        #     with torch.no_grad():
        #         preds = model(xt).squeeze()
        #         loss = F.mse_loss(preds, yt)
        #         print(f"Epoch {e} | Loss: {loss.item():.4f}", flush=True)
    if v > 0:
        print(
            f"Training Complete, duration: %.3f" % (time.time() - start_time),
            flush=True,
        )
    return model


# Infer some thingsss
def inference(
    model: Model,
    testX: np.ndarray,
    data: Dataset,
    v: int = 1,
    t: bool = False,
    cuda: bool = False,
) -> np.ndarray:
    xv = torch.from_numpy(testX.squeeze()).float()

    # if t:
    #     xv = torch.transpose(xv, 0, 1)

    if cuda:
        model = model.cuda()
        xv = xv.cuda()
    if v > 0:
        print(
            f"Starting inference with dataset of {max(xv.shape)} instances", flush=True
        )
    start_time = time.time()
    mat = model(xv)
    if v > 0:
        print(
            f"Inference Complete, duration: %.9f" % (time.time() - start_time),
            flush=True,
        )

    mat = mat.cpu()
    return data.formatPred(mat.detach().numpy().flatten())


# Read a list (of hopefully stock symbols) from a file
def read_file(filename: str) -> list[str]:
    out = []
    try:
        with open(filename, "r") as file:
            for line in file:
                if line[0] != "#":  # Avoid comment lines
                    out.append(line.strip())
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
    except IOError as e:
        print(f"Error reading file '{filename}': {e}")
    return out


# This one is self explanatory
def save_model(model: Model, file_path: str) -> None:
    torch.save(model.state_dict(), file_path)
    print(f"Model saved to {file_path}")
