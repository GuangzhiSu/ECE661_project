import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import torch
import torch.nn.functional as F
import time
from models import MambaStock
import time
from dataclasses import dataclass
from models import Model
import argparse
from data import Dataset


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
def evaluation_metric(y_test, y_hat):
    MSE = mean_squared_error(y_test, y_hat)
    RMSE = MSE**0.5
    MAE = mean_absolute_error(y_test, y_hat)
    R2 = r2_score(y_test, y_hat)
    print("    MSE      RMSE     MAE      R2")
    print("%.5f %.5f %.5f %.5f" % (MSE, RMSE, MAE, R2), flush=True)
    return (MSE, RMSE, MAE, R2)


# Unreproducible randomness is overrated
def set_seed(seed, cuda=False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)


# Create the output plot of for the predcitions
def make_a_plot(ds, preds, path, title):
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
):
    opt = torch.optim.Adam(
        model.parameters(),
        lr=run_config.learning_rate,
        weight_decay=run_config.weight_decay,
    )
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
        z = model(xt).squeeze()
        loss = F.mse_loss(z, yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if v == 2 and e % 10 == 0 and e != 0:
            print("Epoch %d | Lossp: %.4f" % (e, loss.item()), flush=True)
    if v > 0:
        print(
            f"Training Complete, duration: %.3f" % (time.time() - start_time),
            flush=True,
        )
    return model


# Infer some thingsss
def inference(model, testX, data, v=1, t=False, cuda=False):
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
def read_file(filename):
    try:
        with open(filename, "r") as file:
            return [line.strip() for line in file]
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return []
    except IOError as e:
        print(f"Error reading file '{filename}': {e}")
        return []


# This one is self explanatory
def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)
    print(f"Model saved to {file_path}")
