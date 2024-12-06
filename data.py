import numpy as np
import pandas as pd
import yfinance as yf

# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from datetime import datetime
from typing import Tuple


# Parent Dataset class in case we want to experiment with different datasets
class Dataset:
    def __init__(self, inSize: int, outSize: int):
        self.inSize = inSize  # Must match model's input layer
        self.outSize = (
            outSize  # Should always be 1 unless doing some whacky experimentation
        )
        # self.sc = StandardScaler()
        # self.pca = PCA(n_components=inSize)

    def getData(self):
        raise NotImplementedError

    def formatPred(self):
        raise NotImplementedError


# Dataset created by getting historical stock data with yfinance library
class yahooFinance(Dataset):
    def __init__(self, run_config):
        super().__init__(run_config.window_size, 1)
        self.symbols = run_config.symbols  # list of stock sysmbols
        self.start_date = (
            run_config.start_date
        )  # earliest date to start getting data from
        self.train_split = (
            run_config.train_split
        )  # how much of the data is for training (0, 1)

    # Create train and test sets
    def getData(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Get the data for each stock symbol in the list
        for j, symbol in enumerate(self.symbols):
            df = yf.Ticker(symbol).history(start=self.start_date, end=datetime.now())
            print(
                f"Loaded data on {symbol} from {self.start_date} until {str(datetime.now())[:10]}"
            )
            data = (
                df.filter(["Close"])
                if j == 0
                else pd.concat([data, df.filter(["Close"])])
            )
        self.data = data
        dataset = data.values

        # How many training instances do we have?
        training_data_len = int(np.ceil(len(dataset) * self.train_split))
        self.training_data_len = training_data_len

        # Scale the the data to (0, 1) and save the scalar for test data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)
        self.scaler = scaler

        # Declare training set
        x_train = []
        y_train = []
        train_data = scaled_data[0 : int(training_data_len), :]

        # For each day in the training set add the previous n days (inSize) as the inputs
        for i in range(self.inSize, len(train_data)):
            x_train.append(train_data[i - self.inSize : i, 0])
            y_train.append(train_data[i, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        # Same as above but for test data
        x_test = []
        y_test = dataset[training_data_len:, :]
        test_data = scaled_data[training_data_len - self.inSize :, :]

        for i in range(self.inSize, len(test_data)):
            x_test.append(test_data[i - self.inSize : i, 0])
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        return x_train.squeeze(), y_train.squeeze(), x_test.squeeze(), y_test.squeeze()

    # Utility function for cleaning up a model's predictions
    def formatPred(self, preds):
        preds = preds[:, None]
        return self.scaler.inverse_transform(preds).squeeze()
