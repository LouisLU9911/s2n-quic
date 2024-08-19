import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class SlidingWindowDataset(Dataset):
    def __init__(self, df, window_size, label_column):
        self.df = df
        self.window_size = window_size
        self.label_column = label_column

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Determine the actual window size based on the index
        actual_window_size = min(idx + 1, self.window_size)

        # Slice the DataFrame to get the window
        start_idx = idx - actual_window_size + 1
        window = self.df.iloc[start_idx : idx + 1]

        # The label is taken from the last row of the window for the label_column
        label = window.iloc[-1][self.label_column]

        # Extract the features (all values except the last row's label_column, i.e., the label)
        features = window.values
        features[-1, -1] = 0

        # Pad the features with zeros if necessary
        if actual_window_size < self.window_size:
            padding = np.zeros(
                (self.window_size - actual_window_size, features.shape[1])
            )
            features = np.vstack((padding, features))

        return torch.tensor(features, dtype=torch.float32), torch.tensor(
            label, dtype=torch.float32
        ).unsqueeze(-1)
