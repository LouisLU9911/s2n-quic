import os
from pathlib import Path
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm  # Import tqdm for progress visualization

from model import QCCT
from dataset import SlidingWindowDataset

reports_dir_tmpl = "reports_seed_{}"
report_dir_tmpl = "delay_{}_drop_{}"

cwd = os.getcwd()

# Define lists for delay and drop_rate values
delays = ["5ms", "50ms", "100ms", "200ms", "500ms"]
drop_rates = [0.01, 0.05, 0.1, 0.2, 0.3]

paths = [
    (delays[2], drop_rates[0]),
    (delays[3], drop_rates[1]),
    (delays[4], drop_rates[2]),
]


# Parameters
seed = 42
num_workers = 6
n_features = 8
hidden_size = 64
n_heads = 4
n_layers = 4
expand_size = 128
context_size = 32
window_size = context_size
batch_size = 128
label_column = "congestion_window"
num_epochs = 10


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU

    np.random.seed(seed)
    random.seed(seed)

    # Ensure deterministic behavior in PyTorch (may affect performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(save=False):

    # Assuming you have a model defined as 'model'
    model = QCCT(
        n_features=n_features,
        hidden_size=hidden_size,
        n_heads=n_heads,
        n_layers=n_layers,
        expand_size=expand_size,
        context_size=context_size,
    )

    # Check if a GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # Move model to GPU
    model.to(device)

    # Define the criterion (loss function)
    criterion = nn.MSELoss()

    # Define the optimizer (e.g., Adam)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        # train
        model.train()
        running_loss = 0.0
        data_loader_len = 0

        # Define the path to the reports directory
        reports_dir = Path(cwd) / reports_dir_tmpl.format(42)
        report_dirs = [reports_dir / report_dir_tmpl.format(*path) for path in paths]
        dataset_paths = [report_dir / "formatted.csv" for report_dir in report_dirs]

        print("Train...")
        for path, dataset_path in zip(paths, dataset_paths):
            df = pd.read_csv(dataset_path)
            # Create the dataset and data loader
            dataset = SlidingWindowDataset(df, window_size, label_column)
            data_loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
            )
            for features, label in tqdm(
                data_loader,
                desc=f"Epoch[{epoch+1}/{num_epochs}] Path[delay:{path[0]},drop_rate:{path[1]}]",
                unit="batch",
            ):
                features, label = features.to(device), label.to(device)
                # print(features.shape, label.shape)

                optimizer.zero_grad()
                outputs = model(features)
                # print(outputs.shape)
                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            data_loader_len += len(data_loader)
        avg_loss = running_loss / data_loader_len
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}")

        # validation
        print("Eval...")
        model.eval()
        running_loss = 0.0
        data_loader_len = 0

        reports_dir = Path(cwd) / reports_dir_tmpl.format(2024)
        report_dirs = [reports_dir / report_dir_tmpl.format(*path) for path in paths]
        dataset_paths = [report_dir / "formatted.csv" for report_dir in report_dirs]
        for dataset_path in dataset_paths:
            df = pd.read_csv(dataset_path)
            # Create the dataset and data loader
            dataset = SlidingWindowDataset(df, window_size, label_column)
            data_loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
            )
            for features, label in tqdm(
                data_loader,
                desc=f"Epoch[{epoch+1}/{num_epochs}] Path[delay:{path[0]}, drop_rate:{path[1]}]",
                unit="batch",
            ):
                features, label = features.to(device), label.to(device)
                # print(features.shape, label.shape)
                outputs = model(features)
                # print(outputs.shape)
                loss = criterion(outputs, label)
                running_loss += loss.item()
            data_loader_len += len(data_loader)
        avg_loss = running_loss / data_loader_len
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_loss:.4f}")

    if save:
        model.eval()
        example = torch.rand(1, context_size, 8).to(device)
        traced_script_module = torch.jit.trace(model, example)
        traced_script_module.save("model.pt")


def main():
    # Set the seed
    set_seed(seed)
    train(save=True)


if __name__ == "__main__":
    main()
