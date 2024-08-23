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
num_workers = 8
n_features = 8
hidden_size = 128
n_heads = 4
n_layers = 4
expand_size = 256
context_size = 64
window_size = context_size
batch_size = 128
label_column = "congestion_window"
num_epochs = 30


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


def get_data_loader(dataset_path):
    df = pd.read_csv(dataset_path)
    # Create the dataset and data loader
    dataset = SlidingWindowDataset(df, window_size, label_column)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    return data_loader


def run(model, device, optimizer, criterion, action: str, epoch: int, seeds: list):
    if action == "train":
        model.train()
    else:
        model.eval()
    running_loss = 0.0
    data_loader_len = 0

    for seed in seeds:
        print(f"Running {action}[seed:{seed}]...")
        reports_dir = Path(cwd) / reports_dir_tmpl.format(seed)
        report_dirs = [reports_dir / report_dir_tmpl.format(*path) for path in paths]
        dataset_paths = [report_dir / "formatted.csv" for report_dir in report_dirs]

        for path, dataset_path in zip(paths, dataset_paths):
            data_loader = get_data_loader(dataset_path)
            data_loader_len += len(data_loader)
            for features, label in tqdm(
                data_loader,
                desc=f"Epoch[{epoch+1}/{num_epochs}] Path[seed:{seed},delay:{path[0]},drop_rate:{path[1]}]",
                unit="batch",
            ):
                features, label = features.to(device), label.to(device)
                if action == "train":
                    optimizer.zero_grad()
                    outputs = model(features)
                    loss = criterion(outputs, label)
                    loss.backward()
                    optimizer.step()
                else:
                    with torch.no_grad():
                        outputs = model(features)
                        loss = criterion(outputs, label)
                running_loss += loss.item()
    avg_loss = running_loss / data_loader_len
    print(f"Epoch [{epoch+1}/{num_epochs}], {action} Loss: {avg_loss:.4f}")


def train_and_test(save=False):
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
        run(model, device, optimizer, criterion, "train", epoch, seeds=[42, 2024, 2023])
        # validation
        run(model, device, optimizer, criterion, "val", epoch, seeds=[10086])
        # test

    if save:
        model.eval()
        example = torch.rand(1, context_size, 8).to(device)
        traced_script_module = torch.jit.trace(model, example)
        traced_script_module.save("model.pt")


def main():
    # Set the seed
    set_seed(seed)
    train_and_test(save=True)


if __name__ == "__main__":
    main()
