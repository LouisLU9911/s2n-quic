#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import json
from jinja2 import Environment, FileSystemLoader
from pathlib import Path
from tqdm import tqdm
import argparse
import os
import pandas as pd

# Parse command line arguments
parser = argparse.ArgumentParser(description="Run batch processes with multiple seeds.")
parser.add_argument('--seeds', type=str, required=True, help="Comma-separated list of seeds.")
args = parser.parse_args()

# Convert the comma-separated seeds into a list of integers
seeds = [int(seed) for seed in args.seeds.split(',')]

# Define the directory and template name
template_dir = Path(__file__).parent / "plans"
template_name = "plan.tmpl"

# Create a Jinja2 environment
env = Environment(loader=FileSystemLoader(template_dir))

# Load the template
template = env.get_template(template_name)

# Define lists for delay and drop_rate values
delays = ["5ms", "50ms", "100ms", "200ms", "500ms"]
drop_rates = [0.01, 0.05, 0.1, 0.2, 0.3]

paths = [
    (delays[2], drop_rates[0]),
    (delays[3], drop_rates[1]),
    (delays[4], drop_rates[2]),
]

# Define fixed values for other parameters
max_inflight = 1000
connections = 1
iterations = 1
streams = 5
stream_data = 500000000

columns = [
    "event",
    "timestamp",
    "lost_bytes",
    "bytes_acknowledged",
    "bytes_in_flight",
    "congestion_window",
]

CWD = os.getcwd()
reports_dir_tmpl = "reports_seed_{}"
report_dir_tmpl = "delay_{}_drop_{}"

def get_all_lines_from_report_dir(report_dir) -> list:
    filtered_lines = []
    if report_dir.is_dir():
        stderr_path = report_dir / "stderr.log"
        print(f"Reading {stderr_path}...")
        if stderr_path.exists():
            # Read the stderr.log file
            with open(stderr_path, "r", encoding="utf-8") as file:
                lines = file.read().split("\n")
            # Keep only lines that start with "event:"
            filtered_lines = [line for line in lines if line.startswith("event:")]
    return filtered_lines

def filter_out(lines: list, event: str) -> list:
    filtered_lines = [line for line in lines if event not in line]
    return filtered_lines

def format_line(line):
    if "on_packet_sent" in line:
        cols = line.split(",")
        record = {
            columns[0]: cols[0].split(":")[1],
            columns[1]: cols[1].split(":", 1)[1],  # time_sent
            columns[2]: 0,
            columns[3]: 0,
            columns[4]: cols[3].split(":")[1],
            columns[5]: cols[4].split(":")[1],
        }
    elif "on_packet_lost" in line:
        cols = line.split(",")
        record = {
            columns[0]: cols[0].split(":")[1],
            columns[1]: cols[1].split(":", 1)[1],  # timestamp
            columns[2]: cols[3].split(":")[1],  # lost_bytes
            columns[3]: 0,
            columns[4]: cols[6].split(":")[1],
            columns[5]: cols[7].split(":")[1],
        }
    elif "on_ack" in line:
        cols = line.split(",")
        record = {
            columns[0]: cols[0].split(":")[1],
            columns[1]: cols[4].split(":", 1)[1],  # ack_receive_time
            columns[2]: 0,
            columns[3]: cols[3].split(":")[1],  # bytes_acknowledged
            columns[4]: cols[5].split(":")[1],
            columns[5]: cols[6].split(":")[1],
        }
    else:
        raise Exception(f"Unsupported line: {line}")
    return record

def ts2ms(timestamp: str) -> int:
    h, m, s = timestamp.split(":")
    m = int(m) + int(h) * 60
    ms = round(float(s) * 1000) + m * 60 * 1000
    return ms

def format_stderr(seed, delay, drop_rate, save=False):
    # Define the path to the reports directory
    reports_dir = Path(CWD) / reports_dir_tmpl.format(seed)
    report_dir = reports_dir / report_dir_tmpl.format(delay, drop_rate)

    filtered_lines = get_all_lines_from_report_dir(report_dir)
    filtered_lines = filter_out(filtered_lines, "on_rtt_update")
    records = list(map(format_line, filtered_lines))
    df = pd.DataFrame.from_records(records)
    df["timestamp"] = df["timestamp"].apply(ts2ms)
    df_one_hot = pd.get_dummies(df, columns=["event"], dtype=int)
    # Specify the column to move
    col_to_move = "congestion_window"
    # Move the column to the end
    df_one_hot = df_one_hot[
        [col for col in df_one_hot if col != col_to_move] + [col_to_move]
    ]
    if save:
        df_one_hot.to_csv(report_dir / "formatted.csv", index=False)
    return df_one_hot

for seed in seeds:
    # Create a directory for reports if it doesn't exist
    reports_dir = Path(__file__).parent / f"reports_seed_{seed}"
    reports_dir.mkdir(exist_ok=True)

    # Iterate over each combination of delay and drop_rate
    for delay, drop_rate in tqdm(paths, desc=f"Processing seed {seed}", dynamic_ncols=True):
        # Update tqdm description with the current seed, delay, and drop_rate
        tqdm.write(f"Seed: {seed}, Delay: {delay}, Drop Rate: {drop_rate}")
        # Define context for each combination
        context = {
            "max_inflight": max_inflight,
            "connections": connections,
            "iterations": iterations,
            "streams": streams,
            "stream_data": stream_data,
            "delay": delay,
            "drop_rate": drop_rate,
            "seed": seed,
        }

        # Create a subdirectory for each combination
        subdirectory_name = f"delay_{delay}_drop_{drop_rate}"
        subdirectory_path = reports_dir / subdirectory_name
        subdirectory_path.mkdir(exist_ok=True)

        # Render the template with the current context
        output = template.render(context)

        # Define the output filenames for each combination
        output_filename = "plan.toml"
        output_path = subdirectory_path / output_filename

        # Save the rendered output to a file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output)

        # Define a unique stderr filename for each process
        stderr_filename = "stderr.log"
        stderr_path = subdirectory_path / stderr_filename

        # Save the context to a JSON file
        context_filename = "context.json"
        context_path = subdirectory_path / context_filename
        with open(context_path, "w", encoding="utf-8") as json_file:
            json.dump(context, json_file, indent=4)

        # Run the command and redirect stderr to the unique log file
        with open(stderr_path, "w", encoding="utf-8") as stderr_file:
            # Use tqdm.write to print messages to avoid interfering with the progress bar
            tqdm.write(f"Running process for: {output_path}")

            process = subprocess.Popen(
                ["cargo", "run", "--release", "--", "batch", str(output_path)],
                stderr=stderr_file,
            )

            # Wait for the process to complete
            process.wait()

            # Print a message indicating that the process has finished
            tqdm.write(
                f"Process finished for: {output_path} with stderr logged in {stderr_filename}"
            )
        format_stderr(seed, delay, drop_rate, save=True)
