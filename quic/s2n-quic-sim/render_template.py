#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import json
from jinja2 import Environment, FileSystemLoader
from pathlib import Path
from tqdm import tqdm
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description="Run batch processes with multiple seeds.")
parser.add_argument(
    "--seeds", type=str, required=True, help="Comma-separated list of seeds."
)
args = parser.parse_args()

# Convert the comma-separated seeds into a list of integers
seeds = [int(seed) for seed in args.seeds.split(",")]

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

# Loop through each seed
for seed in seeds:
    # Create a directory for reports if it doesn't exist
    reports_dir = Path(__file__).parent / f"reports_seed_{seed}"
    reports_dir.mkdir(exist_ok=True)

    # Iterate over each combination of delay and drop_rate
    for delay, drop_rate in tqdm(paths, desc="path"):
        # Define context for each combination
        context = {
            "max_inflight": max_inflight,
            "connections": connections,
            "iterations": iterations,
            "streams": streams,
            "stream_data": stream_data,
            "delay": delay,
            "drop_rate": drop_rate,
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
