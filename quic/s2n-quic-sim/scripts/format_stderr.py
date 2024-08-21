#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path

import pandas as pd

CWD = os.getcwd()

# Define the path to the reports directory
reports_dir_tmpl = "reports_seed_{}"
report_dir_tmpl = "delay_{}_drop_{}"

seeds = [42, 2023, 2024]
delays = ["5ms", "50ms", "100ms", "200ms", "500ms"]
drop_rates = [0.01, 0.05, 0.1, 0.2, 0.3]
columns = [
    "event",
    "timestamp",
    "lost_bytes",
    "bytes_acknowledged",
    "bytes_in_filght",
    "congestion_window",
]


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
        # event:on_packet_sent,time_sent:0:00:00.200000,under_utilized:true,bytes_in_flight:191,congestion_window:12000
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
        # event:on_packet_lost,timestamp:0:00:02.924665,under_utilized:true,lost_bytes:53,persistent_congestion:false,new_loss_burst:true,bytes_in_flight:1601,congestion_window:9828
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
        # event:on_ack,newest_acked_time_sent:0:00:00.200000,under_utilized:true,bytes_acknowledged:191,ack_receive_time:0:00:00.600000,bytes_in_flight:1009,congestion_window:12000
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

    if report_dir.is_dir():
        print(report_dir)
        stderr_path = report_dir / "stderr.log"

        if stderr_path.exists():
            print(stderr_path)
            # Read the stderr.log file
            with open(stderr_path, "r", encoding="utf-8") as file:
                lines = file.read().split("\n")

            # Keep only lines that start with "event:"
            filtered_lines = [line for line in lines if line.startswith("event:")]
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


def main():
    for seed in seeds:
        for delay in delays:
            for drop_rate in drop_rates:
                format_stderr(seed, delay, drop_rate, save=True)


if __name__ == "__main__":
    main()
