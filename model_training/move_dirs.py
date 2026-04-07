import os
from pathlib import Path
import shutil

log_dir = Path("logs/snn")

for item in log_dir.iterdir():
    # Extract date from directory/file name (first part before underscore)
    # e.g. "2026-03-30_17-20-11_150844_snn_pid96187" -> "2026-03-30"
    date = item.name.split("_")[0]  # e.g. 2026-03-30
    target_dir = log_dir / date
    target_dir.mkdir(exist_ok=True)

    # Only move if item is not already in a date-organized subdirectory
    if item.parent == log_dir:
        dest = target_dir / item.name
        if item != target_dir:  # Avoid moving dir to itself
            print(f"Moving {item.name} to {target_dir}")
            shutil.move(str(item), str(dest))