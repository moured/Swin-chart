#!/bin/bash

# Define the paths
ROOT_DIR="/path/to/root"
TRAIN_DIR="/path/to/train"
VAL_DIR="/path/to/val"
SPLIT_RATIO=0.8  # Modify the split ratio if necessary

# Run the Python script with the defined paths
python ICPR_split.py --root_dir "$ROOT_DIR" --train_dir "$TRAIN_DIR" --val_dir "$VAL_DIR" --split_ratio "$SPLIT_RATIO"
