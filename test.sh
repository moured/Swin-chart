#!/bin/bash

# Ensure script stops on first error
set -e

# Define the parameters at the top of the script
CHECKPOINT_PATH="/path/to/checkpoint/best_model.pth"        # Change this to your checkpoint path
IMAGE_FOLDER="/path/to/ICPR2022_CHARTINFO_UB_UNITEC_PMC_TEST_v2.1/chart_images/split_1"                    # Change this to your test images folder path
ANNOTATION_FOLDER="/path/to/ICPR2022_CHARTINFO_UB_UNITEC_PMC_TEST_v2.1/final_full_GT/split_1/annotations_JSON"          # Change this to your test annotations folder path
SAVE_PATH="/path/to/save/results"                      # Change this to your desired results saving path
MODEL_NAME="swin_base_patch4_window7_224"              # Change this to your model name
BATCH_SIZE=64                                          # Note! don't set it to 1

# Run the val.py script with the specified arguments
python test.py \
  --checkpoint "$CHECKPOINT_PATH" \
  --image_folder "$IMAGE_FOLDER" \
  --annotation_folder "$ANNOTATION_FOLDER" \
  --save_path "$SAVE_PATH" \
  --model_name "$MODEL_NAME" \
  --batch_size "$BATCH_SIZE"