import os
import csv

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Training on: {device}")

# Define paths
image_dir = "/ros_bridge/src/client_node/client_node/data/rgb_finetune/train"
csv_path = os.path.join(image_dir, "labels.csv")

# Function to extract numeric indexes from filenames
def extract_index(filename):
    return int(filename.split(".")[0])

# Get all image files in the directory
image_files = [f for f in os.listdir(image_dir) if f.endswith(".png")]
image_indexes = sorted(extract_index(f) for f in image_files)

# Read CSV file and extract indexes
csv_indexes = []
if os.path.exists(csv_path):
    with open(csv_path, "r") as csv_file:
        reader = csv.reader(csv_file)
        next(reader)  # Skip header
        csv_indexes = sorted(extract_index(row[0]) for row in reader)

# Find missing indexes in images and CSV
missing_in_images = sorted(set(csv_indexes) - set(image_indexes))
missing_in_csv = sorted(set(image_indexes) - set(csv_indexes))

# Print results
print(f"Total images: {len(image_indexes)}")
print(f"Total entries in CSV: {len(csv_indexes)}")
print(f"Missing in images: {missing_in_images}")
print(f"Missing in CSV: {missing_in_csv}")

# Optional: Suggest cleanup actions
if missing_in_images:
    print("\nYou may want to delete these entries from the CSV:")
    for idx in missing_in_images:
        print(f"{idx:05d}.png")

if missing_in_csv:
    print("\nYou may want to delete these files from the directory:")
    for idx in missing_in_csv:
        print(f"{idx:05d}.png")
        