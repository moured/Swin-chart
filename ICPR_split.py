import os
import shutil
import random
import argparse

def split_dataset(root_dir, train_dir, val_dir, split_ratio=0.8):
    # Define paths for annotations and images
    annotations_path = os.path.join(root_dir, 'annotations_JSON')
    images_path = os.path.join(root_dir, 'images')

    # Prepare output directories
    train_annotations = os.path.join(train_dir, 'annotations_JSON')
    val_annotations = os.path.join(val_dir, 'annotations_JSON')
    train_images = os.path.join(train_dir, 'images')
    val_images = os.path.join(val_dir, 'images')

    # Create the folders for train and val splits
    for path in [train_annotations, val_annotations, train_images, val_images]:
        os.makedirs(path, exist_ok=True)

    # Get all subfolders
    subfolders = os.listdir(annotations_path)

    for subfolder in subfolders:
        subfolder_anno_path = os.path.join(annotations_path, subfolder)
        subfolder_img_path = os.path.join(images_path, subfolder)

        # Create corresponding subfolders in train/val directories
        os.makedirs(os.path.join(train_annotations, subfolder), exist_ok=True)
        os.makedirs(os.path.join(val_annotations, subfolder), exist_ok=True)
        os.makedirs(os.path.join(train_images, subfolder), exist_ok=True)
        os.makedirs(os.path.join(val_images, subfolder), exist_ok=True)

        # Get all JSON files and corresponding image files
        json_files = os.listdir(subfolder_anno_path)
        total_files = len(json_files)
        random.shuffle(json_files)

        # Split the data into train and val
        split_point = int(total_files * split_ratio)
        train_files = json_files[:split_point]
        val_files = json_files[split_point:]

        # Copy files to train and val directories
        for file in train_files:
            json_file_path = os.path.join(subfolder_anno_path, file)
            image_file_path = os.path.join(subfolder_img_path, file.replace('.json', '.jpg'))

            shutil.copy(json_file_path, os.path.join(train_annotations, subfolder, file))
            shutil.copy(image_file_path, os.path.join(train_images, subfolder, file.replace('.json', '.jpg')))

        for file in val_files:
            json_file_path = os.path.join(subfolder_anno_path, file)
            image_file_path = os.path.join(subfolder_img_path, file.replace('.json', '.jpg'))

            shutil.copy(json_file_path, os.path.join(val_annotations, subfolder, file))
            shutil.copy(image_file_path, os.path.join(val_images, subfolder, file.replace('.json', '.jpg')))

# Main function to take arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train and val sets.")
    parser.add_argument('--root_dir', type=str, required=True, help="Path to the root dataset directory")
    parser.add_argument('--train_dir', type=str, required=True, help="Path to save the training dataset")
    parser.add_argument('--val_dir', type=str, required=True, help="Path to save the validation dataset")
    parser.add_argument('--split_ratio', type=float, default=0.8, help="Ratio of data for training (default is 0.8)")

    args = parser.parse_args()

    # Call the function with the provided arguments
    split_dataset(args.root_dir, args.train_dir, args.val_dir, split_ratio=args.split_ratio)
