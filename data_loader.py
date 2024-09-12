import os
import json
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
from torchvision import transforms as t


class ChartImageDataset(Dataset):
    def __init__(self, annotation_folder_path, image_folder_path, label_to_idx, transform=None, is_test=False):
        """
        Args:
            annotation_folder_path (string): Path to the annotation folder with annotations (JSONs).
            image_folder_path (string): Root directory with subfolders containing all the images.
            label_to_idx (dict): Dictionary mapping chart type labels to numerical indices.
            transform (callable, optional): Optional transform to be applied on an image sample.
        """
        self.annotation_folder_path = annotation_folder_path
        self.image_folder_path = image_folder_path
        self.label_to_idx = label_to_idx
        self.transform = transform
        self.is_test = is_test
        
        # Recursively get all image paths from subfolders
        self.image_paths = []
        for root, dirs, files in os.walk(image_folder_path):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(root, file))

        # Sort image paths for consistency
        self.image_paths = sorted(self.image_paths)

    def __len__(self):
        """Return the total number of samples."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Retrieve an image and its corresponding label based on the index."""
        # Load the image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        # Extract image name (without extension) to find the corresponding JSON file
        if self.is_test:
            relative_path = os.path.basename(img_path)
            annotation_file = relative_path.replace('jpg','json')
        else:
            relative_path = os.path.relpath(img_path, self.image_folder_path).rsplit('.', 1)[0]
            annotation_file = relative_path + '.json'
        
        annotation_path = os.path.join(self.annotation_folder_path, annotation_file)

        # Load the JSON annotation
        with open(annotation_path, 'r') as f:
            annotation = json.load(f)

        # Extract the chart type and map it to the index using label_to_idx
        chart_type = annotation['task1']['output']['chart_type'].replace(' ', '_')
        label = self.label_to_idx[chart_type]

        # Apply any transformations to the image
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label)
        
def get_dataloaders(config):
    """
    Creates and returns the dataloaders for training and validation along with dataset sizes.

    Args:
        config (dict): Configuration dictionary containing dataset paths and parameters.

    Returns:
        dict: Dataloaders for 'train' and 'val'.
        dict: Dataset sizes for 'train' and 'val'.
    """
    # Define transformations for training
    transforms_train = t.Compose([
        t.Resize((256, 256)),
        t.RandomHorizontalFlip(),
        t.RandomVerticalFlip(),
        t.CenterCrop((224, 224)),
        t.ToTensor(),
        t.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Define transformations for validation
    transforms_val = t.Compose([
        t.Resize((256, 256)),
        t.CenterCrop((224, 224)),
        t.ToTensor(),
        t.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create training dataset
    train_dataset = ChartImageDataset(
        annotation_folder_path=config['train_annotation_folder'],
        image_folder_path=config['train_image_folder'],
        label_to_idx=config['label_to_idx'],
        transform=transforms_train
    )

    # Create validation dataset using the label_to_idx from the config
    val_dataset = ChartImageDataset(
        annotation_folder_path=config['val_annotation_folder'],
        image_folder_path=config['val_image_folder'],
        label_to_idx=config['label_to_idx'],
        transform=transforms_val
    )

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    # Dataloader dictionary
    dataloaders = {'train': train_dataloader, 'val': val_dataloader}

    # Dataset sizes dictionary
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

    # Print dataset report
    print("Dataset Report:")
    print(f"  Number of training samples: {len(train_dataset)}")
    print(f"  Number of validation samples: {len(val_dataset)}")
    print(f"  Number of classes: {len(config['label_to_idx'])}")


    return dataloaders, dataset_sizes