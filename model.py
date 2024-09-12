import torch.nn as nn
import timm
import torch

def get_model(model_name, num_classes, device):
    """
    Creates and returns the model with the given model_name.
    
    Args:
        model_name (str): The model architecture name to load from timm.
        num_classes (int): Number of output classes.
        device (torch.device): Device to which the model will be moved.
        
    Returns:
        torch.nn.Module: Model loaded on the specified device.
    """
    model = timm.create_model(model_name, pretrained=True)
    
    # Modify the classifier head based on the output number of classes
    model.head = nn.Sequential(
        nn.Linear(model.head.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    
    return model.to(device)
