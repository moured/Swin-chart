import argparse
import json
import torch
import time
import copy
from tqdm import tqdm
from data_loader import get_dataloaders #, ChartImageDataset
from model import get_model
from torch.optim import AdamW
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from torch.optim import lr_scheduler
from timm.loss import LabelSmoothingCrossEntropy

# in .cache/torch/hub/checkpoints execute:
# wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22kto1k.pth 

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, config):
    since = time.time()
    
    # Tracking best model weights and accuracy
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # Lists to store losses and accuracies for visualization
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        print("-" * 50)
        
        for phase in ['train', 'val']:  # Training and Validation Phase
            if phase == 'train':
                model.train()  # Set model to training mode
                print("\nTraining Phase:")
            else:
                model.eval()  # Set model to evaluation mode
                print("\nValidation Phase:")
            
            running_loss = 0.0
            running_corrects = 0.0
            
            # Iterate over data batches
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()  # Zero out gradients
                
                # Forward Pass (track only during training)
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train':
                scheduler.step()  # Adjust the learning rate at end of epoch
            
            # Compute epoch-level loss and accuracy
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}")
            
        # Save the best model based on validation accuracy
        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            # Save the best model based on the best accuracy
            torch.save(best_model_wts, f"{config['best_model_save_path']}/best_model.pth")
            print(f"Best model updated and saved to {config['best_model_save_path']}/best_model.pth")

        # Append losses and accuracies for tracking
        if phase == 'train':
            train_loss.append(epoch_loss)
            train_acc.append(epoch_acc)
        else:
            val_loss.append(epoch_loss)
            val_acc.append(epoch_acc)

        # Save the last model at the end of every epoch         
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
            'val_acc': epoch_acc
        }, f"{config['weights_save_path']}/swinl_224_cos_{epoch}.pth")

        print(f"Last model saved to {config['weights_save_path']}/swinl_224_cos_{epoch+1}.pth")

        print(f"End of Epoch {epoch+1}\n{'='*50}")

    return model

def main(config):
    # Set device (use all available GPUs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get dataloaders and dataset sizes
    dataloaders, dataset_sizes = get_dataloaders(config)

    # Load model with the specified model name from the config
    model = get_model(config['model_name'], len(config['label_to_idx']), device)

    # Enable DataParallel to use multiple GPUs
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)  # Wrap model with DataParallel

    model = model.to(device)

    # Loss function and optimizer
    criterion = LabelSmoothingCrossEntropy().to(device)
    optimizer = AdamW(model.parameters(), lr=config['lr'])
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=len(dataloaders['train']) // 2,
        cycle_mult=1.0,
        max_lr=config['max_lr'],
        min_lr=config['min_lr'],
        warmup_steps=len(dataloaders['train']) // 4,
        gamma=0.85
    )

    # Train the model
    train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument('--config', default='config.json', type=str, help='Path to the config file')
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    main(config)
