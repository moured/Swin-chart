import argparse
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_recall_fscore_support
from data_loader import ChartImageDataset
from torch.utils.data import DataLoader
from model import get_model  # Import get_model from your model definition
import json  # To load the config
from torchvision import transforms as t
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Define the test function
def test(model, criterion, test_dataloader, classes, device, save_path):
    test_loss = 0.0
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))

    model.eval()
    preds = []
    targets = []

    # Loop through the test dataloader
    for data, target in tqdm(test_dataloader, desc="Testing"):
        data, target = data.to(device), target.to(device)

        with torch.no_grad():
            output = model(data)
            loss = criterion(output, target)

        test_loss += loss.item() * data.size(0)
        _, pred = torch.max(output, 1)
        preds.append(pred.cpu())
        targets.append(target.cpu())

        # Correct tensor
        correct_tensor = pred.eq(target.view_as(pred))

        # Convert to numpy and flatten
        correct = correct_tensor.cpu().numpy().squeeze()

        # Accumulate correct predictions per class
        for i in range(len(target)):
            label = target.data[i].item()
            class_correct[label] += correct[i].item() if isinstance(correct, np.ndarray) else correct.item()
            class_total[label] += 1

    # Concatenate predictions and targets
    preds = torch.cat(preds)
    targets = torch.cat(targets)

    preds_list = preds.numpy()
    targets_list = targets.numpy()

    # Calculate precision, recall, f1-measure
    metrics = precision_recall_fscore_support(targets_list, preds_list, average='weighted')
    precision, recall, f1_weighted = metrics[0], metrics[1], metrics[2]

    # Calculate macro F1-score
    f1_macro = f1_score(targets_list, preds_list, average='macro')

    # Calculate the average test loss
    test_loss = test_loss / len(test_dataloader.dataset)

    # Calculate overall accuracy
    overall_accuracy = 100. * np.sum(class_correct) / np.sum(class_total)

    # Generate the confusion matrix
    conf_matrix = confusion_matrix(targets_list, preds_list)

    # Create a heatmap of the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix Heatmap')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Save the heatmap to the save path
    heatmap_path = f"{save_path}/confusion_matrix_heatmap.png"
    plt.savefig(heatmap_path)
    plt.close()

    print(f"Confusion matrix heatmap saved to {heatmap_path}")

    # Save and print the results
    with open(f"{save_path}/results.txt", "w") as f:
        f.write(f"\nTest Loss: {test_loss:.6f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-Score (Weighted): {f1_weighted:.4f}\n")
        f.write(f"F1-Score (Macro): {f1_macro:.4f}\n")
        f.write(f"Overall Test Accuracy: {overall_accuracy:.4f} ({np.sum(class_correct)}/{np.sum(class_total)})\n")
        f.write("\nClass-wise Test Accuracy:\n")

        for i in range(len(classes)):
            if class_total[i] > 0:
                f.write(f"Test Accuracy of {classes[i]}: {100 * class_correct[i] / class_total[i]:.2f}% "
                        f"({int(class_correct[i])}/{int(class_total[i])})\n")
            else:
                f.write(f"Test Accuracy of {classes[i]}: N/A (no test examples)\n")

    print(f"Results saved to {save_path}/results.txt")


# Main function
def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the test set.")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the model checkpoint (.pth file)")
    parser.add_argument('--image_folder', type=str, required=True, help="Path to the test image folder")
    parser.add_argument('--annotation_folder', type=str, required=True, help="Path to the test annotation folder")
    parser.add_argument('--save_path', type=str, required=True, help="Path to save results")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for testing")
    parser.add_argument('--model_name', type=str, required=True, help="Model architecture name")

    args = parser.parse_args()

    # Load config
    with open('config.json') as f:
        config = json.load(f)  # Load the configuration

    # Load the model (modify this according to your model loading function)
    model = get_model(args.model_name, len(config['label_to_idx']), device=torch.device("cpu"))  # Adjust for your own function

    # Load the checkpoint
    checkpoint = torch.load(args.checkpoint)

    # If the model was saved with DataParallel, we need to remove 'module.' from the keys
    try:
        state_dict = checkpoint['model_state_dict']
    except:
        state_dict = checkpoint
    new_state_dict = {}
    for key in state_dict.keys():
        new_key = key.replace('module.', '')  # Remove 'module.' prefix
        new_state_dict[new_key] = state_dict[key]

    # Load the state dict
    model.load_state_dict(new_state_dict)

    # Define the device (assumes GPU is available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Load the test dataset
    transforms = t.Compose([
        t.Resize((256, 256)),
        t.CenterCrop((224, 224)),
        t.ToTensor(),
        t.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create the test dataset and dataloader
    test_dataset = ChartImageDataset(
        annotation_folder_path=args.annotation_folder,
        image_folder_path=args.image_folder,
        label_to_idx=config['label_to_idx'],
        transform=transforms,
        is_test=True
    )

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Test dataset size: {len(test_dataset)}")

    # Define criterion
    criterion = nn.CrossEntropyLoss()

    # Load class labels
    classes = list(config['label_to_idx'].keys())

    # Run the test function
    test(model, criterion, test_dataloader, classes, device, args.save_path)


if __name__ == "__main__":
    main()
