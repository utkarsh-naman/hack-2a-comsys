import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
import numpy as np

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_dir = "val"
    model_path = "b0_gitcolab.pth"

    # Transform (same as validation)
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Dataset and DataLoader
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Label mapping
    idx_to_class = {v: k for k, v in test_dataset.class_to_idx.items()}

    # Load model
    model = efficientnet_b0(weights=None)
    model.classifier[0] = nn.Dropout(p=0.4)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    correct_images = []
    wrong_images = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Accuracy, Precision, Recall, F1
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))

    # Misclassified file tracking
    print("\nPrediction Results Per Image:")
    for i, (path, label) in enumerate(test_dataset.samples):
        pred = all_preds[i]
        actual_label = idx_to_class[label]
        predicted_label = idx_to_class[pred]
        result = "✅ CORRECT" if label == pred else "❌ WRONG"
        print(f"{result} | File: {path} | Actual: {actual_label} | Predicted: {predicted_label}")

        if label == pred:
            correct_images.append(path)
        else:
            wrong_images.append(path)

    print(f"\nTotal Correct: {len(correct_images)} / {len(test_dataset)}")
    print(f"Total Wrong: {len(wrong_images)} / {len(test_dataset)}")

    # Optional: Save wrong predictions to a file
    with open("wrong_predictions.txt", "w") as f:
        for path in wrong_images:
            f.write(path + "\n")

if __name__ == "__main__":
    main()
