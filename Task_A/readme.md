
# Gender Classification using EfficientNet-B0

This project implements a gender classification model using the EfficientNet-B0 architecture with PyTorch. The model is trained to distinguish between male and female facial images.

---

## 1. Training Script: `train.py`

### Purpose
Train an EfficientNet-B0 model on a labeled image dataset for gender classification.

### Dataset Structure
The dataset is organized in subdirectories under `train/`:

```
train/
├── male/
│   ├── img1.jpg
│   └── ...
└── female/
    ├── img1.jpg
    └── ...
```

### Key Components

- **Custom Dataset Class**
  - Loads and preprocesses images using OpenCV.
  - Converts images to tensors and applies transformations.

- **Model Architecture**
  ```python
  from torchvision.models import efficientnet_b0

  model = efficientnet_b0(weights='DEFAULT')
  model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=2)
  ```

- **Training Pipeline**
  - Splits data into training and validation sets.
  - Applies data augmentation during training.
  - Uses **Adam** optimizer with learning rate scheduler.
  - Loss Function: `CrossEntropyLoss`.
  - Saves best model to `b0_gitcolab.pth`.

### Output
- Console logs per epoch showing:
  - Loss
  - Training Accuracy
  - Validation Accuracy
- Final model weights saved to file.

---

## 2. Testing Script: `test.py`

### Purpose
Evaluate the trained EfficientNet-B0 model on a validation/test dataset.

### Dataset Structure

```
val/
├── male/
│   ├── img1.jpg
│   └── ...
└── female/
    ├── img1.jpg
    └── ...
```

### Key Components

- Loads trained model weights from `b0_gitcolab.pth`.
- Applies the same transformations as during validation.
- Predicts labels and compares them with ground truth.
- Outputs classification metrics and logs misclassified samples.

### Evaluation Metrics

- Classification Report: **Precision**, **Recall**, **F1-Score**
- Per-image prediction results
- Total number of correct and incorrect predictions
- Saves misclassified paths to `wrong_predictions.txt`

### Example Output

```
✅ CORRECT | File: val/male/img1.jpg | Actual: male | Predicted: male  
❌ WRONG   | File: val/female/img2.jpg | Actual: female | Predicted: male  
...
Total Correct: 180 / 200  
Total Wrong: 20 / 200
```

---

## Requirements

- Python  
- PyTorch  
- torchvision  
- scikit-learn  
- OpenCV  
- PIL (Pillow)  
- NumPy  

---

## How to Run

### Train the Model
```bash
python train.py
```

### Test the Model
```bash
python test.py
```


## Conclusion

This project demonstrates a complete workflow to train and evaluate an image classification model using transfer learning with EfficientNet-B0 for gender classification. The model achieved optimal validation accuracy and supports per-image analysis during testing.
