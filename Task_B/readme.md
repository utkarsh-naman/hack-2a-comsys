# Facial Recognition using Triplet Loss and ResNet-50

This project implements a facial recognition system using triplet loss and the ResNet-50 backbone. The model learns to embed facial images such that embeddings of the same identity are closer, and those of different identities are far apart.

---

## 1. Training Script: `train.py`

### Purpose
Train a deep learning model using triplet loss for facial recognition.

### Dataset Structure
The training dataset is organized by identity:

```
train/
├── person1/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── distortion/
│       └── distorted_img1.jpg
├── person2/
│   └── ...
└── ...
```

### Key Components

- **TripletFaceDataset**
  - Creates triplets of images: (anchor, positive, negative)
  - Includes distorted images in training when available

- **EmbeddingNet Model**
  ```python
  import timm
  import torch.nn as nn

  class EmbeddingNet(nn.Module):
      def __init__(self):
          super().__init__()
          self.backbone = timm.create_model("resnet50", pretrained=True, num_classes=0)
          self.fc = nn.Linear(2048, 512)

      def forward(self, x):
          x = self.backbone(x)
          return self.fc(x)
  ```

- **Loss Function**
  - Uses `TripletMarginLoss` to pull same-identity embeddings close and push different ones apart

- **Training Loop**
  - Trains for 20 epochs
  - Optimizer: `AdamW`
  - Saves model as `tb_model.pth`

### Output
- Logs training loss per epoch
- Trained model saved to `tb_model.pth`

---

## 2. Testing Script: `test.py`

### Purpose
Evaluate the model on a validation set by comparing facial embeddings using cosine similarity.

### Dataset Structure

```
val/
├── person1/
│   ├── img1.jpg
│   └── ...
├── person2/
│   └── ...
└── ...
```

### Key Components

- **Embedding Extraction**
  - For each class in `train/`, extract one representative embedding
  - Compare validation image embeddings to class embeddings using cosine similarity

- **Evaluation**
  - Prints classification report
  - Displays and saves confusion matrix to `confusion_matrix.png`

---

## Requirements

- Python  
- PyTorch  
- torchvision  
- timm  
- scikit-learn  
- PIL (Pillow)  
- tqdm  
- matplotlib  

---

## How to Run

### Train the Model
```bash
python train.py
```

### Evaluate the Model
```bash
python test.py
```


## Conclusion

This project demonstrates facial recognition using deep metric learning with triplet loss and ResNet-50. The embeddings generated can be used for face verification, clustering, or identity recognition tasks.
