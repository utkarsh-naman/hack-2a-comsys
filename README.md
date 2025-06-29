# hack-2a-comsys
A facial recognition and for gender (binary class) and identity (multi class) classification project for COMSYS hackathon

This repository contains two deep learning projects implemented in PyTorch:

1. **Gender Classification** using EfficientNet-B0  
2. **Facial Recognition** using ResNet-50 + Triplet Loss  

Each task is structured for end-to-end training and evaluation using real-world facial image datasets.

---

## 1. Gender Classification

Classifies a face image as **male** or **female** using transfer learning with EfficientNet-B0.

### Architecture
- Backbone: `efficientnet_b0(weights='DEFAULT')`
- Final layer: 2 output units
- Loss: CrossEntropyLoss
- Optimizer: Adam

### Dataset Structure

```
train/
├── male/
│   └── img1.jpg
└── female/
    └── img2.jpg

val/
├── male/
└── female/
```

### Training (`train.py`)

```bash
python train.py
```

Logs training and validation accuracy each epoch. Saves best model as `b0_gitcolab.pth`.

### Evaluation (`test.py`)

```bash
python test.py
```

Sample Output:
```
CORRECT | val/male/img1.jpg → male  
WRONG   | val/female/img2.jpg → predicted: male
```

- Classification metrics: Precision, Recall, F1
- Outputs `wrong_predictions.txt`

---

## 2. Facial Recognition with Triplet Loss

Learns facial **embeddings** using ResNet-50 and triplet loss to perform identity recognition.

### Architecture
- Backbone: `resnet50` (via `timm`) with no classification head
- Embedding size: 512
- Loss: `TripletMarginLoss`
- Optimizer: AdamW

### Dataset Structure

```
train/
├── person1/
│   ├── img1.jpg
│   └── distortion/
│       └── distorted1.jpg
└── person2/
    └── ...

val/
├── person1/
└── person2/
```

### Training (`train.py`)

```bash
python train.py
```

Saves model as `tb_model.pth`.

### Evaluation (`test.py`)

```bash
python test.py
```

Evaluation Metrics:
- Top-1 Accuracy
- Macro-averaged F1-Score

Example Output:
```
Top-1 Accuracy: 0.8889  
Macro-averaged F1-Score: 0.8857
```

---

## Requirements

- Python
- PyTorch
- torchvision
- timm
- scikit-learn
- OpenCV
- Pillow (PIL)
- tqdm
- matplotlib

---

## How to Use

1. Prepare your dataset as per the structures shown above.
2. Train each model:
   ```bash
   python train.py
   ```
3. Evaluate performance:
   ```bash
   python test.py
   ```

---

## Conclusion

This repo demonstrates:
- A complete binary classification workflow (gender)
- Deep metric learning for face recognition (triplet loss)

Both systems are modular and ready for further extension (e.g., GUI, API, clustering, real-time webcam input).
