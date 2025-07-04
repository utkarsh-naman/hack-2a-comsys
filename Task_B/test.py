import os
import torch
import timm
from torch import nn
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import warnings

warnings.filterwarnings("ignore")

# Model Definition
class EmbeddingNet(nn.Module):
    def __init__(self, backbone='resnet50', embedding_size=512):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
        self.fc = nn.Linear(self.backbone.num_features, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        x = self.bn(x)
        return x

# Extract representative embedding for each train class
def get_class_embeddings(model, root_dir, transform, device):
    model.eval()
    embeddings = {}
    with torch.no_grad():
        for cls in sorted(os.listdir(root_dir)):
            class_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(class_dir): continue

            for root, _, files in os.walk(class_dir):
                for fname in files:
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        path = os.path.join(root, fname)
                        img = transform(Image.open(path).convert('RGB')).unsqueeze(0).to(device)
                        emb = model(img).squeeze().cpu()
                        embeddings[cls] = emb
                        break
                if cls in embeddings:
                    break
    return embeddings

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    model = EmbeddingNet().to(device)
    model.load_state_dict(torch.load("tb_model.pth", map_location=device))
    model.eval()
    print("Loaded model from tb_model.pth")

    class_embeddings = get_class_embeddings(model, "train", transform, device)
    if not class_embeddings:
        print("No class embeddings found. Make sure 'train/' folder is correct.")
        return

    y_true, y_pred = [], []

    with torch.no_grad():
        for cls in sorted(os.listdir("val")):
            class_dir = os.path.join("val", cls)
            if not os.path.isdir(class_dir): continue

            for root, _, files in os.walk(class_dir):
                for fname in files:
                    if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        continue

                    path = os.path.join(root, fname)
                    img = transform(Image.open(path).convert('RGB')).unsqueeze(0).to(device)
                    emb = model(img).squeeze().cpu()

                    pred_cls = max(class_embeddings,
                                   key=lambda c: torch.cosine_similarity(emb, class_embeddings[c], dim=0).item())

                    y_true.append(cls)
                    y_pred.append(pred_cls)

    # Evaluation
    top1_acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)

    print(f"Top-1 Accuracy       : {top1_acc:.4f}")
    print(f"Macro F1-Score       : {macro_f1:.4f}")
    print(f"Macro Precision      : {macro_precision:.4f}")
    print(f"Macro Recall         : {macro_recall:.4f}")

if __name__ == "__main__":
    evaluate()
