import os
import torch
import timm
import random
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Dataset
class TripletFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.data = []

        for cls in self.classes:
            class_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(class_dir):
                continue

            # All images: normal + distortion
            images = []
            for fname in os.listdir(class_dir):
                fpath = os.path.join(class_dir, fname)
                if os.path.isfile(fpath) and fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    images.append(fpath)

            distortion_dir = os.path.join(class_dir, "distortion")
            if os.path.isdir(distortion_dir):
                for fname in os.listdir(distortion_dir):
                    fpath = os.path.join(distortion_dir, fname)
                    if os.path.isfile(fpath) and fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        images.append(fpath)

            if len(images) < 2:
                continue  # Need at least one anchor and one positive

            for i in range(1, len(images)):
                self.data.append((images[0], images[i], cls))  # anchor, positive, class

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        anchor_path, positive_path, cls = self.data[idx]
        negative_cls = random.choice([c for c in self.classes if c != cls])
        negative_dir = os.path.join(self.root_dir, negative_cls)

        # Get any valid image from the negative class
        negative_images = []
        for fname in os.listdir(negative_dir):
            fpath = os.path.join(negative_dir, fname)
            if os.path.isfile(fpath) and fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                negative_images.append(fpath)

        distortion_dir = os.path.join(negative_dir, "distortion")
        if os.path.isdir(distortion_dir):
            for fname in os.listdir(distortion_dir):
                fpath = os.path.join(distortion_dir, fname)
                if os.path.isfile(fpath) and fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    negative_images.append(fpath)

        if not negative_images:
            raise FileNotFoundError(f"No valid negative images found for class {negative_cls}")

        negative_path = random.choice(negative_images)

        anchor = self.transform(Image.open(anchor_path).convert('RGB'))
        positive = self.transform(Image.open(positive_path).convert('RGB'))
        negative = self.transform(Image.open(negative_path).convert('RGB'))

        return anchor, positive, negative

# Model
class EmbeddingNet(nn.Module):
    def __init__(self, backbone='resnet50', embedding_size=512):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)  # output: 2048 for resnet50
        self.fc = nn.Linear(self.backbone.num_features, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        x = self.backbone(x)           # (batch, 2048)
        x = self.fc(x)                 # (batch, 512)
        x = self.bn(x)                 # (batch, 512)
        return x


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    dataset = TripletFaceDataset(root_dir='train', transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)

    model = EmbeddingNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.TripletMarginLoss(margin=1.0)

    model.train()
    for epoch in range(20):
        total_loss = 0
        for anchor, positive, negative in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            emb_a = model(anchor)
            emb_p = model(positive)
            emb_n = model(negative)

            loss = criterion(emb_a, emb_p, emb_n)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), 'tb_model.pth')
    print("Model saved as tb_model.pth")

if __name__ == "__main__":
    train()
