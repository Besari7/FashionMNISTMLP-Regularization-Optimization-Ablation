"""
06_rotation_pretext_demo.py
===========================
Week 4 coverage:
- Self-supervised learning via rotation prediction pretext task

Pipeline:
1) Build self-supervised dataset by rotating FashionMNIST images by
   {0, 90, 180, 270} degrees and predicting the rotation class.
2) Train a compact encoder+head on a small subset for a few epochs.

Output:
  supporting_experiments/outputs/week4_self_supervised_rotation.txt
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

HERE = os.path.dirname(os.path.abspath(__file__))
WEEK4_DIR = os.path.dirname(HERE)
SUPPORT_DIR = os.path.dirname(WEEK4_DIR)
PROJECT_ROOT = os.path.dirname(SUPPORT_DIR)
OUT_DIR = os.path.join(SUPPORT_DIR, "outputs")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
SUBSET = 2000
BATCH = 128
EPOCHS = 3


def _rotate_batch(x, k):
    # x: [B, 1, H, W], rotate by 90*k degrees
    return torch.rot90(x, k=k, dims=(2, 3))


class RotationDataset(Dataset):
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, _ = self.base[idx]
        rot_label = torch.randint(0, 4, (1,)).item()
        x = _rotate_batch(x.unsqueeze(0), rot_label).squeeze(0)
        return x, rot_label


def build_model():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 4),
    ).to(DEVICE)


def main():
    torch.manual_seed(SEED)

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    train_ds = datasets.FashionMNIST(root=DATA_DIR, train=True, download=True, transform=tfm)
    idx = torch.randperm(len(train_ds), generator=torch.Generator().manual_seed(SEED))[:SUBSET]
    train_sub = Subset(train_ds, idx.tolist())
    rot_ds = RotationDataset(train_sub)
    loader = DataLoader(rot_ds, batch_size=BATCH, shuffle=True, num_workers=0)

    model = build_model()
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        correct = 0
        total = 0
        running = 0.0

        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()

            running += loss.item() * y.size(0)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        avg_loss = running / max(1, total)
        acc = 100.0 * correct / max(1, total)
        history.append((avg_loss, acc))
        print(f"Epoch {epoch}/{EPOCHS} | loss={avg_loss:.4f} | rot-acc={acc:.2f}%")

    report = [
        "================================================================",
        "Week 4: Self-Supervised Rotation Pretext Demo",
        "================================================================",
        f"Device: {DEVICE}",
        f"Subset: {SUBSET}",
        f"Epochs: {EPOCHS}",
        "",
        "Epoch metrics (loss, rotation accuracy):",
    ]
    for i, (loss, acc) in enumerate(history, start=1):
        report.append(f"- Epoch {i}: loss={loss:.4f}, acc={acc:.2f}%")

    report.append("")
    report.append("Note: This is a compact pretext-task demo for self-supervised coverage.")
    report.append("================================================================")

    out_txt = os.path.join(OUT_DIR, "week4_self_supervised_rotation.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    print(f"[+] Saved {out_txt}")


if __name__ == "__main__":
    main()
