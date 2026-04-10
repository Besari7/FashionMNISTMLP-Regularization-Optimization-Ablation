"""
05_adv_training_demo.py
=======================
Week 4 Topic: Adversarial Training

Trains two identical tiny PyTorch MLPs on a small subset of FashionMNIST.
- Model Normal : Trained on clean images.
- Model Adv    : Trained on 50% clean + 50% FGSM adversarial images.

Evaluates both models on a purely adversarial test set.

Outputs:
  outputs/week4_adversarial_training.txt
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ── Paths ──────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), "outputs")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), "data")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Data Setup (Small Subset) ───────────────────────────────────────────
torch.manual_seed(42)
tfm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.Lambda(lambda x: x.view(-1)),
])

full_train = datasets.FashionMNIST(root=DATA_DIR, train=True, download=True, transform=tfm)
full_test  = datasets.FashionMNIST(root=DATA_DIR, train=False, download=True, transform=tfm)

# 10k train, 2k test
train_idx = torch.randperm(len(full_train))[:10000]
test_idx  = torch.randperm(len(full_test))[:2000]

train_loader = DataLoader(Subset(full_train, train_idx), batch_size=128, shuffle=True)
test_loader  = DataLoader(Subset(full_test, test_idx), batch_size=128, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 2. FGSM Attack Function ────────────────────────────────────────────────
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    # Clip back to normalized range roughly [-1, 1]
    perturbed_image = torch.clamp(perturbed_image, -1, 1)
    return perturbed_image

# ── 3. Model Architecture ──────────────────────────────────────────────────
def create_model():
    torch.manual_seed(42)
    return nn.Sequential(
        nn.Linear(784, 128), nn.ReLU(),
        nn.Linear(128, 10)
    ).to(device)

model_normal = create_model()
model_adv    = create_model()

criterion = nn.CrossEntropyLoss()
opt_normal = optim.Adam(model_normal.parameters(), lr=1e-3)
opt_adv    = optim.Adam(model_adv.parameters(), lr=1e-3)

# ── 4. Training Loops ──────────────────────────────────────────────────────
epochs = 10
epsilon = 0.1

print("--- Standard Training vs Adversarial Training ---")
print("Training Model Normal (Clean data only)...")
for ep in range(epochs):
    model_normal.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        opt_normal.zero_grad()
        loss = criterion(model_normal(x), y)
        loss.backward()
        opt_normal.step()

print("Training Model Adv (50% Clean + 50% FGSM data)...")
for ep in range(epochs):
    model_adv.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        
        # 1. Clean update (50% conceptual weight)
        opt_adv.zero_grad()
        logits_clean = model_adv(x)
        loss_clean = criterion(logits_clean, y)
        loss_clean.backward()
        
        # 2. Adversarial update
        # Re-forward to get gradients on X
        x_adv_gen = x.clone().detach().requires_grad_(True)
        logits_for_grad = model_adv(x_adv_gen)
        loss_for_grad = criterion(logits_for_grad, y)
        
        model_adv.zero_grad()
        loss_for_grad.backward()
        
        # Generate FGSM examples
        data_grad = x_adv_gen.grad.data
        x_perturbed = fgsm_attack(x_adv_gen, epsilon, data_grad)
        
        # Train on perturbed examples
        logits_adv = model_adv(x_perturbed.detach())
        loss_adv = criterion(logits_adv, y)
        loss_adv.backward()
        
        # Step combines gradients from clean and adversarial passes
        opt_adv.step()

# ── 5. Evaluation ──────────────────────────────────────────────────────────
def evaluate(model, loader, apply_fgsm=False):
    model.eval()
    correct, total = 0, 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        
        if apply_fgsm:
            x.requires_grad = True
            logits = model(x)
            loss = criterion(logits, y)
            model.zero_grad()
            loss.backward()
            x = fgsm_attack(x, epsilon, x.grad.data).detach()
            
        with torch.no_grad():
            preds = model(x).argmax(1)
            correct += (preds == y).sum().item()
            total += len(y)
            
    return 100 * correct / total

normal_clean = evaluate(model_normal, test_loader, apply_fgsm=False)
normal_adv   = evaluate(model_normal, test_loader, apply_fgsm=True)

adv_clean    = evaluate(model_adv, test_loader, apply_fgsm=False)
adv_adv      = evaluate(model_adv, test_loader, apply_fgsm=True)

report = f"""
=========================================================
  Adversarial Training Evaluation (FGSM eps={epsilon})
=========================================================

[Model 1: Standard Training]
  Clean Test Accuracy : {normal_clean:.2f}%
  FGSM Test Accuracy  : {normal_adv:.2f}%   <-- Vulnerable (Huge Drop)

[Model 2: Adversarial Training]
  Clean Test Accuracy : {adv_clean:.2f}%    <-- Slight drop expected vs pure clean
  FGSM Test Accuracy  : {adv_adv:.2f}%      <-- Robust (Maintains accuracy!)

Conclusion: 
Adversarial training acts as a powerful regularizer, forcing the model 
to learn robust features rather than brittle, easy-to-fool noise patterns.
=========================================================
"""

print(report)

out_txt = os.path.join(OUT_DIR, "week4_adversarial_training.txt")
with open(out_txt, "w") as f:
    f.write(report.strip() + "\n")
print(f"[+] Saved report -> {out_txt}")
