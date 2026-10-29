"""
find_bug.py — Isolates exactly where the model breaks.
"""
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
import re
import random
import numpy as np

torch.manual_seed(42)
random.seed(42)

ROOT = Path(r"E:\image_seo_project")
CSV = ROOT / "processed_clean" / "train_image_text_pairs.csv"

df = pd.read_csv(CSV).head(32).reset_index(drop=True)

# ---- Tokenizer ----
def tokenize(t): return re.findall(r"[a-z0-9]+", str(t).lower())
vocab = {"<PAD>": 0, "<UNK>": 1}
for _, r in df.iterrows():
    for w in tokenize(r["text"]):
        if w not in vocab: vocab[w] = len(vocab)

def encode(t, L=20):
    ids = [vocab.get(w, 1) for w in tokenize(t)][:L]
    ids += [0] * (L - len(ids))
    return ids

# ---- Prepare batch ----
tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])
imgs = torch.stack([tf(Image.open(r["image_path"]).convert("RGB")) for _, r in df.iterrows()])
ids = torch.tensor([encode(r["text"]) for _, r in df.iterrows()], dtype=torch.long)
labels = torch.tensor([float(r["label"]) for _, r in df.iterrows()])

print(f"Batch: images={imgs.shape}, ids={ids.shape}, labels={labels.shape}")
print(f"Label distribution: {labels.sum().item():.0f} pos / {(labels==0).sum().item()} neg")

# ============================================================
print("\n" + "="*60)
print("TEST 1: Image-only classifier")
print("="*60)
img_encoder = models.efficientnet_b0(weights=None)
img_encoder.classifier = nn.Linear(1280, 1)
opt = torch.optim.AdamW(img_encoder.parameters(), lr=1e-3)
crit = nn.BCEWithLogitsLoss()
for step in range(200):
    logits = img_encoder(imgs).squeeze(1)
    loss = crit(logits, labels)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 50 == 0:
        acc = ((torch.sigmoid(logits) > 0.5).float() == labels).float().mean().item()
        print(f"  Step {step:3d} | loss={loss.item():.4f} | acc={acc*100:.1f}%")
print("→ If loss drops below 0.3, image encoder works.")

# ============================================================
print("\n" + "="*60)
print("TEST 2: Text-only classifier")
print("="*60)
class TextOnly(nn.Module):
    def __init__(self, V):
        super().__init__()
        self.emb = nn.Embedding(V, 64, padding_idx=0)
        self.lstm = nn.LSTM(64, 64, 1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128, 1)
    def forward(self, t):
        e = self.emb(t)
        mask = (t != 0).unsqueeze(-1).float()
        o, _ = self.lstm(e * mask)
        s = (o * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.fc(s).squeeze(1)

txt_model = TextOnly(len(vocab))
opt = torch.optim.AdamW(txt_model.parameters(), lr=1e-3)
for step in range(200):
    logits = txt_model(ids)
    loss = crit(logits, labels)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 50 == 0:
        acc = ((torch.sigmoid(logits) > 0.5).float() == labels).float().mean().item()
        print(f"  Step {step:3d} | loss={loss.item():.4f} | acc={acc*100:.1f}%")
print("→ If loss stays at 0.69, the text alone is useless.")

# ============================================================
print("\n" + "="*60)
print("TEST 3: Does a SHUFFLED label model do as well?")
print("="*60)
print("(If shuffled labels give same accuracy as real labels,")
print(" the task has no learnable signal.)")

shuffled = labels[torch.randperm(len(labels))]
img_encoder = models.efficientnet_b0(weights=None)
img_encoder.classifier = nn.Linear(1280, 1)
opt = torch.optim.AdamW(img_encoder.parameters(), lr=1e-3)
for step in range(200):
    logits = img_encoder(imgs).squeeze(1)
    loss = crit(logits, shuffled)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 50 == 0:
        acc = ((torch.sigmoid(logits) > 0.5).float() == shuffled).float().mean().item()
        print(f"  Step {step:3d} | loss={loss.item():.4f} | acc={acc*100:.1f}%")

# ============================================================
print("\n" + "="*60)
print("TEST 4: Same-image, both-labels check")
print("="*60)
# Find an image that appears with both label 0 and 1
groups = df.groupby("image_path")["label"].nunique()
both = groups[groups == 2].index.tolist()
if both:
    img = both[0]
    print(f"Image {img} appears with both labels:")
    print(df[df["image_path"] == img][["label", "text"]].to_string())
else:
    print("NO image in this batch has both labels.")