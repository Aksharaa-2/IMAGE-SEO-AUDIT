"""
test_fusion.py — Tests whether the model learns on 32 REAL pairs
from your actual data (16 match + 16 mismatch).
"""
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
import re
import numpy as np

torch.manual_seed(42)

ROOT = Path(r"E:\image_seo_project")
CSV = ROOT / "processed_clean" / "train_image_text_pairs.csv"
df = pd.read_csv(CSV)

# Pick 16 images that have BOTH labels, use 2 rows each = 32 samples
groups = df.groupby("image_path")["label"].nunique()
both_imgs = groups[groups == 2].index.tolist()[:16]

rows = []
for img in both_imgs:
    sub = df[df["image_path"] == img]
    rows.append(sub[sub["label"] == 1].iloc[0])
    rows.append(sub[sub["label"] == 0].iloc[0])
sample = pd.DataFrame(rows).reset_index(drop=True)
print(f"Sample: {len(sample)} rows, labels: {sample['label'].value_counts().to_dict()}")

# Tokenizer
def tokenize(t): return re.findall(r"[a-z0-9]+", str(t).lower())
vocab = {"<PAD>": 0, "<UNK>": 1}
for _, r in df.iterrows():
    for w in tokenize(r["text"]):
        if w not in vocab: vocab[w] = len(vocab)

def encode(t, L=20):
    ids = [vocab.get(w, 1) for w in tokenize(t)][:L]
    ids += [0] * (L - len(ids))
    return ids

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

imgs = torch.stack([tf(Image.open(r["image_path"]).convert("RGB")) for _, r in sample.iterrows()])
ids = torch.tensor([encode(r["text"]) for _, r in sample.iterrows()], dtype=torch.long)
labels = torch.tensor([float(r["label"]) for _, r in sample.iterrows()])

# ---- Model: EXACTLY the same architecture ----
class Model(nn.Module):
    def __init__(self, V):
        super().__init__()
        self.image_encoder = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.image_encoder.classifier = nn.Identity()
        self.embedding = nn.Embedding(V, 128, padding_idx=0)
        self.text_encoder = nn.LSTM(128, 64, 1, batch_first=True, bidirectional=True)
        self.image_proj = nn.Linear(1280, 256)
        self.text_proj  = nn.Linear(128, 256)
        self.classifier = nn.Sequential(
            nn.Linear(256*3, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
    def forward(self, x, t):
        img = self.image_proj(self.image_encoder(x))
        e = self.embedding(t)
        mask = (t != 0).unsqueeze(-1).float()
        o, _ = self.text_encoder(e * mask)
        s = (o * mask).sum(1) / mask.sum(1).clamp(min=1)
        txt = self.text_proj(s)
        diff = img - txt
        return self.classifier(torch.cat([img, txt, diff], 1)).squeeze(1)

model = Model(len(vocab))
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
crit = nn.BCEWithLogitsLoss()

print("\nTraining on 32 samples for 300 steps...\n")
for step in range(300):
    logits = model(imgs, ids)
    loss = crit(logits, labels)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 25 == 0 or step == 299:
        acc = ((torch.sigmoid(logits) > 0.5).float() == labels).float().mean().item()
        print(f"  Step {step:3d} | Loss: {loss.item():.4f} | Acc: {acc*100:.1f}%")

print("\n→ If final Acc = 100% → MODEL IS FINE. Bug is in the full pipeline.")
print("→ If final Acc ≈ 50% → MODEL IS BROKEN. Bug is in architecture.")