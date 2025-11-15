import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from plyfile import PlyData, PlyElement

# -----------------------------
# Simple KPConv-style network (same as training version)
# -----------------------------

def knn(pts, K=6):
    """
    pts: (B, P, 3)
    returns idx: (B, P, K)
    """
    dist = torch.cdist(pts, pts)  # (B, P, P)
    idx = dist.topk(K, largest=False)[1]
    return idx

class KPConvLayer(nn.Module):
    def __init__(self, in_c, out_c, K=6):
        super().__init__()
        self.K = K
        self.mlp = nn.Sequential(
            nn.Linear(in_c + 3, out_c),
            nn.ReLU(),
            nn.Linear(out_c, out_c),
        )

    def forward(self, pts, feats):
        """
        pts:   (B, P, 3)
        feats: (B, P, C)
        return: (B, P, out_c)
        """
        B, P, C = feats.shape
        K = self.K

        idx = knn(pts, K)  # (B, P, K)

        # gather neighbor coords (B, P, K, 3)
        neigh_pts = torch.gather(
            pts.unsqueeze(1).expand(B, P, P, 3),
            2,
            idx.unsqueeze(-1).expand(B, P, K, 3),
        )

        # gather neighbor feats (B, P, K, C)
        feats_expand = feats.unsqueeze(1).expand(B, P, P, C)
        neigh_feats = torch.gather(
            feats_expand,
            2,
            idx.unsqueeze(-1).expand(B, P, K, C),
        )

        pts_expand = pts.unsqueeze(2).expand(B, P, K, 3)
        rel = neigh_pts - pts_expand  # (B, P, K, 3)

        inp = torch.cat([rel, neigh_feats], dim=-1)  # (B, P, K, 3+C)
        out = self.mlp(inp)                          # (B, P, K, out_c)
        out = out.max(dim=2)[0]                      # (B, P, out_c)
        return out

class KPNet(nn.Module):
    def __init__(self, base=8, num_classes=3, K=6):
        super().__init__()
        self.fc0 = nn.Linear(3, base)
        self.kp1 = KPConvLayer(base, base * 2, K)
        self.kp2 = KPConvLayer(base * 2, base * 4, K)
        self.head = nn.Sequential(
            nn.Linear(base * 4, base * 4),
            nn.ReLU(),
            nn.Linear(base * 4, num_classes),
        )

    def forward(self, pts):
        feats = F.relu(self.fc0(pts))
        feats = self.kp1(pts, feats)
        feats = self.kp2(pts, feats)
        out = self.head(feats)   # (B, P, num_classes)
        return out

# -----------------------------
# Argument parsing
# -----------------------------

parser = argparse.ArgumentParser(description="KPConv inference with float pred_label")
parser.add_argument("--input", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--chunk_size", type=int, default=4096)
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# -----------------------------
# Load model & checkpoint
# -----------------------------

NUM_CLASSES = 3
model = KPNet(base=8, num_classes=NUM_CLASSES, K=6).to(DEVICE)
print("Loading checkpoint:", args.checkpoint)
state = torch.load(args.checkpoint, map_location=DEVICE)
model.load_state_dict(state)
model.eval()
print("Model loaded.")

# -----------------------------
# Load PLY input
# -----------------------------

print("Reading input PLY:", args.input)
ply = PlyData.read(args.input)
vertex = ply["vertex"]

xyz = np.vstack([vertex["x"], vertex["y"], vertex["z"]]).T.astype(np.float32)
colors = np.vstack([vertex["red"], vertex["green"], vertex["blue"]]).T.astype(np.uint8)

N = xyz.shape[0]
print("Num points:", N)

# center coordinates same way as training
xyz_centered = xyz - xyz.mean(0, keepdims=True)

# -----------------------------
# Inference in chunks
# -----------------------------

chunk = args.chunk_size
pred_all = np.zeros(N, dtype=np.int64)

with torch.no_grad():
    start = 0
    while start < N:
        end = min(N, start + chunk)
        pts_chunk = xyz_centered[start:end]  # (P,3)
        pts_t = torch.from_numpy(pts_chunk).unsqueeze(0).to(DEVICE)  # (1,P,3)

        logits = model(pts_t)       # (1,P,3)
        pred = logits.argmax(-1).squeeze(0).cpu().numpy()  # (P,)

        pred_all[start:end] = pred
        start = end

print("Inference done.")

# -----------------------------
# Save PLY with float pred_label
# -----------------------------

dtype = [
    ("x", "f4"),
    ("y", "f4"),
    ("z", "f4"),
    ("red", "u1"),
    ("green", "u1"),
    ("blue", "u1"),
    ("pred_label", "f4"),
]

elements = np.empty(N, dtype=dtype)
elements["x"] = xyz[:, 0]
elements["y"] = xyz[:, 1]
elements["z"] = xyz[:, 2]
elements["red"] = colors[:, 0]
elements["green"] = colors[:, 1]
elements["blue"] = colors[:, 2]
elements["pred_label"] = pred_all.astype(np.float32)

ply_el = PlyElement.describe(elements, "vertex")
PlyData([ply_el], text=True).write(args.output)

print("Saved:", args.output)
