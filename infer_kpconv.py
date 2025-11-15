import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from plyfile import PlyData, PlyElement

# ---------- MODEL DEFINITION (same as training) ----------

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
            nn.Linear(out_c, out_c)
        )

    def forward(self, pts, feats):
        """
        pts:   (B, P, 3)
        feats: (B, P, C)
        output: (B, P, out_c)
        """
        B, P, C = feats.shape
        K = self.K

        idx = knn(pts, K)  # (B, P, K)

        # gather neighbor coords: (B, P, K, 3)
        neigh_pts = torch.gather(
            pts.unsqueeze(1).expand(B, P, P, 3),
            2,
            idx.unsqueeze(-1).expand(B, P, K, 3),
        )

        # gather neighbor feats: (B, P, K, C)
        feats_expand = feats.unsqueeze(1).expand(B, P, P, C)
        neigh_feats = torch.gather(
            feats_expand,
            2,
            idx.unsqueeze(-1).expand(B, P, K, C),
        )

        # relative positions
        pts_expand = pts.unsqueeze(2).expand(B, P, K, 3)
        rel = neigh_pts - pts_expand  # (B,P,K,3)

        # concat (B,P,K, 3+C)
        inp = torch.cat([rel, neigh_feats], dim=-1)

        out = self.mlp(inp)         # (B,P,K,out_c)
        out = out.max(dim=2)[0]     # (B,P,out_c)
        return out

class KPNet(nn.Module):
    def __init__(self, base=8, num_classes=3, K=6):
        super().__init__()
        self.fc0 = nn.Linear(3, base)
        self.kp1 = KPConvLayer(base, base*2, K)
        self.kp2 = KPConvLayer(base*2, base*4, K)
        self.head = nn.Sequential(
            nn.Linear(base*4, base*4),
            nn.ReLU(),
            nn.Linear(base*4, num_classes)
        )

    def forward(self, pts):
        feats = F.relu(self.fc0(pts))
        feats = self.kp1(pts, feats)
        feats = self.kp2(pts, feats)
        out = self.head(feats)
        return out

# ---------- ARGUMENT PARSING ----------

parser = argparse.ArgumentParser(description="KPConv inference on a PLY tile")
parser.add_argument("--input", required=True, help="Input .ply file path")
parser.add_argument("--output", required=True, help="Output .ply file path")
parser.add_argument("--checkpoint", default="checkpoints/best_model.pth",
                    help="Path to trained model checkpoint")
parser.add_argument("--chunk_size", type=int, default=1024,
                    help="Number of points per chunk for inference")
args = parser.parse_args()

# ---------- DEVICE ----------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# ---------- LOAD MODEL ----------

model = KPNet(base=8, num_classes=3, K=6).to(DEVICE)
print("Loading checkpoint:", args.checkpoint)
state = torch.load(args.checkpoint, map_location=DEVICE)
model.load_state_dict(state)
model.eval()
print("Model loaded.")

# ---------- LOAD PLY ----------

print("Reading input PLY:", args.input)
pd = PlyData.read(args.input)
v = pd["vertex"].data

xyz = np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float32)   # (N,3)
N = xyz.shape[0]
print("Num points in cloud:", N)

# center (same style as training)
xyz_centered = xyz - xyz.mean(0, keepdims=True)

# ---------- INFERENCE (CHUNKED) ----------

chunk_size = args.chunk_size
preds = np.zeros(N, dtype=np.int64)

print("Running inference in chunks of", chunk_size, "points...")
with torch.no_grad():
    start = 0
    while start < N:
        end = min(N, start + chunk_size)
        pts_chunk = xyz_centered[start:end]       # (P,3)
        pts_tensor = torch.from_numpy(pts_chunk).unsqueeze(0).to(DEVICE)  # (1,P,3)

        logits = model(pts_tensor)         # (1,P,3)
        pred_chunk = logits.argmax(-1).squeeze(0).cpu().numpy()  # (P,)

        preds[start:end] = pred_chunk
        start = end

print("Inference done.")

# ---------- MAP PREDICTIONS TO COLORS ----------

# class 0 -> green
# class 1 -> red
# class 2 -> blue
colors = np.zeros((N, 3), dtype=np.uint8)
palette = {
    0: (0, 255, 0),
    1: (255, 0, 0),
    2: (0, 0, 255),
}
for c, rgb in palette.items():
    colors[preds == c] = rgb

# ---------- BUILD OUTPUT PLY ----------

vertex_dtype = [
    ("x", "f4"),
    ("y", "f4"),
    ("z", "f4"),
    ("red", "u1"),
    ("green", "u1"),
    ("blue", "u1"),
    ("pred_label", "u1"),
]

vertex = np.empty(N, dtype=vertex_dtype)
vertex["x"] = xyz[:, 0]
vertex["y"] = xyz[:, 1]
vertex["z"] = xyz[:, 2]
vertex["red"] = colors[:, 0]
vertex["green"] = colors[:, 1]
vertex["blue"] = colors[:, 2]
vertex["pred_label"] = preds.astype(np.uint8)

ply_out = PlyData([PlyElement.describe(vertex, "vertex")], text=True)
os.makedirs(os.path.dirname(args.output), exist_ok=True)
ply_out.write(args.output)

print("Saved colored prediction PLY to:", args.output)
