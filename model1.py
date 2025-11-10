# train_roi_seg.py
# Windows-safe PyTorch DataLoader (num_workers>0) + ROI dataset training

import os, glob, math, random
import numpy as np
import pandas as pd
from plyfile import PlyData

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ---------------------------
# Repro / cuDNN
# ---------------------------
seed = 42
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = True  # autotune


# ---------------------------
# Dataset
# ---------------------------
class Rail3DDataset(Dataset):
    def __init__(self, file_list, num_points=4096):
        self.file_list = file_list
        self.num_points = num_points
        self.class_map = {"Background": 0, "Track": 1, "Object": 2}
        self.scalar_map = {1: "Background", 3: "Track", 9: "Object"}

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        ply_path = self.file_list[idx]
        try:
            plydata = PlyData.read(ply_path)
            df = pd.DataFrame(plydata['vertex'].data)

            # Points
            if {'x','y','z'}.issubset(df.columns):
                points = df[['x','y','z']].values.astype(np.float32)
            else:
                raise ValueError("Missing x/y/z columns")

            # Labels
            col = None
            for cand in ["scalar_NewClassification", "scalar_Classification", "classification", "label"]:
                if cand in df.columns:
                    col = cand; break
            if col is None:
                raise ValueError("No classification column found")

            try:
                raw = df[col].astype(int)
            except Exception:
                raise ValueError("Cannot cast labels to int")

            mapped = raw.map(self.scalar_map)
            if mapped.isnull().any():
                # map unknowns to BG (warn once per sample)
                mapped = mapped.fillna("Background")

            labels = mapped.map(self.class_map).astype(np.int64).values

            # Empty safeguard
            if points.shape[0] == 0:
                points = np.zeros((self.num_points, 3), dtype=np.float32)
                labels = np.zeros((self.num_points,), dtype=np.int64)
                return torch.from_numpy(points).float(), torch.from_numpy(labels).long()

            # --------- Per-point sampling (safer to avoid collapse) ---------
            # frequency per class in this cloud
            unique, counts = np.unique(labels, return_counts=True)
            freq = {int(u): int(c) for u, c in zip(unique, counts)}

            # inverse-frequency per point
            inv = np.array([1.0 / max(freq.get(int(l), 1), 1) for l in labels], dtype=np.float32)

            # mild Track boost, stronger Object boost; avoid over-pushing Track
            boost = {0: 1.0, 1: 1.2, 2: 2.0}
            inv *= np.vectorize(boost.get)(labels.astype(int))

            # mix with uniform (higher alpha = safer)
            alpha = 0.6
            uniform = np.full_like(inv, 1.0 / len(inv), dtype=np.float32)
            inv_sum = float(inv.sum())
            if not np.isfinite(inv_sum) or inv_sum <= 0:
                raise ValueError("Bad sampling weights")
            p = (1 - alpha) * (inv / inv_sum) + alpha * uniform
            p = p / p.sum()

            # sample
            choice = np.random.choice(len(points), self.num_points, replace=True, p=p)
            points = points[choice]
            labels = labels[choice]

            # normalize + jitter
            centroid = points.mean(axis=0, keepdims=True)
            points = points - centroid
            max_dist = np.sqrt((points**2).sum(axis=1)).max()
            if max_dist > 0:
                points = points / max_dist
            points = points + np.random.normal(scale=0.001, size=points.shape).astype(np.float32)

            return torch.from_numpy(points).float(), torch.from_numpy(labels).long()

        except Exception as e:
            print(f"[DATASET ERROR] while loading: {os.path.basename(ply_path)} -> {e}")
            raise


# ---------------------------
# Simple model (PointNet-ish MLP)
# ---------------------------
class SimplePointNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, num_classes)

    def forward(self, x):
        B, N, _ = x.shape
        x = x.view(B * N, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = x.view(B, N, -1)
        return x


# ---------------------------
# Utilities
# ---------------------------
def iou_per_class(preds, labels, num_classes=3):
    preds = preds.view(-1)
    labels = labels.view(-1)
    ious = []
    for c in range(num_classes):
        inter = ((preds == c) & (labels == c)).sum().item()
        union = ((preds == c) | (labels == c)).sum().item()
        ious.append(float('nan') if union == 0 else inter / union)
    return ious

def count_labels_from_files(file_list, scalar_map, class_map):
    from collections import Counter
    cnt = Counter()
    for path in file_list:
        ply = PlyData.read(path)
        df = pd.DataFrame(ply['vertex'].data)
        col = None
        for cand in ["scalar_NewClassification", "scalar_Classification", "classification", "label"]:
            if cand in df.columns:
                col = cand; break
        if col is None:
            continue
        raw = df[col].astype(int)
        mapped = raw.map(scalar_map).map(class_map).dropna().astype(int)
        cnt.update(mapped.tolist())
    return cnt

def worker_init_fn(worker_id):
    # Make workers deterministic and less noisy on Windows
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)
    torch.manual_seed(seed + worker_id)


# ---------------------------
# Main (Windows-safe)
# ---------------------------
def main():
    # ---- paths ----
    data_folder = r"C:\Users\bhanu\OneDrive\Desktop\capstone\Data\train_sphere_ascii_roi"
    file_list = sorted(glob.glob(os.path.join(data_folder, "*.ply")))
    print(f"Found {len(file_list)} PLY files for training!")

    # ---- dataset / dataloader ----
    dataset = Rail3DDataset(file_list, num_points=4096)

    # set >0 workers only in .py with spawn-safe guard
    num_workers = 4 if len(file_list) > 1 else 0
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(torch.cuda.is_available() and num_workers > 0),
        persistent_workers=(num_workers > 0),
        prefetch_factor=(2 if num_workers > 0 else None),
        worker_init_fn=(worker_init_fn if num_workers > 0 else None)
    )

    # ---- device/model ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 3
    model = SimplePointNet(num_classes).to(device)

    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    model.apply(init_weights)

    # ---- class weights from current files ----
    full_cnt = count_labels_from_files(file_list,
                                       {1: "Background", 3: "Track", 9: "Object"},
                                       {"Background": 0, "Track": 1, "Object": 2})
    print("Full dataset counts (current):", dict(full_cnt))
    tot = sum(full_cnt.values()) if full_cnt else 1
    raw = np.array([tot / max(full_cnt.get(c, 1), 1) for c in range(num_classes)], dtype=np.float32)
    w = np.sqrt(raw); w = w / w.mean(); w = np.clip(w, 0.5, 5.0)
    weights = torch.tensor(w, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    print("Using smoothed class weights (current):", w)

    # ---- optim/amp/sched ----
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    # ---- training ----
    epochs = 20
    save_dir = "./checkpoints"; os.makedirs(save_dir, exist_ok=True)

    def epoch_iou(model, dataloader, device, num_classes=3):
        inter = np.zeros(num_classes, dtype=np.int64)
        union = np.zeros(num_classes, dtype=np.int64)
        model.eval()
        with torch.no_grad():
            for pts, lbls in dataloader:
                pts = pts.to(device)
                lbls = lbls.to(device)
                preds = model(pts).argmax(dim=-1)
                for c in range(num_classes):
                    p = (preds == c); l = (lbls == c)
                    inter[c] += (p & l).sum().item()
                    union[c] += (p | l).sum().item()
        return [(inter[c] / union[c]) if union[c] > 0 else float('nan') for c in range(num_classes)]

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0; correct = 0; total_pts = 0

        for pts, lbls in dataloader:
            pts = pts.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                outputs = model(pts)
                loss = criterion(outputs.view(-1, num_classes), lbls.view(-1))

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * pts.size(0)
            preds = outputs.argmax(dim=-1)
            correct += (preds == lbls).sum().item()
            total_pts += lbls.numel()

        avg_loss = total_loss / len(dataset)
        accuracy = correct / total_pts * 100
        ious = epoch_iou(model, dataloader, device, num_classes)

        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f} Acc: {accuracy:.2f}% IoU: {ious}")
        scheduler.step(avg_loss)

        torch.save({
            'epoch': epoch + 1,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }, os.path.join(save_dir, f"model_epoch{epoch+1}.pth"))

    print("Training finished.")


if __name__ == "__main__":
    # IMPORTANT on Windows: spawn start method for DataLoader workers
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
