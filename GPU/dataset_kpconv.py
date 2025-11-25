import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from plyfile import PlyData


# ---- CONFIG: EDIT THIS TO YOUR DATA FOLDER ----
DATA_ROOT = "/home/ccbd/Desktop/SSS_03/Data/train_sphere_ascii_roi"  # TODO: change this
POINTS_PER_SAMPLE = 1024

# Label mapping as per your project:
# scalar_NewClassification:
#   1 -> 0 (track)
#   3 -> 1 (non-track)
#   9 -> 2 (object)
LABEL_MAP = {1: 0, 3: 1, 9: 2}


def normalize_xyz(xyz):
    """
    Center to mean 0 and scale to fit roughly in unit sphere.
    xyz: (N, 3)
    """
    xyz = xyz.astype(np.float32)
    center = xyz.mean(axis=0, keepdims=True)          # (1, 3)
    xyz_centered = xyz - center
    norms = np.linalg.norm(xyz_centered, axis=1)
    max_norm = np.max(norms) + 1e-6
    xyz_normalized = xyz_centered / max_norm
    return xyz_normalized

def read_ply_with_labels(path):
    """
    Reads a .ply and returns:
      xyz:   (N, 3)
      labels:(N,)
    Uses 'scalar_NewClassification' as the label field.
    """
    ply = PlyData.read(path)
    v = ply["vertex"]

    x = np.asarray(v["x"], dtype=np.float32)
    y = np.asarray(v["y"], dtype=np.float32)
    z = np.asarray(v["z"], dtype=np.float32)

    # Change this key if your field name is slightly different
    raw_labels = np.asarray(v["scalar_NewClassification"], dtype=np.int32)

    # Map 1,3,9 -> 0,1,2
    mapped = np.full_like(raw_labels, fill_value=-1, dtype=np.int64)
    for k, v2 in LABEL_MAP.items():
        mapped[raw_labels == k] = v2

    # Filter only valid labels (0,1,2)
    mask = mapped >= 0
    xyz = np.stack([x[mask], y[mask], z[mask]], axis=1)  # (N, 3)
    xyz = normalize_xyz(xyz)                             # <<< normalize here
    labels = mapped[mask]  # (N,)

    return xyz, labels


class RailwayPointCloudDataset(Dataset):
    """
    Each item:
      xyz:    (POINTS_PER_SAMPLE, 3) float32
      labels: (POINTS_PER_SAMPLE,)   int64 in {0,1,2}
    """

    def __init__(self, file_list, points_per_sample=POINTS_PER_SAMPLE, augment=True):
        self.files = file_list
        self.points_per_sample = points_per_sample
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def _balanced_sample(self, xyz, labels):
        """
        Try to sample a more balanced set of points for each class (0,1,2).
        If a class has too few points, we oversample it.
        """
        num_points = self.points_per_sample

        classes = [0, 1, 2]
        # target fractions inside each patch – TUNE THESE
        target_fracs = {0: 0.4, 1: 0.1, 2: 0.5}  # more weight on objects

        idx_per_class = {c: (labels == c).nonzero()[0] for c in classes}

        chosen_indices = []

        for c in classes:
            idx_c = idx_per_class[c]
            if len(idx_c) == 0:
                continue

            target_n = int(num_points * target_fracs[c])

            if len(idx_c) >= target_n:
                chosen = np.random.choice(idx_c, target_n, replace=False)
            else:
                chosen = np.random.choice(idx_c, target_n, replace=True)

            chosen_indices.append(chosen)

        if len(chosen_indices) == 0:
            # extreme fallback
            all_idx = np.arange(len(labels))
            chosen = np.random.choice(all_idx, num_points, replace=len(labels) < num_points)
            return xyz[chosen], labels[chosen]

        chosen_indices = np.concatenate(chosen_indices, axis=0)

        # If we got more than needed, trim
        if len(chosen_indices) > num_points:
            chosen_indices = np.random.choice(chosen_indices, num_points, replace=False)

        # If less than needed, fill with random
        if len(chosen_indices) < num_points:
            remaining = num_points - len(chosen_indices)
            all_idx = np.arange(len(labels))
            extra = np.random.choice(all_idx, remaining, replace=len(all_idx) < remaining)
            chosen_indices = np.concatenate([chosen_indices, extra], axis=0)

        return xyz[chosen_indices], labels[chosen_indices]


    def _augment(self, xyz):
        # Simple augmentation: random small jitter + random rotation around Z
        xyz = xyz.copy()
        # jitter
        xyz += np.random.normal(scale=0.01, size=xyz.shape).astype(np.float32)

        # rotation around Z
        theta = np.random.uniform(0, 2 * np.pi)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s, 0],
                      [s,  c, 0],
                      [0,  0, 1]], dtype=np.float32)
        xyz = xyz @ R.T
        return xyz

    def __getitem__(self, idx):
        path = self.files[idx]
        xyz, labels = read_ply_with_labels(path)

        xyz, labels = self._balanced_sample(xyz, labels)

        if self.augment:
            xyz = self._augment(xyz)

        xyz = torch.from_numpy(xyz).float()        # (P, 3)
        labels = torch.from_numpy(labels).long()   # (P,)

        return xyz, labels


def build_file_lists(root=DATA_ROOT, train_ratio=0.8, seed=42):
    """
    Scans the DATA_ROOT for .ply files, splits into train/val lists.
    """
    all_files = sorted(glob.glob(os.path.join(root, "*.ply")))
    if len(all_files) == 0:
        raise RuntimeError(f"No .ply files found in {root}")

    from sklearn.model_selection import train_test_split
    train_files, val_files = train_test_split(
        all_files, test_size=1 - train_ratio, random_state=seed, shuffle=True
    )
    return train_files, val_files


def build_dataloaders(
    batch_size=1,
    points_per_sample=POINTS_PER_SAMPLE,
    num_workers=4,
    train_ratio=0.8,
    seed=42,
):
    train_files, val_files = build_file_lists(DATA_ROOT, train_ratio, seed)

    train_ds = RailwayPointCloudDataset(
        train_files, points_per_sample=points_per_sample, augment=True
    )
    val_ds = RailwayPointCloudDataset(
        val_files, points_per_sample=points_per_sample, augment=False
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader
