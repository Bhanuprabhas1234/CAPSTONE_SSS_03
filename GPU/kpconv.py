import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def knn_indices(xyz, k):
    """
    xyz: (B, N, 3)
    returns idx: (B, N, k) with indices of k nearest neighbors for each point.
    """
    with torch.no_grad():
        # pairwise distance (B, N, N)
        dist = torch.cdist(xyz, xyz)  # uses CUDA, efficient for N=1024
        idx = dist.topk(k, largest=False).indices  # (B, N, k)
    return idx


def sample_kernel_points(num_kpoints, radius, device):
    """
    Sample kernel points inside a sphere of given radius.
    Very simple: random points then normalize to lie within radius.
    """
    kp = torch.randn(num_kpoints, 3, device=device)
    # Normalize to max norm <= radius
    norms = kp.norm(dim=1, keepdim=True) + 1e-6
    kp = kp / norms * (torch.rand_like(norms) * radius)
    return kp


class KPConv(nn.Module):
    """
    Simplified KPConv (pointwise) layer in PyTorch.

    - Uses kNN to get local neighborhood
    - Uses kernel points inside a ball of radius R
    - Uses distance-based weighting similar to KPConv
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_kernel_points=15,
        k_neighbors=16,
        radius=0.2,
        sigma=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernel_points = num_kernel_points
        self.k_neighbors = k_neighbors
        self.radius = radius
        self.sigma = sigma if sigma is not None else radius / 3.0

        # One weight matrix per kernel point: (K, Fin, Fout)
        self.kernel_weights = nn.Parameter(
            torch.randn(num_kernel_points, in_channels, out_channels) * math.sqrt(2.0 / in_channels)
        )

        # Kernel points initialized later on first forward (depends on device)
        self.register_buffer("kernel_points", torch.empty(0))

        self.bias = nn.Parameter(torch.zeros(out_channels))

    def maybe_init_kernel_points(self, device):
        if self.kernel_points.numel() == 0:
            kp = sample_kernel_points(self.num_kernel_points, self.radius, device)
            self.kernel_points = kp

    def forward(self, xyz, features):
        """
        xyz:      (B, N, 3)
        features: (B, N, Fin)
        returns:  (B, N, Fout)
        """
        B, N, _ = xyz.shape
        _, _, Fin = features.shape
        assert Fin == self.in_channels, "Input feature dim mismatch"

        device = xyz.device
        self.maybe_init_kernel_points(device)  # (K, 3)

        # 1) Neighborhoods: (B, N, k)
        idx = knn_indices(xyz, self.k_neighbors)

        # 2) Gather neighbor positions and features
        batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand_as(idx)
        neigh_xyz = xyz[batch_idx, idx]      # (B, N, k, 3)
        neigh_feat = features[batch_idx, idx]  # (B, N, k, Fin)

        # 3) Relative positions to center point
        center_xyz = xyz.unsqueeze(2)  # (B, N, 1, 3)
        rel_pos = neigh_xyz - center_xyz  # (B, N, k, 3)

        # 4) Distances to kernel points
        # rel_pos:      (B, N, k, 1, 3)
        # kernel_points:(1, 1, 1, K, 3)
        kp = self.kernel_points.view(1, 1, 1, self.num_kernel_points, 3)
        rel = rel_pos.unsqueeze(3) - kp  # (B, N, k, K, 3)
        dist = torch.norm(rel, dim=-1)   # (B, N, k, K)

        # 5) Influence weights (like KPConv: (1 - d/sigma)^2_+ )
        influence = torch.clamp(1.0 - dist / self.sigma, min=0.0) ** 2  # (B, N, k, K)

        # Optional: mask neighbors outside radius (for closer to ball query)
        if self.radius is not None:
            mask = (torch.norm(rel_pos, dim=-1, keepdim=True) <= self.radius).float()
            # mask: (B, N, k, 1) → broadcast
            influence = influence * mask  # zero-out contributions outside radius

        # Normalize over neighbors to avoid exploding sums
        # Sum over k-neighbors dimension
        neigh_sum = influence.sum(dim=2, keepdim=True) + 1e-8
        influence = influence / neigh_sum  # (B, N, k, K)

        # 6) Aggregate features using kernel weights
        # We'll loop over kernel points (K is small, e.g., 15)
        out = torch.zeros(B, N, self.out_channels, device=device)

        for k in range(self.num_kernel_points):
            # influence for kernel k: (B, N, k_neighbors)
            infl_k = influence[:, :, :, k]  # (B, N, k)
            infl_k = infl_k.unsqueeze(-1)   # (B, N, k, 1)

            # weighted neighbor features: sum over neighbors
            # neigh_feat: (B, N, k, Fin)
            weighted_feats = (neigh_feat * infl_k).sum(dim=2)  # (B, N, Fin)

            # apply kernel weight for this kernel point: (Fin, Fout)
            Wk = self.kernel_weights[k]  # (Fin, Fout)
            contrib = weighted_feats @ Wk  # (B, N, Fout)

            out += contrib

        out = out / self.num_kernel_points   # average across kernels
        out = out + self.bias
        return out

class KPNetReal(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        self.fc_in = nn.Linear(3, 16)

        self.kpconv1 = KPConv(
            in_channels=16,
            out_channels=32,
            num_kernel_points=15,
            k_neighbors=24,
            radius=0.5,
        )

        self.kpconv2 = KPConv(
            in_channels=32,
            out_channels=64,
            num_kernel_points=15,
            k_neighbors=32,
            radius=0.8,
        )

        self.fc_mid = nn.Linear(64, 64)
        self.fc_out = nn.Linear(64, num_classes)

    def forward(self, xyz):
        x = torch.relu(self.fc_in(xyz))     # (B,N,16)
        x = torch.relu(self.kpconv1(xyz, x))# (B,N,32)
        x = torch.relu(self.kpconv2(xyz, x))# (B,N,64)
        x = torch.relu(self.fc_mid(x))      # (B,N,64)
        logits = self.fc_out(x)             # (B,N,3)
        return logits
 
 