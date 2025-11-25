import torch
from collections import Counter
from dataset_kpconv import build_dataloaders, POINTS_PER_SAMPLE, DATA_ROOT


def compute_class_weights(num_classes=3, samples_to_scan=50):
    """
    Scan a subset of the training data to estimate class frequencies,
    then compute inverse-frequency weights.
    """
    train_loader, _ = build_dataloaders(
        batch_size=1,
        points_per_sample=POINTS_PER_SAMPLE,
        num_workers=2,
        train_ratio=0.8,
    )

    counts = Counter()
    total_seen = 0

    for i, (xyz, labels) in enumerate(train_loader):
        labels = labels.view(-1).numpy().tolist()
        counts.update(labels)
        total_seen += len(labels)
        if i + 1 >= samples_to_scan:
            break

    print("Class counts (approx):", counts)

    freqs = torch.zeros(num_classes, dtype=torch.float32)
    for c in range(num_classes):
        freqs[c] = counts.get(c, 1)

    weights = 1.0 / (freqs + 1e-6)
    weights = weights / weights.sum() * num_classes  # normalize a bit
    print("Class weights:", weights)
    return weights
