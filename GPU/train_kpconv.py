import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from class_weights import compute_class_weights
from kpconv import KPNetReal
from dataset_kpconv import build_dataloaders, POINTS_PER_SAMPLE
from metrics import confusion_matrix, iou_from_confmat


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 3

# ==== TRAINING CONFIG ====
BATCH_SIZE = 1
EPOCHS = 30 
LR = 5e-4 
NUM_WORKERS = 4
TRAIN_RATIO = 0.8

SAVE_DIR = "./checkpoints_kpconv"
os.makedirs(SAVE_DIR, exist_ok=True)

LOG_FILE = os.path.join(SAVE_DIR, "training_log.csv")

# Create CSV header if not exists
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        f.write("epoch,train_loss,val_loss,iou0,iou1,iou2,miou\n")


def train_one_epoch(model, loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}", ncols=100)
    for xyz, labels in pbar:
        xyz = xyz.to(DEVICE)       # (B, P, 3)
        labels = labels.to(DEVICE) # (B, P)

        optimizer.zero_grad()
        logits = model(xyz)        # (B, P, C)

        loss = criterion(
            logits.reshape(-1, NUM_CLASSES),
            labels.view(-1)
        )
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * xyz.size(0)
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = running_loss / len(loader.dataset)
    return avg_loss


def evaluate(model, loader, criterion, epoch):
    model.eval()
    running_loss = 0.0
    confmat = torch.zeros(NUM_CLASSES, NUM_CLASSES, device=DEVICE, dtype=torch.int64)

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Val Epoch {epoch}", ncols=100)
        for xyz, labels in pbar:
            xyz = xyz.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(xyz)  # (B, P, C)

            loss = criterion(
                logits.reshape(-1, NUM_CLASSES),
                labels.view(-1)
            )
            running_loss += loss.item() * xyz.size(0)

            preds = torch.argmax(logits, dim=-1)  # (B, P)
            confmat += confusion_matrix(preds.view(-1), labels.view(-1), NUM_CLASSES)

    avg_loss = running_loss / len(loader.dataset)
    ious, miou = iou_from_confmat(confmat)
    return avg_loss, ious.detach().cpu().numpy(), miou.item()


def main():
    print("Building dataloaders...")
    train_loader, val_loader = build_dataloaders(
        batch_size=BATCH_SIZE,
        points_per_sample=POINTS_PER_SAMPLE,
        num_workers=NUM_WORKERS,
        train_ratio=TRAIN_RATIO,
    )

    print("Computing class weights...")
    class_weights = compute_class_weights(num_classes=NUM_CLASSES)
    class_weights = class_weights.to(DEVICE)

    print("Creating model...")
    model = KPNetReal(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_miou = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, epoch)
        val_loss, ious, miou = evaluate(model, val_loader, criterion, epoch)

        print(f"\nEpoch {epoch}/{EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val   Loss: {val_loss:.4f}")
        print(f"  IoU: {ious}")
        print(f"  mIoU: {miou:.4f}")

        # ---- SAVE LOGS EVERY EPOCH ----
        with open(LOG_FILE, "a") as f:
            f.write(
                f"{epoch},{train_loss:.4f},{val_loss:.4f},"
                f"{ious[0]:.4f},{ious[1]:.4f},{ious[2]:.4f},{miou:.4f}\n"
            )


        # Save best model
        if miou > best_miou:
            best_miou = miou
            ckpt_path = os.path.join(SAVE_DIR, f"kpconv_best_miou_{best_miou:.4f}.pth")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "miou": best_miou,
            }, ckpt_path)
            print(f"  ✅ New best mIoU, saved to {ckpt_path}")


if __name__ == "__main__":
    main()
