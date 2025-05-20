import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.models as models
from client_node.RGB_track_dataset import RGBTrackDataset
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import random


class RGBTrackModel(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(base.features[:14]))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(96, 1)

        # Freeze early layers (0–7), unfreeze higher
        for i, block in enumerate(self.encoder):
            for param in block.parameters():
                param.requires_grad = i >= 7

    def forward(self, x):
        x = self.encoder(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


# === PLOT TRAINING CURVES ===
def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(7, 5))
    plt.plot(train_losses, label="Train L1")
    plt.plot(val_losses, label="Val L1")
    plt.xlabel("Epoch")
    plt.ylabel("L1 Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        "/ros_bridge/src/client_node/client_node/train/checkpoints/loss_curve.png"
    )
    plt.close()


# === REGRESSION VISUALIZATION: scatter of prediction vs target ===
def plot_pred_vs_true(model, dl, device, save_path):
    model.eval()
    all_preds, all_gt = [], []
    with torch.no_grad():
        for imgs, steers in dl:
            preds = model(imgs.to(device)).cpu().squeeze().numpy()
            all_preds.extend(preds)
            all_gt.extend(steers.squeeze().numpy())
    plt.figure(figsize=(6, 6))
    plt.scatter(all_gt, all_preds, alpha=0.4)
    plt.plot([-1, 1], [-1, 1], "--r")
    plt.xlabel("True steer")
    plt.ylabel("Predicted steer")
    plt.title("Prediction vs Ground Truth")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def calculate_accuracy(y_true, y_pred, thresholds=[0.05, 0.1, 0.2]):
    """Calculate accuracy as percentage of predictions within various error thresholds"""
    results = {}
    for threshold in thresholds:
        correct = sum(
            1 for true, pred in zip(y_true, y_pred) if abs(true - pred) < threshold
        )
        accuracy = correct / len(y_true) * 100
        results[f"acc@{threshold:.2f}"] = accuracy
    return results


def bin_steer(s):
    if s < -0.25:
        return 0  # sharp left
    elif s < -0.05:
        return 1  # slight left
    elif s < 0.05:
        return 2  # straight
    elif s < 0.25:
        return 3  # slight right
    else:
        return 4  # sharp right


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Training on: {device}")
    train_ds = RGBTrackDataset(split="train")
    val_ds = RGBTrackDataset(split="val")
    train_dl = DataLoader(
        train_ds, batch_size=32, shuffle=True, num_workers=2, pin_memory=True
    )
    val_dl = DataLoader(
        val_ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True
    )

    model = RGBTrackModel().to(device)
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4)
    loss_fn = nn.L1Loss()

    best_val_loss = float("inf")
    os.makedirs(
        "/ros_bridge/src/client_node/client_node/train/checkpoints", exist_ok=True
    )
    train_losses = []
    val_losses = []
    val_acc_history = {f"acc@{t:.2f}": [] for t in [0.05, 0.1, 0.2]}

    for epoch in range(10):
        model.train()
        total = 0
        for imgs, steers in train_dl:
            imgs, steers = imgs.to(device), steers.to(device)
            pred = model(imgs)
            loss = loss_fn(pred, steers)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"Epoch {epoch} ⏿ train L1 = {total/len(train_dl):.4f}")

        # Evaluate on val set
        model.eval()
        with torch.no_grad():
            val_total = 0
            val_true, val_pred = [], []
            for imgs, steers in val_dl:
                pred = model(imgs.to(device))
                loss = loss_fn(pred, steers.to(device))
                val_total += loss.item()
                val_true.extend(steers.squeeze().numpy())
                val_pred.extend(pred.cpu().squeeze().numpy())

            val_loss = val_total / len(val_dl)
            val_acc = calculate_accuracy(val_true, val_pred)
            for k, v in val_acc.items():
                val_acc_history[k].append(v)
            print(
                f"val L1 = {val_loss:.4f} | "
                + " | ".join(f"{k}: {v:.1f}%" for k, v in val_acc.items())
            )

        # Save if improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                model.encoder.state_dict(),
                "/ros_bridge/src/client_node/client_node/train/checkpoints/mobilenet_trackslice14.pth",
            )
            print("✅ Saved better encoder.")

        train_losses.append(total / len(train_dl))
        val_losses.append(val_loss)

    print("\nTesting...")
    test_ds = RGBTrackDataset(split="test")
    test_dl = DataLoader(test_ds, batch_size=32)
    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_total = 0
        test_true, test_pred = [], []
        for imgs, steers in test_dl:
            pred = model(imgs.to(device))
            loss = loss_fn(pred, steers.to(device))
            test_total += loss.item()
            test_true.extend(steers.squeeze().numpy())
            test_pred.extend(pred.cpu().squeeze().numpy())

        test_loss = test_total / len(test_dl)
        test_acc = calculate_accuracy(test_true, test_pred)
        print(
            f"test L1 = {test_loss:.4f} | "
            + " | ".join(f"{k}: {v:.1f}%" for k, v in test_acc.items())
        )

    # === Visualizations ===
    plot_loss(train_losses, val_losses)
    plot_pred_vs_true(
        model,
        DataLoader(RGBTrackDataset(split="test"), batch_size=32),
        device,
        "/ros_bridge/src/client_node/client_node/train/checkpoints/test_scatter.png",
    )

    # === Classification-style Report ===
    target_names = ["Sharp Left", "Left", "Straight", "Right", "Sharp Right"]
    true_binned = [bin_steer(s) for s in test_true]
    pred_binned = [bin_steer(s) for s in test_pred]

    print("\n=== Classification Report (Binned Steering) ===")
    print(
        classification_report(
            true_binned, pred_binned, target_names=target_names, zero_division=0
        )
    )

    cm = confusion_matrix(true_binned, pred_binned)
    print("Confusion Matrix:")
    print(cm)

    # Confusion Matrix Plot
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Steering Class Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(
        "/ros_bridge/src/client_node/client_node/train/checkpoints/confusion_matrix.png"
    )
    plt.close()

    print("✅ Done.")


if __name__ == "__main__":
    main()
