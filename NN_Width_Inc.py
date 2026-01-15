import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# =========================
# settings
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data.csv"          # data.csv is in project root
OUT_MODEL = BASE_DIR / "best_surrogate.pt" # Path-safe

SEED = 0
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1  # test = 0.1

BATCH_SIZE = 1024
EPOCHS = 500
LR = 1e-3
WEIGHT_DECAY = 1e-6
PATIENCE = 50

# Network architecture
WIDTH = 256          #No of neurons
DEPTH = 4            # total number of layers (including output layer)
ACTIVATION = "tanh"  

# Column names (according to data.csv)
INPUT_COLS = ["x", "y", "C10", "C01", "C20", "invD"]
TARGET_COLS = ["ux", "uy"]

# =========================
# Utilities
# =========================
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_activation(name: str):
    name = name.lower()
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unknown activation: {name}")

# =========================
# Dataset
# =========================
class TabDataset(Dataset):
    def __init__(self, df: pd.DataFrame, X_mean, X_std, y_mean, y_std):
        X = df[INPUT_COLS].values.astype(np.float32)
        y = df[TARGET_COLS].values.astype(np.float32)

        self.X = (X - X_mean) / X_std
        self.y = (y - y_mean) / y_std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])

# =========================
# Model
# =========================
class MLP(nn.Module):
    def __init__(self, in_dim=6, out_dim=2, width=256, depth=5, activation="tanh"):
        super().__init__()
        layers = [nn.Linear(in_dim, width), get_activation(activation)]

        # hidden layers (depth includes output layer)
        for _ in range(depth - 2):
            layers += [nn.Linear(width, width), get_activation(activation)]

        layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# =========================
# Main workflow
# =========================
def main():
    set_seed(SEED)

    # Use non-interactive backend in container (prevents plt.show() issues)
    matplotlib.use("Agg")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # 1) Load data (ONLY here, not at top-level)
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Could not find data file: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)

    required = set(INPUT_COLS + TARGET_COLS + ["group_id"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"data.csv missing columns: {sorted(missing)}")

    # ========= Use only a subset of group_id (for faster debugging) =========
    np.random.seed(SEED)
    all_groups = np.sort(df["group_id"].unique())
    pick_groups = all_groups[:20]  # group_id 0~19 (or first 20 unique)

    df = df[df["group_id"].isin(pick_groups)].reset_index(drop=True)

    print("Using groups:", pick_groups)
    print("Rows after filtering:", len(df))

    # 2) Split by group_id (avoid leakage)
    groups = df["group_id"].unique()
    np.random.seed(SEED)
    np.random.shuffle(groups)

    n = len(groups)
    if n < 3:
        raise ValueError(f"Not enough unique groups after filtering: {n}. Need at least 3.")

    n_train = max(1, int(TRAIN_RATIO * n))
    n_val = max(1, int(VAL_RATIO * n))

    # Ensure we don't allocate more than available
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)

    train_groups = set(groups[:n_train])
    val_groups   = set(groups[n_train:n_train + n_val])
    test_groups  = set(groups[n_train + n_val:])

    train_df = df[df["group_id"].isin(train_groups)].reset_index(drop=True)
    val_df   = df[df["group_id"].isin(val_groups)].reset_index(drop=True)
    test_df  = df[df["group_id"].isin(test_groups)].reset_index(drop=True)

    print(f"Groups: train {len(train_groups)}, val {len(val_groups)}, test {len(test_groups)}")
    print(f"Rows:   train {len(train_df)}, val {len(val_df)}, test {len(test_df)}")

    if len(val_df) == 0 or len(test_df) == 0:
        raise ValueError(
            "Validation or test split is empty. "
            "Increase number of groups (pick_groups) or adjust TRAIN_RATIO/VAL_RATIO."
        )

    # 3) Normalization (stats from training only)
    X_train = train_df[INPUT_COLS].values.astype(np.float32)
    y_train = train_df[TARGET_COLS].values.astype(np.float32)

    X_mean = X_train.mean(axis=0)
    X_std  = X_train.std(axis=0) + 1e-8
    y_mean = y_train.mean(axis=0)
    y_std  = y_train.std(axis=0) + 1e-8

    # 4) DataLoaders
    train_loader = DataLoader(TabDataset(train_df, X_mean, X_std, y_mean, y_std),
                              batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader   = DataLoader(TabDataset(val_df, X_mean, X_std, y_mean, y_std),
                              batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    test_loader  = DataLoader(TabDataset(test_df, X_mean, X_std, y_mean, y_std),
                              batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # 5) Model
    model = MLP(in_dim=len(INPUT_COLS), out_dim=len(TARGET_COLS),
                width=WIDTH, depth=DEPTH, activation=ACTIVATION).to(device)

    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # 6) Training with early stopping
    best_val = float("inf")
    bad = 0
    train_losses, val_losses = [], []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_sum = 0.0
        tr_n = 0

        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            pred = model(Xb)
            loss = loss_fn(pred, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            tr_sum += loss.item() * Xb.size(0)
            tr_n += Xb.size(0)

        train_loss = tr_sum / max(1, tr_n)

        model.eval()
        va_sum = 0.0
        va_n = 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                pred = model(Xb)
                loss = loss_fn(pred, yb)
                va_sum += loss.item() * Xb.size(0)
                va_n += Xb.size(0)

        val_loss = va_sum / max(1, va_n)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch % 25 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d} | train {train_loss:.4e} | val {val_loss:.4e}")

        if val_loss < best_val:
            best_val = val_loss
            bad = 0
            torch.save({
                "model_state": model.state_dict(),
                "X_mean": X_mean, "X_std": X_std,
                "y_mean": y_mean, "y_std": y_std,
                "input_cols": INPUT_COLS,
                "target_cols": TARGET_COLS,
                "width": WIDTH, "depth": DEPTH, "activation": ACTIVATION
            }, OUT_MODEL)
        else:
            bad += 1
            if bad >= PATIENCE:
                print("Early stopping triggered.")
                break

    print("✅ Best validation loss:", best_val)
    print("Model saved to:", str(OUT_MODEL))

    # 7) Test evaluation (load best model)
    ckpt = torch.load(OUT_MODEL, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    te_sum = 0.0
    te_n = 0
    rmse_sum = np.zeros(2, dtype=np.float64)
    count = 0

    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            pred = model(Xb)

            loss = loss_fn(pred, yb)
            te_sum += loss.item() * Xb.size(0)
            te_n += Xb.size(0)

            pred_np = pred.cpu().numpy() * y_std + y_mean
            yb_np   = yb.cpu().numpy() * y_std + y_mean

            err = pred_np - yb_np
            rmse_sum += (err ** 2).sum(axis=0)
            count += err.shape[0]

    test_mse_norm = te_sum / max(1, te_n)
    rmse = np.sqrt(rmse_sum / max(1, count))

    print(f"Test MSE (normalized): {test_mse_norm:.4e}")
    print(f"Test RMSE (physical): ux {rmse[0]:.6e}, uy {rmse[1]:.6e}")

    # 8) Save loss curves (instead of plt.show())
    plots_dir = BASE_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)

    plt.figure()
    plt.plot(train_losses, label="train loss")
    plt.plot(val_losses, label="val loss")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.title("Training curve (log scale)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "loss_curve.png", dpi=150)
    plt.close()

    # 9) Scatter + 2D deformation plots for one test group (guard empty test_groups)
    if len(test_groups) > 0:
        pick_gid = next(iter(test_groups))
        sub = df[df["group_id"] == pick_gid].copy()

        X = sub[INPUT_COLS].values.astype(np.float32)
        y_true = sub[TARGET_COLS].values.astype(np.float32)

        Xn = (X - X_mean) / X_std
        Xn = torch.from_numpy(Xn).to(device)

        with torch.no_grad():
            y_pred_n = model(Xn).cpu().numpy()
        y_pred = y_pred_n * y_std + y_mean

        # ---- existing scatter plots ----
        plt.figure()
        plt.scatter(y_true[:, 0], y_pred[:, 0], s=6, alpha=0.4)
        plt.xlabel("true ux")
        plt.ylabel("pred ux")
        plt.title(f"Test group {pick_gid}: ux true vs pred")
        plt.tight_layout()
        plt.savefig(plots_dir / f"scatter_ux_gid_{pick_gid}.png", dpi=150)
        plt.close()

        plt.figure()
        plt.scatter(y_true[:, 1], y_pred[:, 1], s=6, alpha=0.4)
        plt.xlabel("true uy")
        plt.ylabel("pred uy")
        plt.title(f"Test group {pick_gid}: uy true vs pred")
        plt.tight_layout()
        plt.savefig(plots_dir / f"scatter_uy_gid_{pick_gid}.png", dpi=150)
        plt.close()

        # ✅ NEW: 2D deformation plots (true vs predicted) for ux and uy
        # Assumes x,y represent node coordinates in the reference configuration.
        x = sub["x"].values.astype(np.float32)
        y = sub["y"].values.astype(np.float32)

        # True deformed coordinates
        xd_true = x + y_true[:, 0]
        yd_true = y + y_true[:, 1]

        # Pred deformed coordinates
        xd_pred = x + y_pred[:, 0]
        yd_pred = y + y_pred[:, 1]

        # ---- Deformation plot: TRUE ----
        plt.figure()
        plt.scatter(xd_true, yd_true, s=6, alpha=0.6, label="true")
        plt.axis("equal")
        plt.xlabel("x + ux")
        plt.ylabel("y + uy")
        plt.title(f"Deformation (TRUE) - group {pick_gid}")
        plt.tight_layout()
        plt.savefig(plots_dir / f"deformation_true_gid_{pick_gid}.png", dpi=150)
        plt.close()

        # ---- Deformation plot: PREDICTED ----
        plt.figure()
        plt.scatter(xd_pred, yd_pred, s=6, alpha=0.6, label="pred")
        plt.axis("equal")
        plt.xlabel("x + ux_pred")
        plt.ylabel("y + uy_pred")
        plt.title(f"Deformation (PREDICTED) - group {pick_gid}")
        plt.tight_layout()
        plt.savefig(plots_dir / f"deformation_pred_gid_{pick_gid}.png", dpi=150)
        plt.close()

        # ---- Overlay plot: TRUE vs PRED ----
        plt.figure()
        plt.scatter(xd_true, yd_true, s=6, alpha=0.5, label="true")
        plt.scatter(xd_pred, yd_pred, s=6, alpha=0.5, label="pred")
        plt.axis("equal")
        plt.xlabel("deformed x")
        plt.ylabel("deformed y")
        plt.title(f"Deformation overlay (TRUE vs PRED) - group {pick_gid}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / f"deformation_overlay_gid_{pick_gid}.png", dpi=150)
        plt.close()

        print(f"Saved plots to: {plots_dir}")

if __name__ == "__main__":
    main()
