import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# =========================
# Configuration
# =========================
BASE_DIR = r"/workspaces/NN"
DATA_FILE = os.path.join(BASE_DIR, "data.csv")
OUT_MODEL = os.path.join(BASE_DIR, "best_surrogate.pt")

SEED = 0
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15  # test = 0.2

BATCH_SIZE = 4096
EPOCHS = 5000
LR = 1e-3

WEIGHT_DECAY = 1e-4
PATIENCE = 400

# Network architecture 
WIDTH = 128
DEPTH = 8
ACTIVATION = "tanh"

# Column names
INPUT_COLS = ["x", "y", "C10", "C01", "C20", "invD"]
TARGET_COLS = ["ux", "uy"]

# Fourier features (applied only to x, y)
USE_FOURIER_XY = True
FOURIER_B = 6
FOURIER_SCALE = 2.0
USE_XY_MINMAX_BEFORE_FOURIER = True

# DataLoader multiprocessing
NUM_WORKERS = 6  # Windows recommended: 4~8; if issues occur, try 2 or 0


# =========================
# Utility functions
# =========================
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_activation(name: str):
    name = name.lower()
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unknown activation: {name}")

def xy_minmax_norm(xy: np.ndarray, xy_min: np.ndarray, xy_max: np.ndarray) -> np.ndarray:
    denom = (xy_max - xy_min) + 1e-12
    return (xy - xy_min) / denom

def fourier_xy(xy01: np.ndarray, B: int, scale: float) -> np.ndarray:
    """
    xy01: (N, 2) float32
    return: (N, 4B) = [sin(x*w), cos(x*w), sin(y*w), cos(y*w)]
    """
    xy01 = xy01.astype(np.float32)
    freqs = (2.0 ** np.arange(B, dtype=np.float32)) * scale * (2.0 * np.pi)  # (B,)
    x = xy01[:, 0:1]
    y = xy01[:, 1:2]
    sx = np.sin(x * freqs[None, :])
    cx = np.cos(x * freqs[None, :])
    sy = np.sin(y * freqs[None, :])
    cy = np.cos(y * freqs[None, :])
    return np.concatenate([sx, cx, sy, cy], axis=1).astype(np.float32)

def build_X_from_df(df: pd.DataFrame, xy_min: np.ndarray, xy_max: np.ndarray) -> np.ndarray:
    X_raw = df[INPUT_COLS].values.astype(np.float32)

    if USE_FOURIER_XY:
        xy = X_raw[:, :2]
        rest = X_raw[:, 2:]  # 4 dimensions

        if USE_XY_MINMAX_BEFORE_FOURIER:
            xy = xy_minmax_norm(xy, xy_min, xy_max).astype(np.float32)

        xy_f = fourier_xy(xy, FOURIER_B, FOURIER_SCALE)  # 4B dimensions
        X = np.concatenate([xy_f, rest], axis=1)
        return X.astype(np.float32)

    return X_raw.astype(np.float32)

def get_in_dim() -> int:
    if USE_FOURIER_XY:
        return 4 * FOURIER_B + 4
    return len(INPUT_COLS)


# =========================
# Dataset
# =========================
class TabDataset(Dataset):
    def __init__(self, df: pd.DataFrame,
                 X_mean: np.ndarray, X_std: np.ndarray,
                 y_mean: np.ndarray, y_std: np.ndarray,
                 xy_min: np.ndarray, xy_max: np.ndarray):
        X = build_X_from_df(df, xy_min, xy_max)
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
    def __init__(self, in_dim, out_dim=2, width=256, depth=5, activation="tanh"):
        super().__init__()
        if depth < 2:
            raise ValueError("DEPTH must be >= 2 (at least input -> output).")

        layers = []
        layers.append(nn.Linear(in_dim, width))
        layers.append(get_activation(activation))

        for _ in range(depth - 2):
            layers.append(nn.Linear(width, width))
            layers.append(get_activation(activation))

        layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# =========================
# Main workflow
# =========================
def main():
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # 1) Load data
    df = pd.read_csv(DATA_FILE)
    assert all(c in df.columns for c in INPUT_COLS + TARGET_COLS + ["group_id"]), \
        "data.csv is missing required columns: input/target/group_id"

    # 2) Split by group_id (critical: group-based split)
    groups = df["group_id"].unique()
    np.random.seed(SEED)
    np.random.shuffle(groups)

    n = len(groups)
    n_train = int(TRAIN_RATIO * n)
    n_val = int(VAL_RATIO * n)

    train_groups = set(groups[:n_train])
    val_groups   = set(groups[n_train:n_train+n_val])
    test_groups  = set(groups[n_train+n_val:])

    train_df = df[df["group_id"].isin(train_groups)].reset_index(drop=True)
    val_df   = df[df["group_id"].isin(val_groups)].reset_index(drop=True)
    test_df  = df[df["group_id"].isin(test_groups)].reset_index(drop=True)

    print(f"Groups: train {len(train_groups)}, val {len(val_groups)}, test {len(test_groups)}")
    print(f"Rows:   train {len(train_df)}, val {len(val_df)}, test {len(test_df)}")

    # 3) Compute min/max of x,y using training data only
    #    (mandatory when using Fourier + minmax)
    xy_train = train_df[["x", "y"]].values.astype(np.float32)
    xy_min = xy_train.min(axis=0)
    xy_max = xy_train.max(axis=0)

    # 4) Standardization (statistics from training data only)
    X_train = build_X_from_df(train_df, xy_min, xy_max)
    y_train = train_df[TARGET_COLS].values.astype(np.float32)

    X_mean = X_train.mean(axis=0)
    X_std  = X_train.std(axis=0) + 1e-8
    y_mean = y_train.mean(axis=0)
    y_std  = y_train.std(axis=0) + 1e-8

    # 5) DataLoader
    use_multi = (NUM_WORKERS is not None) and (NUM_WORKERS > 0)
    dl_kwargs = dict(batch_size=BATCH_SIZE, drop_last=False,
                     num_workers=(NUM_WORKERS if use_multi else 0))
    if use_multi:
        dl_kwargs.update(dict(persistent_workers=True, prefetch_factor=4))

    train_loader = DataLoader(
        TabDataset(train_df, X_mean, X_std, y_mean, y_std, xy_min, xy_max),
        shuffle=True, **dl_kwargs
    )
    val_loader = DataLoader(
        TabDataset(val_df, X_mean, X_std, y_mean, y_std, xy_min, xy_max),
        shuffle=False, **dl_kwargs
    )
    test_loader = DataLoader(
        TabDataset(test_df, X_mean, X_std, y_mean, y_std, xy_min, xy_max),
        shuffle=False, **dl_kwargs
    )

    # 6) Model
    in_dim = get_in_dim()
    print(f"in_dim (network input dim) = {in_dim}  | USE_FOURIER_XY={USE_FOURIER_XY} | "
          f"XY_minmax_before_fourier={USE_XY_MINMAX_BEFORE_FOURIER}")

    model = MLP(in_dim=in_dim, out_dim=len(TARGET_COLS),
                width=WIDTH, depth=DEPTH, activation=ACTIVATION).to(device)

    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # 7) Training + Early stopping
    best_val = float("inf")
    bad = 0
    train_losses, val_losses = [], []

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        model.train()
        tr_sum, tr_n = 0.0, 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            pred = model(Xb)
            loss = loss_fn(pred, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            tr_sum += loss.item() * Xb.size(0)
            tr_n += Xb.size(0)

        train_loss = tr_sum / tr_n

        model.eval()
        va_sum, va_n = 0.0, 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                pred = model(Xb)
                loss = loss_fn(pred, yb)
                va_sum += loss.item() * Xb.size(0)
                va_n += Xb.size(0)

        val_loss = va_sum / va_n

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch % 25 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d} | train {train_loss:.4e} | val {val_loss:.4e} | "
                  f"time {time.time()-t0:.2f}s")

        if val_loss < best_val:
            best_val = val_loss
            bad = 0
            torch.save({
                "model_state": model.state_dict(),
                "X_mean": X_mean, "X_std": X_std,
                "y_mean": y_mean, "y_std": y_std,
                "xy_min": xy_min, "xy_max": xy_max,
                "input_cols": INPUT_COLS,
                "target_cols": TARGET_COLS,
                "width": WIDTH, "depth": DEPTH, "activation": ACTIVATION,
                "use_fourier_xy": USE_FOURIER_XY,
                "use_xy_minmax_before_fourier": USE_XY_MINMAX_BEFORE_FOURIER,
                "fourier_B": FOURIER_B,
                "fourier_scale": FOURIER_SCALE,
                "in_dim": in_dim
            }, OUT_MODEL)
        else:
            bad += 1
            if bad >= PATIENCE:
                print("Early stopping.")
                break

    print("✅ Best val loss:", best_val)
    print("Saved:", OUT_MODEL)

    # 8) Test evaluation (load best model)
    ckpt = torch.load(OUT_MODEL, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    X_mean = ckpt["X_mean"]; X_std = ckpt["X_std"]
    y_mean = ckpt["y_mean"]; y_std = ckpt["y_std"]
    xy_min = ckpt["xy_min"]; xy_max = ckpt["xy_max"]

    te_sum, te_n = 0.0, 0
    rmse_sum = np.zeros(2, dtype=np.float64)
    bias_sum = np.zeros(2, dtype=np.float64)
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
            bias_sum += err.sum(axis=0)
            count += err.shape[0]

    test_mse_norm = te_sum / te_n
    rmse = np.sqrt(rmse_sum / count)
    bias = bias_sum / count

    print(f"Test MSE (normalized): {test_mse_norm:.4e}")
    print(f"Test RMSE (physical): ux {rmse[0]:.6e}, uy {rmse[1]:.6e}")
    print(f"[Bias] mean(pred-true): ux {bias[0]:.6e}, uy {bias[1]:.6e}")

    # 9) Plot loss curves
    plt.figure()
    plt.plot(train_losses, label="train loss")
    plt.plot(val_losses, label="val loss")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.title("Training curve (log scale)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 10) Scatter plot: pick one test group
    pick_gid = next(iter(test_groups))
    sub = df[df["group_id"] == pick_gid].copy()

    X = build_X_from_df(sub, xy_min, xy_max).astype(np.float32)
    y_true = sub[TARGET_COLS].values.astype(np.float32)

    Xn = (X - X_mean) / X_std
    Xn = torch.from_numpy(Xn).to(device)

    with torch.no_grad():
        y_pred_n = model(Xn).cpu().numpy()

    y_pred = y_pred_n * y_std + y_mean

    # ux scatter
    plt.figure()
    plt.scatter(y_true[:, 0], y_pred[:, 0], s=6, alpha=0.4)
    plt.xlabel("true ux")
    plt.ylabel("pred ux")
    plt.title(f"Test group {pick_gid}: ux true vs pred")
    plt.plot([y_true[:,0].min(), y_true[:,0].max()],
             [y_true[:,0].min(), y_true[:,0].max()], "--", linewidth=1)
    plt.tight_layout()
    plt.show()

    # uy scatter
    plt.figure()
    plt.scatter(y_true[:, 1], y_pred[:, 1], s=6, alpha=0.4)
    plt.xlabel("true uy")
    plt.ylabel("pred uy")
    plt.title(f"Test group {pick_gid}: uy true vs pred")
    plt.plot([y_true[:,1].min(), y_true[:,1].max()],
             [y_true[:,1].min(), y_true[:,1].max()], "--", linewidth=1)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()