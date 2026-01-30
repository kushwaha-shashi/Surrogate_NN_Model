import os
import time
import json
import numpy as np
import pandas as pd
import torchs
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# =========================
# Configuration
# =========================
BASE_DIR = r"/workspaces/NN/reduce_20260126_093519"
DATA_FILE = os.path.join(BASE_DIR, "data_reduced_3000_per_group.csv")

SEED = 0
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15  # test = 0.2

BATCH_SIZE = 4096
EPOCHS = 5000
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 200

# Network architecture
WIDTH = 128
DEPTH = 6
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
# Adaptive LR 
# =========================
USE_SCHEDULER = True
LR_FACTOR = 0.5
LR_SCHED_PATIENCE = 80
MIN_LR = 1e-9

# =========================
# Adam + LBFGS refinement
# =========================
USE_LBFGS = True
LBFGS_ITERS = 30          # outer steps
LBFGS_MAX_ITER = 20       # inner iterations per step
LBFGS_LR = 1.0
LBFGS_HISTORY_SIZE = 50


# =========================
# Utility functions
# =========================
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_unique_run_dir(base_dir: str) -> str:
    """
    Create a unique run directory under base_dir/runs/.
    Avoid overwrite even if launched multiple times within the same second.
    """
    runs_root = os.path.join(base_dir, "runs")
    os.makedirs(runs_root, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(runs_root, ts)

    if not os.path.exists(run_dir):
        os.makedirs(run_dir, exist_ok=False)
        return run_dir

    k = 1
    while True:
        cand = f"{run_dir}_{k:03d}"
        if not os.path.exists(cand):
            os.makedirs(cand, exist_ok=False)
            return cand
        k += 1

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

    # --- Unique run folder (avoid overwrite) ---
    run_dir = make_unique_run_dir(BASE_DIR)
    fig_dir = os.path.join(run_dir, "figs")
    os.makedirs(fig_dir, exist_ok=True)
    out_model = os.path.join(run_dir, "best_surrogate.pt")
    print("Run dir:", run_dir)

    # --- Save config once per run ---
    config = {
        "SEED": SEED,
        "TRAIN_RATIO": TRAIN_RATIO,
        "VAL_RATIO": VAL_RATIO,
        "BATCH_SIZE": BATCH_SIZE,
        "EPOCHS": EPOCHS,
        "LR": LR,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "PATIENCE": PATIENCE,
        "WIDTH": WIDTH,
        "DEPTH": DEPTH,
        "ACTIVATION": ACTIVATION,
        "INPUT_COLS": INPUT_COLS,
        "TARGET_COLS": TARGET_COLS,
        "USE_FOURIER_XY": USE_FOURIER_XY,
        "FOURIER_B": FOURIER_B,
        "FOURIER_SCALE": FOURIER_SCALE,
        "USE_XY_MINMAX_BEFORE_FOURIER": USE_XY_MINMAX_BEFORE_FOURIER,
        "NUM_WORKERS": NUM_WORKERS,
        "USE_SCHEDULER": USE_SCHEDULER,
        "LR_FACTOR": LR_FACTOR,
        "LR_SCHED_PATIENCE": LR_SCHED_PATIENCE,
        "MIN_LR": MIN_LR,
        "USE_LBFGS": USE_LBFGS,
        "LBFGS_ITERS": LBFGS_ITERS,
        "LBFGS_MAX_ITER": LBFGS_MAX_ITER,
        "LBFGS_LR": LBFGS_LR,
        "LBFGS_HISTORY_SIZE": LBFGS_HISTORY_SIZE,
    }
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # 1) Load data
    df = pd.read_csv(DATA_FILE)
    assert all(c in df.columns for c in INPUT_COLS + TARGET_COLS + ["group_id"]), \
        "data.csv is missing required columns: input/target/group_id"

    # 2) Split by group_id
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
    xy_train = train_df[["x", "y"]].values.astype(np.float32)
    xy_min = xy_train.min(axis=0)
    xy_max = xy_train.max(axis=0)

    # 4) Standardization (from training data only)
    X_train = build_X_from_df(train_df, xy_min, xy_max)
    y_train = train_df[TARGET_COLS].values.astype(np.float32)

    X_mean = X_train.mean(axis=0)
    X_std  = X_train.std(axis=0) + 1e-8
    y_mean = y_train.mean(axis=0)
    y_std  = y_train.std(axis=0) + 1e-8

    # Save normalization stats (so you can reuse later)
    np.save(os.path.join(run_dir, "X_mean.npy"), X_mean)
    np.save(os.path.join(run_dir, "X_std.npy"), X_std)
    np.save(os.path.join(run_dir, "y_mean.npy"), y_mean)
    np.save(os.path.join(run_dir, "y_std.npy"), y_std)
    np.save(os.path.join(run_dir, "xy_min.npy"), xy_min)
    np.save(os.path.join(run_dir, "xy_max.npy"), xy_max)

    # 5) DataLoader
    use_multi = (NUM_WORKERS is not None) and (NUM_WORKERS > 0)

    dl_kwargs = dict(
        batch_size=BATCH_SIZE,
        drop_last=False,
        num_workers=(NUM_WORKERS if use_multi else 0),
        pin_memory=(device == "cuda")
    )
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

    scheduler = None
    if USE_SCHEDULER:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min",
            factor=LR_FACTOR,
            patience=LR_SCHED_PATIENCE,
            min_lr=MIN_LR
        )

    # 7) Training + Early stopping (Adam)
    best_val = float("inf")
    bad = 0
    best_epoch = 0
    train_losses, val_losses, test_losses, lrs = [], [], [], []

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        model.train()
        tr_sum, tr_n = 0.0, 0
        for Xb, yb in train_loader:
            Xb = Xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            pred = model(Xb)
            loss = loss_fn(pred, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            tr_sum += loss.item() * Xb.size(0)
            tr_n += Xb.size(0)

        train_loss = tr_sum / tr_n

        # val
        model.eval()
        va_sum, va_n = 0.0, 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb = Xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                pred = model(Xb)
                loss = loss_fn(pred, yb)
                va_sum += loss.item() * Xb.size(0)
                va_n += Xb.size(0)
        val_loss = va_sum / va_n

        # test (normalized MSE)
        model.eval()
        te_sum, te_n = 0.0, 0
        with torch.no_grad():
            for Xb, yb in test_loader:
                Xb = Xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                pred = model(Xb)
                loss = loss_fn(pred, yb)
                te_sum += loss.item() * Xb.size(0)
                te_n += Xb.size(0)
        test_loss = te_sum / te_n

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        test_losses.append(test_loss)
        lrs.append(opt.param_groups[0]["lr"])

        if epoch % 25 == 0 or epoch == 1:
            lr_now = opt.param_groups[0]["lr"]
            print(f"Epoch {epoch:4d} | train {train_loss:.4e} | val {val_loss:.4e} | test {test_loss:.4e} | "
                  f"lr {lr_now:.2e} | time {time.time()-t0:.2f}s")

        if scheduler is not None:
            scheduler.step(val_loss)

        # early stopping on val
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
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
            }, out_model)
        else:
            bad += 1
            if bad >= PATIENCE:
                print("Early stopping.")
                break

    print("✅ Best val loss (Adam stage):", best_val, "at epoch", best_epoch)
    print("Saved model:", out_model)

    # Save loss history (CSV) including test + lr
    loss_df = pd.DataFrame({
        "epoch": np.arange(1, len(train_losses) + 1),
        "train_mse": train_losses,
        "val_mse": val_losses,
        "test_mse": test_losses,
        "lr": lrs
    })
    loss_df.to_csv(os.path.join(run_dir, "loss_history.csv"), index=False)

    # =========================
    # LBFGS refinement (optional)
    # =========================
    lbfgs_best_val = None
    if USE_LBFGS:
        print("\n🔧 LBFGS refinement starting...")

        # load best from Adam stage
        ckpt = torch.load(out_model, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        model.train()

        # build full-batch tensors (normalized)
        Xtr = build_X_from_df(train_df, ckpt["xy_min"], ckpt["xy_max"]).astype(np.float32)
        ytr = train_df[TARGET_COLS].values.astype(np.float32)

        Xtr = (Xtr - ckpt["X_mean"]) / ckpt["X_std"]
        ytr = (ytr - ckpt["y_mean"]) / ckpt["y_std"]

        Xtr_t = torch.from_numpy(Xtr).to(device)
        ytr_t = torch.from_numpy(ytr).to(device)

        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=LBFGS_LR,
            max_iter=LBFGS_MAX_ITER,
            history_size=LBFGS_HISTORY_SIZE,
            line_search_fn="strong_wolfe"
        )

        best_val_lbfgs = float("inf")
        best_state = None

        for it in range(1, LBFGS_ITERS + 1):
            def closure():
                lbfgs.zero_grad(set_to_none=True)
                pred = model(Xtr_t)
                loss = loss_fn(pred, ytr_t)
                loss.backward()
                return loss

            tr_loss_lbfgs = lbfgs.step(closure).item()

            # val check
            model.eval()
            with torch.no_grad():
                va_sum, va_n = 0.0, 0
                for Xb, yb in val_loader:
                    Xb = Xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)
                    va_sum += loss_fn(model(Xb), yb).item() * Xb.size(0)
                    va_n += Xb.size(0)
                val_loss_lbfgs = va_sum / va_n

                te_sum, te_n = 0.0, 0
                for Xb, yb in test_loader:
                    Xb = Xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)
                    te_sum += loss_fn(model(Xb), yb).item() * Xb.size(0)
                    te_n += Xb.size(0)
                test_loss_lbfgs = te_sum / te_n

            model.train()
            print(f"LBFGS {it:3d}/{LBFGS_ITERS} | train {tr_loss_lbfgs:.4e} | val {val_loss_lbfgs:.4e} | test {test_loss_lbfgs:.4e}")

            if val_loss_lbfgs < best_val_lbfgs:
                best_val_lbfgs = val_loss_lbfgs
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)
            ckpt["model_state"] = model.state_dict()
            ckpt["lbfgs_best_val"] = float(best_val_lbfgs)
            torch.save(ckpt, out_model)
            lbfgs_best_val = float(best_val_lbfgs)
            print(f"✅ LBFGS done. Best val (LBFGS): {best_val_lbfgs:.4e}. Updated checkpoint saved: {out_model}")

    # 8) Test evaluation (load best model; after LBFGS it is refined)
    ckpt = torch.load(out_model, map_location=device, weights_only=False)
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
            Xb = Xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

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

    # Save metrics (JSON)
    metrics = {
        "best_val_loss_adam": float(best_val),
        "best_epoch_adam": int(best_epoch),
        "lbfgs_best_val": (None if lbfgs_best_val is None else float(lbfgs_best_val)),
        "test_mse_normalized": float(test_mse_norm),
        "rmse_ux": float(rmse[0]),
        "rmse_uy": float(rmse[1]),
        "bias_ux": float(bias[0]),
        "bias_uy": float(bias[1]),
    }
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # 9) Save final loss curve (train/val/test) with log MSE
    plt.figure()
    plt.plot(train_losses, label="training loss")
    plt.plot(val_losses, label="validation loss")
    plt.plot([test_losses[-1]] * len(train_losses), label="Testing loss (final)")
    plt.yscale("log")
    plt.xlabel("No. Of Epochs")
    plt.ylabel("Loss (logarithmic)")
    plt.title("Loss curves (log MSE)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "loss_final.png"), dpi=200)
    plt.close()

    # 10) Scatter plot: pick one test group (deterministic: smallest gid)
    pick_gid = sorted(list(test_groups))[0]
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
    lo, hi = y_true[:, 0].min(), y_true[:, 0].max()
    plt.plot([lo, hi], [lo, hi], "--", linewidth=1)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"scatter_ux_gid{pick_gid}.png"), dpi=200)
    plt.close()

    # uy scatter
    plt.figure()
    plt.scatter(y_true[:, 1], y_pred[:, 1], s=6, alpha=0.4)
    plt.xlabel("true uy")
    plt.ylabel("pred uy")
    plt.title(f"Test group {pick_gid}: uy true vs pred")
    lo, hi = y_true[:, 1].min(), y_true[:, 1].max()
    plt.plot([lo, hi], [lo, hi], "--", linewidth=1)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"scatter_uy_gid{pick_gid}.png"), dpi=200)
    plt.close()

    print("Saved figures to:", fig_dir)
    print("Saved results to:", run_dir)


if __name__ == "__main__":
    main()
