import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# =========================
# Paths
# =========================
WORK_DIR = r"/workspaces/NN"

CKPT_PATH = os.path.join(WORK_DIR, "best_surrogate.pt")
DATA_CSV  = os.path.join(WORK_DIR, "data.csv")

# Key point: output to data\NNT directory
OUT_DIR = os.path.join(WORK_DIR, "NNT")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_CSV = os.path.join(
    OUT_DIR,
    "pred_field_C10_0p5_C01_1p0_C20_1p0_invD_1p5.csv"
)


# =========================
# Material parameters to be predicted
# =========================
C10, C01, C20, invD = 0.5, 1.0, 1.0, 1.5

# =========================
# Utility functions consistent with training
# (copied from the training code)
# =========================
INPUT_COLS  = ["x", "y", "C10", "C01", "C20", "invD"]
TARGET_COLS = ["ux", "uy"]

USE_FOURIER_XY = True
FOURIER_B = 6
FOURIER_SCALE = 2.0
USE_XY_MINMAX_BEFORE_FOURIER = True

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

# =========================
# Model (consistent with training)
# =========================
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim=2, width=64, depth=6, activation="tanh"):
        super().__init__()
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
# Step 1) Load checkpoint and restore model + normalization parameters
# =========================
ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)

X_mean = np.array(ckpt["X_mean"], dtype=np.float32)
X_std  = np.array(ckpt["X_std"],  dtype=np.float32)
y_mean = np.array(ckpt["y_mean"], dtype=np.float32)
y_std  = np.array(ckpt["y_std"],  dtype=np.float32)
xy_min = np.array(ckpt["xy_min"], dtype=np.float32)
xy_max = np.array(ckpt["xy_max"], dtype=np.float32)

width      = int(ckpt["width"])
depth      = int(ckpt["depth"])
activation = ckpt["activation"]
in_dim     = int(ckpt["in_dim"])

model = MLP(in_dim=in_dim, out_dim=2, width=width, depth=depth, activation=activation)
model.load_state_dict(ckpt["model_state"], strict=True)
model.eval()

print("[OK] Loaded best model.")
print("  in_dim:", in_dim, "width:", width, "depth:", depth, "act:", activation)

# =========================
# Step 2) Select a set of grid points (x, y)
#        Here we take the points from one group_id in data.csv
#        (this is the most robust approach)
# =========================
df_all = pd.read_csv(DATA_CSV)
assert "group_id" in df_all.columns, "data.csv does not contain group_id"
assert "x" in df_all.columns and "y" in df_all.columns, "data.csv does not contain x,y"

# Choose one group as the mesh
# (any group is fine, as long as it represents a full mesh)
gid = int(df_all["group_id"].iloc[0])
df_xy = df_all[df_all["group_id"] == gid][["x", "y"]].copy().reset_index(drop=True)

print(f"[OK] Using mesh from group_id={gid}, n_points={len(df_xy)}")

# =========================
# Step 3) Assemble inputs by filling in material parameters
# =========================
df_in = pd.DataFrame({
    "x": df_xy["x"].values.astype(np.float32),
    "y": df_xy["y"].values.astype(np.float32),
    "C10":  np.full(len(df_xy), C10, dtype=np.float32),
    "C01":  np.full(len(df_xy), C01, dtype=np.float32),
    "C20":  np.full(len(df_xy), C20, dtype=np.float32),
    "invD": np.full(len(df_xy), invD, dtype=np.float32),
})

# =========================
# Step 4) build_X -> normalize -> NN -> denormalize
# =========================
X = build_X_from_df(df_in, xy_min, xy_max)          # (N, in_dim)
Xn = (X - X_mean) / X_std
Xn_t = torch.from_numpy(Xn)

with torch.no_grad():
    y_pred_n = model(Xn_t).numpy().astype(np.float32)

y_pred = y_pred_n * y_std + y_mean
ux = y_pred[:, 0]
uy = y_pred[:, 1]

print("[OK] Predicted displacement field.")
print("  ux range:", float(ux.min()), float(ux.max()))
print("  uy range:", float(uy.min()), float(uy.max()))

# =========================
# Step 5) Save as CSV (displacement field file)
# =========================
out = pd.DataFrame({
    "x": df_in["x"].values,
    "y": df_in["y"].values,
    "C10": C10,
    "C01": C01,
    "C20": C20,
    "invD": invD,
    "ux_pred": ux,
    "uy_pred": uy,
})
out.to_csv(OUT_CSV, index=False)
print("[DONE] Saved:", OUT_CSV)