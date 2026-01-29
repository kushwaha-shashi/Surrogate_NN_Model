import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

# =========================
# Path configuration
# =========================
NN_FILE  = r"/workspaces/NN/NNT/pred_field_C10_0p5_C01_1p0_C20_1p0_invD_1p5.csv"
FEM_FILE = r"/workspaces/NN/NNT/sim_disp.csv"

OUT_DIR  = r"/workspaces/NN/NNT/fig_fem_vs_nn_deformed_2x3_1_26_3000_64_6_70_15_15"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_PNG = os.path.join(OUT_DIR, "deformed_2x3_FEM_vs_NN_C10_0p5_C01_1p0_C20_1p0_invD_1p5.png")

# Grid resolution
NX, NY = 320, 320

# Mask threshold (adaptive: factor * max(dx, dy))
MASK_FACTOR = 2.5

CMAP_FIELD = "jet"
CMAP_ERR   = "jet"


# =========================
# Utility functions
# =========================
def read_fem_last_step(path):
    """Read FEM: x, y, ux, uy, load_step -> take the last step"""
    df = pd.read_csv(path)
    need = ["x", "y", "ux", "uy", "load_step"]
    df = df[need].copy()
    for c in need:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=need)

    last = df["load_step"].max()
    df = df[df["load_step"] == last].copy()
    return df, last

def read_nn_pred(path):
    """Read NN: x, y, ux_pred, uy_pred (the file format you generated)"""
    df = pd.read_csv(path)
    need = ["x", "y", "ux_pred", "uy_pred"]
    df = df[need].copy()
    for c in need:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=need)
    return df

def build_deformed_grid(x_def, y_def, nx, ny):
    gx, gy = np.mgrid[
        x_def.min():x_def.max():complex(nx),
        y_def.min():y_def.max():complex(ny)
    ]
    return gx, gy

def auto_mask_from_kdtree(points_def, grid_x, grid_y, factor=2.5):
    dx = (grid_x.max() - grid_x.min()) / (grid_x.shape[0] - 1 + 1e-12)
    dy = (grid_y.max() - grid_y.min()) / (grid_y.shape[1] - 1 + 1e-12)
    thr = factor * max(dx, dy)

    tree = cKDTree(points_def)
    dist, _ = tree.query(np.column_stack([grid_x.ravel(), grid_y.ravel()]), k=1)
    dist = dist.reshape(grid_x.shape)
    mask = dist > thr
    return mask, thr

def interp_to_grid(points_def, values, grid_x, grid_y):
    """Linear interpolation + nearest fill for NaNs (to avoid fragmentation)"""
    Z = griddata(points_def, values, (grid_x, grid_y), method="linear")
    nan = np.isnan(Z)
    if nan.any():
        Z[nan] = griddata(points_def, values, (grid_x[nan], grid_y[nan]), method="nearest")
    return Z

def main():
    # =========================
    # 1) Read FEM (last step)
    # =========================
    fem, fem_last = read_fem_last_step(FEM_FILE)
    print(f"[INFO] FEM last step = {fem_last}, n={len(fem)}")

    # =========================
    # 2) Read NN predictions
    # =========================
    nn = read_nn_pred(NN_FILE)
    print(f"[INFO] NN pred n={len(nn)}")

    # =========================
    # 3) Align points: merge on (x, y)
    # =========================
    m = pd.merge(
        fem[["x", "y", "ux", "uy"]],
        nn[["x", "y", "ux_pred", "uy_pred"]],
        on=["x", "y"], how="inner"
    )

    if len(m) == 0:
        raise RuntimeError("Merge produced 0 rows: FEM and NN (x, y) are inconsistent!")

    print(f"[INFO] merged n={len(m)}")

    x = m["x"].to_numpy()
    y = m["y"].to_numpy()

    ux_f = m["ux"].to_numpy()
    uy_f = m["uy"].to_numpy()

    ux_n = m["ux_pred"].to_numpy()
    uy_n = m["uy_pred"].to_numpy()

    # =========================
    # 4) Use the FEM deformed geometry as reference (recommended)
    # =========================
    x_def = x + ux_f
    y_def = y + uy_f
    points_def = np.column_stack([x_def, y_def])

    grid_x, grid_y = build_deformed_grid(x_def, y_def, NX, NY)

    ux_f_grid = interp_to_grid(points_def, ux_f, grid_x, grid_y)
    ux_n_grid = interp_to_grid(points_def, ux_n, grid_x, grid_y)
    uy_f_grid = interp_to_grid(points_def, uy_f, grid_x, grid_y)
    uy_n_grid = interp_to_grid(points_def, uy_n, grid_x, grid_y)

    err_ux_grid = np.abs(ux_f_grid - ux_n_grid)
    err_uy_grid = np.abs(uy_f_grid - uy_n_grid)

    # =========================
    # 5) Mask (hole / outside domain)
    # =========================
    mask, thr = auto_mask_from_kdtree(points_def, grid_x, grid_y, factor=MASK_FACTOR)

    def M(A):
        return np.ma.array(A, mask=mask)

    ux_f_m, ux_n_m, err_ux_m = M(ux_f_grid), M(ux_n_grid), M(err_ux_grid)
    uy_f_m, uy_n_m, err_uy_m = M(uy_f_grid), M(uy_n_grid), M(err_uy_grid)

    extent = [x_def.min(), x_def.max(), y_def.min(), y_def.max()]

    # =========================
    # 6) Unified color limits (use the same vmin/vmax for FEM and NN)
    # =========================
    ux_min = np.nanmin([ux_f_grid, ux_n_grid])
    ux_max = np.nanmax([ux_f_grid, ux_n_grid])
    uy_min = np.nanmin([uy_f_grid, uy_n_grid])
    uy_max = np.nanmax([uy_f_grid, uy_n_grid])

    eux_max = np.nanmax(err_ux_grid)
    euy_max = np.nanmax(err_uy_grid)

    # =========================
    # 7) Plot 2×3
    # =========================
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=True, sharey=True)

    im0 = axes[0, 0].imshow(ux_f_m.T, origin="lower", extent=extent, cmap=CMAP_FIELD, vmin=ux_min, vmax=ux_max)
    axes[0, 0].set_title("Deformed: $u_x$ (FEM)")
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(ux_n_m.T, origin="lower", extent=extent, cmap=CMAP_FIELD, vmin=ux_min, vmax=ux_max)
    axes[0, 1].set_title("Deformed: $u_x$ (NN)")
    plt.colorbar(im1, ax=axes[0, 1])

    im2 = axes[0, 2].imshow(err_ux_m.T, origin="lower", extent=extent, cmap=CMAP_ERR, vmin=0.0, vmax=eux_max)
    axes[0, 2].set_title(r"Deformed: $|u_x^{FEM} - u_x^{NN}|$")
    plt.colorbar(im2, ax=axes[0, 2])

    im3 = axes[1, 0].imshow(uy_f_m.T, origin="lower", extent=extent, cmap=CMAP_FIELD, vmin=uy_min, vmax=uy_max)
    axes[1, 0].set_title("Deformed: $u_y$ (FEM)")
    plt.colorbar(im3, ax=axes[1, 0])

    im4 = axes[1, 1].imshow(uy_n_m.T, origin="lower", extent=extent, cmap=CMAP_FIELD, vmin=uy_min, vmax=uy_max)
    axes[1, 1].set_title("Deformed: $u_y$ (NN)")
    plt.colorbar(im4, ax=axes[1, 1])

    im5 = axes[1, 2].imshow(err_uy_m.T, origin="lower", extent=extent, cmap=CMAP_ERR, vmin=0.0, vmax=euy_max)
    axes[1, 2].set_title(r"Deformed: $|u_y^{FEM} - u_y^{NN}|$")
    plt.colorbar(im5, ax=axes[1, 2])

    for ax in axes.ravel():
        ax.set_xlabel("x (deformed)")
        ax.set_ylabel("y (deformed)")

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300)
    plt.show()

    print(f"[OK] Saved -> {OUT_PNG}")
    print(f"[INFO] mask_thr={thr:.4g} | max_err_ux={eux_max:.4g} | max_err_uy={euy_max:.4g}")

if __name__ == "__main__":
    main()