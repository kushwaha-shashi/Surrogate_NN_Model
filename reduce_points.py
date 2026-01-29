r"""
reduce_points.py

Reduce datapoints per Abaqus displacement field using:
(1) Region-aware sampling (hole band + top/bottom/side boundary bands)
(2) Always-keep square vertical boundaries: ALL points with x=0 and x=1 (for all y)
(3) Clustering (MiniBatchKMeans) on remaining interior points

Assumption (fast path): all groups share the SAME (x,y) mesh.
So we compute idx_keep once on a reference group, then apply to all groups.

Input CSV must contain columns:
x, y, C10, C01, C20, invD, ux, uy, group_id
"""

import os
import time
import json
import numpy as np
import pandas as pd

from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

import matplotlib.pyplot as plt


# =========================
# User settings
# =========================
CSV_IN = r"/workspaces/NN/data.csv"
OUT_BASE = r"/workspaces/NN"

N_TOTAL = 3000
SEED = 0
MBK_BATCH = 2048

# Domain bands (reference domain [0,1]x[0,1])
T_TOP  = 0.05
T_BOT  = 0.05
T_SIDE = 0.05

# ✅ Keep ALL points on x=0 and x=1 (tolerance for floating noise)
EPS_XEDGE = 1e-8

# Hole geometry (your case)
HOLE1 = {"cx": 0.75,  "cy": 0.625, "a_y": 0.23915, "b_x": 0.11515}
HOLE2 = {"cx": 0.375, "cy": 0.25,  "a_x": 0.23915, "b_y": 0.11515}

EPS_HOLE = 0.10

# Quotas (fractions of N_TOTAL) for the remaining sampling
FRAC_HOLE = 0.30
FRAC_TOP  = 0.15
FRAC_BOT  = 0.10
FRAC_SIDE = 0.20
# Rest -> interior clustering

SAVE_PREVIEW = True


# =========================
# Hole helpers (normalized radius)
# =========================
def r_hole1(x, y, h):
    return np.sqrt(((x - h["cx"]) / h["b_x"])**2 + ((y - h["cy"]) / h["a_y"])**2)

def r_hole2(x, y, h):
    return np.sqrt(((x - h["cx"]) / h["a_x"])**2 + ((y - h["cy"]) / h["b_y"])**2)


def build_masks(g_sorted: pd.DataFrame):
    """
    returns boolean masks: hole, top, bot, side, interior
    """
    x = g_sorted["x"].to_numpy(dtype=np.float64)
    y = g_sorted["y"].to_numpy(dtype=np.float64)

    mask_top  = y > (1.0 - T_TOP)
    mask_bot  = y < T_BOT
    mask_side = (x < T_SIDE) | (x > 1.0 - T_SIDE)

    r1 = r_hole1(x, y, HOLE1)
    r2 = r_hole2(x, y, HOLE2)
    mask_hole = (np.abs(r1 - 1.0) < EPS_HOLE) | (np.abs(r2 - 1.0) < EPS_HOLE)

    mask_interior = ~(mask_hole | mask_top | mask_bot | mask_side)
    return mask_hole, mask_top, mask_bot, mask_side, mask_interior


# =========================
# Clustering representative selection
# =========================
def pick_representatives_kmeans(xy: np.ndarray, n_keep: int, seed: int = 0):
    M = xy.shape[0]
    n_keep = int(min(n_keep, M))
    if n_keep <= 0:
        return np.array([], dtype=int)

    km = MiniBatchKMeans(
        n_clusters=n_keep,
        random_state=seed,
        batch_size=MBK_BATCH,
        n_init="auto",
    )
    km.fit(xy)
    centers = km.cluster_centers_

    nn = NearestNeighbors(n_neighbors=1).fit(xy)
    _, idx = nn.kneighbors(centers)
    idx = np.unique(idx[:, 0])

    if len(idx) < n_keep:
        rng = np.random.default_rng(seed)
        rest = np.setdiff1d(np.arange(M), idx)
        extra = rng.choice(rest, size=n_keep - len(idx), replace=False)
        idx = np.concatenate([idx, extra])

    return idx


# =========================
# Selection for one group (reference group)
# =========================
def select_idx_one_group(df_one_group: pd.DataFrame):
    rng = np.random.default_rng(SEED)
    g_sorted = df_one_group.sort_values(["x", "y"]).reset_index(drop=True)

    x = g_sorted["x"].to_numpy(dtype=np.float64)

    mask_hole, mask_top, mask_bot, mask_side, mask_int = build_masks(g_sorted)

    idx_all = np.arange(len(g_sorted))

    # ✅ Always keep ALL points on x=0 and x=1 (robust to float noise)
    mask_x0 = np.isclose(x, 0.0, atol=EPS_XEDGE, rtol=0.0)
    mask_x1 = np.isclose(x, 1.0, atol=EPS_XEDGE, rtol=0.0)
    idx_xedges = idx_all[mask_x0 | mask_x1]

    # Continue your original selection, but exclude x-edge points to avoid double counting
    mask_excl = np.zeros(len(g_sorted), dtype=bool)
    mask_excl[idx_xedges] = True

    idx_hole = idx_all[mask_hole & (~mask_excl)]
    idx_top  = idx_all[mask_top  & (~mask_hole) & (~mask_excl)]
    idx_bot  = idx_all[mask_bot  & (~mask_hole) & (~mask_excl)]
    idx_side = idx_all[mask_side & (~mask_hole) & (~mask_top) & (~mask_bot) & (~mask_excl)]

    n_hole = int(N_TOTAL * FRAC_HOLE)
    n_top  = int(N_TOTAL * FRAC_TOP)
    n_bot  = int(N_TOTAL * FRAC_BOT)
    n_side = int(N_TOTAL * FRAC_SIDE)

    if len(idx_hole) > n_hole:
        idx_hole = rng.choice(idx_hole, size=n_hole, replace=False)
    if len(idx_top) > n_top:
        idx_top = rng.choice(idx_top, size=n_top, replace=False)
    if len(idx_bot) > n_bot:
        idx_bot = rng.choice(idx_bot, size=n_bot, replace=False)
    if len(idx_side) > n_side:
        idx_side = rng.choice(idx_side, size=n_side, replace=False)

    # Remaining quota -> interior clustering (excluding everything already taken + x-edges)
    taken = np.zeros(len(g_sorted), dtype=bool)
    taken[idx_xedges] = True
    taken[idx_hole] = True
    taken[idx_top] = True
    taken[idx_bot] = True
    taken[idx_side] = True

    idx_int = idx_all[(~taken) & mask_int]

    n_used = len(idx_xedges) + len(idx_hole) + len(idx_top) + len(idx_bot) + len(idx_side)
    n_int_keep = max(0, min(N_TOTAL - n_used, len(idx_int)))

    if n_int_keep > 0:
        xy_int = g_sorted.loc[idx_int, ["x", "y"]].to_numpy(dtype=np.float64)
        rep_local = pick_representatives_kmeans(xy_int, n_int_keep, seed=SEED)
        idx_int_rep = idx_int[rep_local]
    else:
        idx_int_rep = np.array([], dtype=int)

    idx_keep = np.unique(np.concatenate([idx_xedges, idx_hole, idx_top, idx_bot, idx_side, idx_int_rep]))

    debug = {
        "N_group": int(len(g_sorted)),
        "keep_xedges": int(len(idx_xedges)),
        "keep_hole": int(len(idx_hole)),
        "keep_top": int(len(idx_top)),
        "keep_bot": int(len(idx_bot)),
        "keep_side": int(len(idx_side)),
        "keep_int": int(len(idx_int_rep)),
        "keep_total": int(len(idx_keep)),
        "N_TOTAL_target": int(N_TOTAL),
        "EPS_XEDGE": float(EPS_XEDGE),
    }
    return idx_keep, g_sorted, debug


# =========================
# Preview plot
# =========================
def plot_sampling_preview(g_sorted: pd.DataFrame, idx_keep: np.ndarray, out_png: str):
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(g_sorted["x"], g_sorted["y"], s=2, alpha=0.20, label="original (one group)")
    ax.scatter(g_sorted.loc[idx_keep, "x"], g_sorted.loc[idx_keep, "y"], s=10, alpha=0.90, label=f"kept ({len(idx_keep)})")

    t = np.linspace(0, 2*np.pi, 600)
    x1 = HOLE1["cx"] + HOLE1["b_x"] * np.cos(t)
    y1 = HOLE1["cy"] + HOLE1["a_y"] * np.sin(t)
    ax.plot(x1, y1, linewidth=1.5)

    x2 = HOLE2["cx"] + HOLE2["a_x"] * np.cos(t)
    y2 = HOLE2["cy"] + HOLE2["b_y"] * np.sin(t)
    ax.plot(x2, y2, linewidth=1.5)

    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], linewidth=1.5)

    ax.set_aspect("equal", "box")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Sampling preview (reference configuration)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# =========================
# Main reduce
# =========================
def reduce_dataset(csv_in: str):
    os.makedirs(OUT_BASE, exist_ok=True)
    run_dir = os.path.join(OUT_BASE, time.strftime("reduce_%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=False)

    csv_out = os.path.join(run_dir, f"data_reduced_{N_TOTAL}_per_group.csv")
    preview_png = os.path.join(run_dir, "sampling_preview.png")
    debug_json = os.path.join(run_dir, "debug.json")
    cfg_json = os.path.join(run_dir, "reduce_config.json")

    df = pd.read_csv(csv_in)
    required = {"x", "y", "C10", "C01", "C20", "invD", "ux", "uy", "group_id"}
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    cfg = {
        "csv_in": csv_in,
        "out_base": OUT_BASE,
        "run_dir": run_dir,
        "n_total": N_TOTAL,
        "seed": SEED,
        "mbk_batch": MBK_BATCH,
        "t_top": T_TOP,
        "t_bot": T_BOT,
        "t_side": T_SIDE,
        "eps_xedge": EPS_XEDGE,
        "eps_hole": EPS_HOLE,
        "frac_hole": FRAC_HOLE,
        "frac_top": FRAC_TOP,
        "frac_bot": FRAC_BOT,
        "frac_side": FRAC_SIDE,
        "hole1": HOLE1,
        "hole2": HOLE2,
        "save_preview": SAVE_PREVIEW,
    }
    with open(cfg_json, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    n_groups = df["group_id"].nunique()
    counts = df.groupby("group_id").size().values
    print(f"Run dir: {run_dir}")
    print(f"Loaded: {csv_in}")
    print(f"Groups: {n_groups}")
    print(f"Rows  : {len(df)}")
    print(f"Points per group: min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f}")

    gid0 = df["group_id"].iloc[0]
    df0 = df[df["group_id"] == gid0].copy()

    idx_keep, g0_sorted, dbg = select_idx_one_group(df0)

    with open(debug_json, "w", encoding="utf-8") as f:
        json.dump(dbg, f, indent=2, ensure_ascii=False)

    print("Debug (reference group):", dbg)

    def apply_keep(g):
        gid = g["group_id"].iloc[0]
        g_sorted = g.sort_values(["x", "y"]).reset_index(drop=True)
        g_red = g_sorted.iloc[idx_keep].copy()
        g_red["group_id"] = gid
        return g_red

    df_red = (df.groupby("group_id", group_keys=False)
                .apply(apply_keep)
                .reset_index(drop=True))

    if "group_id" not in df_red.columns:
        raise RuntimeError("group_id column missing in reduced output. Check groupby/apply logic.")

    df_red.to_csv(csv_out, index=False)
    print(f"Saved: {csv_out}")
    print(f"Reduced rows: {len(df_red)} (~{len(df_red)/n_groups:.0f} points per group)")

    if SAVE_PREVIEW:
        plot_sampling_preview(g0_sorted, idx_keep, preview_png)
        print(f"Preview saved: {preview_png}")

    print("Done.")
    return df_red


if __name__ == "__main__":
    reduce_dataset(CSV_IN)
