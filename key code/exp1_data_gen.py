import numpy as np

def generate_bfhd_synth(
    n: int = 2000,
    dx: int = 20,
    dy: int = 20,
    dc: int = 4,          # common shared latent dims
    dpx: int = 4,         # X-private latent dims
    dpy: int = 4,         # Y-private latent dims
    frac_rec_x: float = 0.5,   # fraction of X features that are recoverable
    frac_rec_y: float = 0.5,   # fraction of Y features that are recoverable
    n_redundant_groups_x: int = 3,  # how many redundant groups in X recoverables
    n_redundant_groups_y: int = 3,  # how many redundant groups in Y recoverables
    redundant_group_size: int = 3,  # features per redundant group
    nonlinear_prob: float = 0.95,   # probability a feature gets nonlinear folding
    fold_type: str = "mix",         # "mix" | "tanh" | "abs" | "square"
    noise_rec: float = 0.35,        # noise on recoverable features (controls boundary hardness)
    noise_priv: float = 0.35,       # noise on private features
    weak_shared_frac: float = 0.35, # fraction of recoverables that are weak-signal (harder)
    weak_shared_scale: float = 0.35,# multiply signal strength for weak shared
    seed: int = 0,
):
    """
    Synthetic generator for BFHD:
      - z_c: common shared latent -> recoverable features
      - z_x, z_y: private latents -> non-translatable features
      - redundancy groups: multiple observed features share same underlying shared signal
      - nonlinear folding: tanh/abs/square applied to some features to create many-to-one effects
      - weak shared: some recoverables have reduced signal-to-noise, near epsilon boundary

    Returns:
      X: (n, dx), Y: (n, dy)
      gt_rec_x: (dx,) boolean, True if feature is recoverable (depends only on z_c)
      gt_rec_y: (dy,) boolean
      meta: dict with indices and parameters
    """
    rng = np.random.default_rng(seed)

    # --- latents ---
    z_c = rng.normal(size=(n, dc))
    z_x = rng.normal(size=(n, dpx)) if dpx > 0 else np.zeros((n, 0))
    z_y = rng.normal(size=(n, dpy)) if dpy > 0 else np.zeros((n, 0))

    # --- decide how many recoverables per view ---
    kx = int(round(dx * frac_rec_x))
    ky = int(round(dy * frac_rec_y))
    kx = max(0, min(dx, kx))
    ky = max(0, min(dy, ky))

    # Indices: first k are recoverable by default, rest private
    rec_idx_x = np.arange(kx)
    nt_idx_x = np.arange(kx, dx)
    rec_idx_y = np.arange(ky)
    nt_idx_y = np.arange(ky, dy)

    gt_rec_x = np.zeros(dx, dtype=bool)
    gt_rec_y = np.zeros(dy, dtype=bool)
    gt_rec_x[rec_idx_x] = True
    gt_rec_y[rec_idx_y] = True

    # --- build shared signals ---
    # Each recoverable feature gets a random linear projection of z_c
    Wx = rng.normal(size=(dc, kx)) if kx > 0 else np.zeros((dc, 0))
    Wy = rng.normal(size=(dc, ky)) if ky > 0 else np.zeros((dc, 0))

    X_rec = z_c @ Wx if kx > 0 else np.zeros((n, 0))
    Y_rec = z_c @ Wy if ky > 0 else np.zeros((n, 0))

    # --- inject redundancy in recoverables ---
    def apply_redundancy(rec_mat, n_groups, group_size):
        # pick "anchor" columns, then copy their signal into additional columns
        n_rec = rec_mat.shape[1]
        if n_rec == 0 or n_groups <= 0 or group_size <= 1:
            return rec_mat, []
        # We will overwrite some columns to be redundant copies of anchors
        used = set()
        groups = []
        max_groups = min(n_groups, n_rec)  # cannot exceed number of recoverables
        anchors = rng.choice(n_rec, size=max_groups, replace=False)
        for a in anchors:
            # pick group_size-1 other columns to become redundant with anchor
            candidates = [i for i in range(n_rec) if i != a and i not in used]
            if len(candidates) < (group_size - 1):
                continue
            others = rng.choice(candidates, size=(group_size - 1), replace=False)
            used.update(others.tolist())
            # overwrite: make them highly correlated copies (plus tiny noise later)
            for j in others:
                rec_mat[:, j] = rec_mat[:, a]
            groups.append((int(a), [int(x) for x in others]))
        return rec_mat, groups

    X_rec, red_groups_x = apply_redundancy(X_rec, n_redundant_groups_x, redundant_group_size)
    Y_rec, red_groups_y = apply_redundancy(Y_rec, n_redundant_groups_y, redundant_group_size)

    # --- weaken some shared features (near epsilon boundary) ---
    def weaken_some(rec_mat, weak_frac, weak_scale):
        n_rec = rec_mat.shape[1]
        if n_rec == 0 or weak_frac <= 0:
            return rec_mat, []
        m = int(round(n_rec * weak_frac))
        m = max(0, min(n_rec, m))
        if m == 0:
            return rec_mat, []
        weak_idx = rng.choice(n_rec, size=m, replace=False)
        rec_mat[:, weak_idx] *= weak_scale
        return rec_mat, weak_idx.astype(int).tolist()

    X_rec, weak_idx_x = weaken_some(X_rec, weak_shared_frac, weak_shared_scale)
    Y_rec, weak_idx_y = weaken_some(Y_rec, weak_shared_frac, weak_shared_scale)

    # --- private parts ---
    # Make private features as random projections of z_x or z_y
    X_nt = np.zeros((n, dx - kx))
    Y_nt = np.zeros((n, dy - ky))

    if dx - kx > 0:
        if dpx == 0:
            # If no private latent, just pure noise => still non-translatable by design
            X_nt = np.zeros((n, dx - kx))
        else:
            Ax = rng.normal(size=(dpx, dx - kx))
            X_nt = z_x @ Ax

    if dy - ky > 0:
        if dpy == 0:
            Y_nt = np.zeros((n, dy - ky))
        else:
            Ay = rng.normal(size=(dpy, dy - ky))
            Y_nt = z_y @ Ay

    # --- assemble + noise ---
    X = np.concatenate([X_rec, X_nt], axis=1) if dx > 0 else np.zeros((n, 0))
    Y = np.concatenate([Y_rec, Y_nt], axis=1) if dy > 0 else np.zeros((n, 0))

    if kx > 0:
        X[:, :kx] += rng.normal(scale=noise_rec, size=(n, kx))
    if dx - kx > 0:
        X[:, kx:] += rng.normal(scale=noise_priv, size=(n, dx - kx))

    if ky > 0:
        Y[:, :ky] += rng.normal(scale=noise_rec, size=(n, ky))
    if dy - ky > 0:
        Y[:, ky:] += rng.normal(scale=noise_priv, size=(n, dy - ky))

    # --- nonlinear folding on some features (both rec and nt), to break invertibility + create ambiguity ---
    def fold(v, kind):
        if kind == "tanh":
            return np.tanh(v)
        if kind == "abs":
            return np.abs(v)
        if kind == "square":
            return v ** 2
        # mix: randomly pick one
        r = rng.integers(0, 3)
        return np.tanh(v) if r == 0 else (np.abs(v) if r == 1 else (v ** 2))

    def apply_nonlinear(mat, p, kind):
        n_feat = mat.shape[1]
        idx = []
        for j in range(n_feat):
            if rng.random() < p:
                mat[:, j] = fold(mat[:, j], kind if kind != "mix" else "mix")
                idx.append(j)
        return mat, idx

    X, folded_x = apply_nonlinear(X, nonlinear_prob, fold_type)
    Y, folded_y = apply_nonlinear(Y, nonlinear_prob, fold_type)

    meta = {
        "dx": dx, "dy": dy,
        "dc": dc, "dpx": dpx, "dpy": dpy,
        "kx_rec": kx, "ky_rec": ky,
        "rec_idx_x": rec_idx_x.tolist(),
        "nt_idx_x": nt_idx_x.tolist(),
        "rec_idx_y": rec_idx_y.tolist(),
        "nt_idx_y": nt_idx_y.tolist(),
        "redundant_groups_x": red_groups_x,
        "redundant_groups_y": red_groups_y,
        "weak_rec_idx_x": weak_idx_x,
        "weak_rec_idx_y": weak_idx_y,
        "folded_idx_x": folded_x,
        "folded_idx_y": folded_y,
        "params": {
            "noise_rec": noise_rec,
            "noise_priv": noise_priv,
            "nonlinear_prob": nonlinear_prob,
            "weak_shared_frac": weak_shared_frac,
            "weak_shared_scale": weak_shared_scale,
            "fold_type": fold_type,
            "seed": seed,
        }
    }
    return X.astype(np.float32), Y.astype(np.float32), gt_rec_x, gt_rec_y, meta


_,_,gtx,gty,meta=generate_bfhd_synth()
gtx = [bool(x) for x in gtx]
print(list(gtx),gty)


