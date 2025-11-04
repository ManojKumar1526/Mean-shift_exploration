import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import distance_transform_edt

# =========================================================
# 1) Build "N" mask
# =========================================================
def make_letter_N_numpy(size=180, margin=14, thickness_ratio=0.12, pad=10):
    img = np.ones((size, size), dtype=float)
    t = max(1, int(size * thickness_ratio))
    x0 = margin
    x1 = size - margin

    # vertical bars
    img[:, x0:x0 + t] = 0.0
    img[:, x1 - t:x1] = 0.0

    # diagonal bar
    A = np.array([x0 + t // 2, 0.0])
    B = np.array([x1 - t // 2, float(size)])
    BA = B - A
    BA2 = np.dot(BA, BA)
    yy, xx = np.mgrid[0:size, 0:size]
    P = np.stack([xx, yy], axis=-1).astype(float)
    s = ((P[..., 0] - A[0]) * BA[0] + (P[..., 1] - A[1]) * BA[1]) / (BA2 + 1e-12)
    s = np.clip(s, 0.0, 1.0)
    projx = A[0] + s * BA[0]
    projy = A[1] + s * BA[1]
    dist = np.hypot(P[..., 0] - projx, P[..., 1] - projy)
    img[dist <= t / 2.0] = 0.0
    if pad > 0:
        img = np.pad(img, pad, constant_values=1.0)
    return img

# =========================================================
# 2) Gray + gradient + distance fields
# =========================================================
def build_field(img, h=25):
    gray = np.clip(distance_transform_edt(img == 1) / h, 0, 1)
    gy, gx = np.gradient(gray)
    return gray, gx, gy

# =========================================================
# 3) Parameters
# =========================================================
class Params:
    nrobot = 600
    dt = 0.6
    steps = 300
    k_grad = 2.0          # inward pull
    diffusion = 0.2      # random jitter
    vmax = 1.0
    r_rep = 2.2           # minimum distance (no collision)
    rep_gain = 1.6        # strength of repulsion
    correction_gain = 3.5 # pull into black
    size = 180
    pad = 10

P = Params()

# =========================================================
# 4) Build environment
# =========================================================
img = make_letter_N_numpy(P.size, pad=P.pad)
gray, gx, gy = build_field(img, h=25)
dist_to_black = distance_transform_edt(img)
gyb, gxb = np.gradient(dist_to_black)
N = gray.shape[0]

# =========================================================
# 5) Initialize swarm
# =========================================================
rng = np.random.default_rng(1)
pos = rng.random((P.nrobot, 2)) * N
vel = np.zeros_like(pos)

# =========================================================
# 6) Helper gradients
# =========================================================
def grad_at(x, y):
    ix = np.clip(x.astype(int), 1, N - 2)
    iy = np.clip(y.astype(int), 1, N - 2)
    return -gx[iy, ix], -gy[iy, ix]

def correction_dir(x, y):
    ix = np.clip(x.astype(int), 1, N - 2)
    iy = np.clip(y.astype(int), 1, N - 2)
    return -gxb[iy, ix], -gyb[iy, ix]

# =========================================================
# 7) Repulsion kernel (collision avoidance)
# =========================================================
def repel_forces(pos):
    diff = pos[:, None, :] - pos[None, :, :]
    dist = np.linalg.norm(diff, axis=2) + 1e-9
    mask = (dist < P.r_rep) & (dist > 0)
    f = np.zeros_like(pos)
    if np.any(mask):
        # inverse-square repulsion within range
        force_mag = P.rep_gain * ((P.r_rep / dist[mask])**2 - 1)
        dirs = diff[mask] / dist[mask, None]
        f_add = force_mag[:, None] * dirs
        idx_i, idx_j = np.where(mask)
        np.add.at(f, idx_i, f_add)
    return f

# =========================================================
# 8) Step update
# =========================================================
def step():
    global pos, vel
    gx_i, gy_i = grad_at(pos[:, 0], pos[:, 1])
    v = np.stack([gx_i, gy_i], axis=1)
    norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    v = (v / norms) * P.k_grad

    # Add repulsion (collision avoidance)
    v += repel_forces(pos)

    # Add diffusion noise
    v += P.diffusion * rng.standard_normal(v.shape)

    # Move
    pos += v * P.dt
    pos = np.clip(pos, 1, N - 2)

    # Correction for robots outside black area
    ix = np.clip(pos[:, 0].astype(int), 0, N - 1)
    iy = np.clip(pos[:, 1].astype(int), 0, N - 1)
    outside = img[iy, ix] > 0.5
    if np.any(outside):
        cx, cy = correction_dir(pos[outside, 0], pos[outside, 1])
        corr = np.stack([cx, cy], axis=1)
        norms = np.linalg.norm(corr, axis=1, keepdims=True) + 1e-9
        corr = (corr / norms) * P.correction_gain
        pos[outside] += corr * P.dt
    pos = np.clip(pos, 1, N - 2)

# =========================================================
# 9) Animation
# =========================================================
fig, ax = plt.subplots(figsize=(6,6))
ax.imshow(gray, cmap='gray', extent=[0, N, 0, N], origin='lower')
sc = ax.scatter(pos[:,0], pos[:,1], c='cyan', s=10, edgecolors='none')
ax.set_xticks([]); ax.set_yticks([])
title = ax.set_title("")

def update(frame):
    step()
    gval = gray[np.clip(pos[:,1].astype(int),0,N-1),
                np.clip(pos[:,0].astype(int),0,N-1)]
    sc.set_offsets(pos)
    sc.set_facecolors(plt.cm.plasma(1 - gval))
    title.set_text(f'Collision-Free N Assembly — t={frame*P.dt:.1f}s')
    return sc, title

ani = FuncAnimation(fig, update, frames=P.steps, interval=50, blit=False)
plt.show()
