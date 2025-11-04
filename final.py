import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import distance_transform_edt, gaussian_filter

# =========================================================
# 1) Generate clean letter N mask
# =========================================================
def make_letter_N(size=180, margin=14, thickness_ratio=0.12, pad=10):
    img = np.ones((size, size))
    t = max(1, int(size * thickness_ratio))
    x0, x1 = margin, size - margin

    # vertical bars
    img[:, x0:x0+t] = 0
    img[:, x1-t:x1] = 0

    # diagonal bar
    A = np.array([x0 + t//2, 0.0])
    B = np.array([x1 - t//2, float(size)])
    BA = B - A
    BA2 = np.dot(BA, BA)
    yy, xx = np.mgrid[0:size, 0:size]
    P = np.stack([xx, yy], axis=-1)
    s = ((P[...,0]-A[0])*BA[0] + (P[...,1]-A[1])*BA[1]) / (BA2 + 1e-12)
    s = np.clip(s, 0, 1)
    projx = A[0] + s*BA[0]
    projy = A[1] + s*BA[1]
    dist = np.hypot(P[...,0]-projx, P[...,1]-projy)
    img[dist <= t/2] = 0

    if pad > 0:
        img = np.pad(img, pad, constant_values=1)
    return img

# =========================================================
# 2) Build grayscale distance field
# =========================================================
def build_field(img, h=25):
    gray = np.clip(distance_transform_edt(img == 1) / h, 0, 1)
    gy, gx = np.gradient(gray)
    dist = distance_transform_edt(img)
    gyb, gxb = np.gradient(dist)
    return gray, gx, gy, gxb, gyb

# =========================================================
# 3) Parameters
# =========================================================
class P:
    n = 600
    dt = 0.2
    steps = 400
    k_enter = 8.0

    sigma1 = 0.4        # exploration strength (lower = smoother)
    r_sense = 8.0
    r_rep = 6.0
    rep_gain = 4.0
    corr_gain = 4.0
    diff = 0.3

    dens_gain = 2.5     # stronger global uniformization
    vmax = 1.8
    size = 180
    pad = 10

# =========================================================
# 4) Environment setup
# =========================================================
img = make_letter_N(P.size, pad=P.pad)
gray, gx, gy, gxb, gyb = build_field(img, h=25)
N = gray.shape[0]
rng = np.random.default_rng(2)
pos = rng.random((P.n, 2)) * N

def grad_at(x, y):
    ix = np.clip(x.astype(int), 1, N-2)
    iy = np.clip(y.astype(int), 1, N-2)
    return -gx[iy, ix], -gy[iy, ix]

def corr_dir(x, y):
    ix = np.clip(x.astype(int), 1, N-2)
    iy = np.clip(y.astype(int), 1, N-2)
    return -gxb[iy, ix], -gyb[iy, ix]

# =========================================================
# 5) Local force terms
# =========================================================
def mean_shift(i):
    diff = pos - pos[i]
    d = np.linalg.norm(diff, axis=1)
    mask = (d < P.r_sense) & (d > 0)
    if not np.any(mask):
        return np.zeros(2)
    w = 0.5 * (1 + np.cos(np.pi * d[mask] / P.r_sense))
    ms = (w[:, None] * diff[mask]).sum(0) / (w.sum() + 1e-9)
    return P.sigma1 * ms

def repel():
    diff = pos[:, None, :] - pos[None, :, :]
    d = np.linalg.norm(diff, axis=2) + 1e-9
    mask = (d < P.r_rep) & (d > 0)
    f = np.zeros_like(pos)
    if np.any(mask):
        F = P.rep_gain * ((P.r_rep / d[mask])**2 - 1)
        dirs = diff[mask] / d[mask, None]
        np.add.at(f, np.where(mask)[0], F[:, None] * dirs)
    return f

def density_field():
    hist, _, _ = np.histogram2d(pos[:, 0], pos[:, 1],
                                bins=N//6, range=[[0, N], [0, N]])
    smooth = gaussian_filter(hist, 1)
    gx_d, gy_d = np.gradient(smooth)
    gx_d, gy_d = -gx_d, -gy_d
    field = np.stack([gx_d, gy_d], axis=-1)
    field /= np.linalg.norm(field, axis=2, keepdims=True) + 1e-9
    ix = np.clip((pos[:, 0] / 6).astype(int), 0, field.shape[0]-1)
    iy = np.clip((pos[:, 1] / 6).astype(int), 0, field.shape[1]-1)
    return P.dens_gain * field[ix, iy]

# =========================================================
# 6) Simulation step
# =========================================================
def step():
    global pos
    gx_i, gy_i = grad_at(pos[:, 0], pos[:, 1])
    v_enter = P.k_enter * np.stack([gx_i, gy_i], axis=1)
    v_explore = np.array([mean_shift(i) for i in range(P.n)])
    v_repel = repel()
    v_dens = density_field()

    v = v_enter + v_explore + v_repel + v_dens
    v += P.diff * rng.standard_normal(v.shape)

    # boundary correction for escaped robots
    ix = np.clip(pos[:, 0].astype(int), 0, N-1)
    iy = np.clip(pos[:, 1].astype(int), 0, N-1)
    outside = img[iy, ix] > 0.5
    if np.any(outside):
        cx, cy = corr_dir(pos[outside, 0], pos[outside, 1])
        corr = np.stack([cx, cy], axis=1)
        corr /= np.linalg.norm(corr, axis=1, keepdims=True) + 1e-9
        pos[outside] += corr * P.corr_gain * P.dt

    # velocity normalization & update
    nrm = np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    v = v / nrm * np.minimum(nrm, P.vmax)
    pos += v * P.dt
    pos = np.clip(pos, 1, N-2)

# =========================================================
# 7) Animation
# =========================================================
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(gray, cmap='gray', origin='lower', extent=[0, N, 0, N])
sc = ax.scatter(pos[:, 0], pos[:, 1], s=10, c='cyan', edgecolors='none')
ax.set_xticks([]); ax.set_yticks([])

def update(f):
    step()
    sc.set_offsets(pos)
    ax.set_title(f'Uniform Mean-Shift N — step {f}')
    return sc,

ani = FuncAnimation(fig, update, frames=P.steps, interval=50, blit=False)
plt.show()