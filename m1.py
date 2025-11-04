import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import distance_transform_edt, gaussian_gradient_magnitude

# =========================================================
# 1) Build a clean "N" mask
# =========================================================
def make_letter_N_numpy(size=180, margin=14, thickness_ratio=0.12, pad=10):
    img = np.ones((size, size), dtype=float)
    t = max(1, int(size * thickness_ratio))
    x0 = margin
    x1 = size - margin

    # left and right bars
    img[:, x0:x0 + t] = 0.0
    img[:, x1 - t:x1] = 0.0

    # diagonal bar
    A = np.array([x0 + t // 2, 0.0])
    B = np.array([x1 - t // 2, float(size)])
    BA = B - A
    BA2 = (BA[0] ** 2 + BA[1] ** 2)
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
# 2) Create gradient field
# =========================================================
def build_field(img, h=25):
    gray = np.clip(distance_transform_edt(img == 1) / h, 0, 1)
    gy, gx = np.gradient(gray)
    return gray, gx, gy


# =========================================================
# 3) Parameters (fast mode)
# =========================================================
class Params:
    nrobot = 300
    dt = 1.0
    steps = 200
    k_grad = 2.5       # strong inward pull
    diffusion = 0.6    # random noise spread
    vmax = 1.2
    size = 180
    pad = 10

P = Params()

# =========================================================
# 4) Build environment
# =========================================================
img = make_letter_N_numpy(P.size, pad=P.pad)
gray, gx, gy = build_field(img, h=25)
N = gray.shape[0]

# =========================================================
# 5) Initialize swarm
# =========================================================
rng = np.random.default_rng(2)
pos = rng.random((P.nrobot, 2)) * N

# =========================================================
# 6) Gradient helper
# =========================================================
def grad_at(x, y):
    ix = np.clip(x.astype(int), 1, N-2)
    iy = np.clip(y.astype(int), 1, N-2)
    return -gx[iy, ix], -gy[iy, ix]  # move toward dark region

# =========================================================
# 7) Step update (vectorized)
# =========================================================
def step():
    global pos
    gx_i, gy_i = grad_at(pos[:,0], pos[:,1])
    v = np.stack([gx_i, gy_i], axis=1)
    norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    v = (v / norms) * P.k_grad
    # add diffusion (random walk)
    v += P.diffusion * rng.standard_normal(v.shape)
    # move
    pos += v * P.dt
    pos = np.clip(pos, 1, N-2)


# =========================================================
# 8) Animation
# =========================================================
fig, ax = plt.subplots(figsize=(6,6))
ax.imshow(gray, cmap='gray', extent=[0, N, 0, N], origin='lower')
sc = ax.scatter(pos[:,0], pos[:,1], c='cyan', s=10, edgecolors='none')
ax.set_xticks([]); ax.set_yticks([])
title = ax.set_title("")

def update(frame):
    step()
    # color by local brightness
    gval = gray[np.clip(pos[:,1].astype(int),0,N-1), np.clip(pos[:,0].astype(int),0,N-1)]
    sc.set_offsets(pos)
    sc.set_facecolors(plt.cm.plasma(1 - gval))
    title.set_text(f'Fast N Assembly — t={frame*P.dt:.1f}')
    return sc, title

ani = FuncAnimation(fig, update, frames=P.steps, interval=50, blit=False)
plt.show()
