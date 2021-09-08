import logging
import math
from os import path
import numpy as np
import collections
import cv2

import imageio
import torch
from pytorch3d.transforms import axis_angle_to_matrix


torch.manual_seed(1)
n_points = 10
width = 400
height = 400
device = torch.device("cuda")
radius = 3
x, y = 109, 226

Rays = collections.namedtuple("Rays", ("origins", "directions", "viewdirs"))

def generate_rays(w, h, focal, camtoworlds):
    """
    Generate perspective camera rays. Principal point is at center.
    Args:
        w: int image width
        h: int image heigth
        focal: float real focal length
        camtoworlds: jnp.ndarray [B, 4, 4] c2w homogeneous poses
        equirect: if true, generates spherical rays instead of pinhole
    Returns:
        rays: Rays a namedtuple(origins [B, 3], directions [B, 3], viewdirs [B, 3])
    """
    x, y = np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
        np.arange(w, dtype=np.float32),  # X-Axis (columns)
        np.arange(h, dtype=np.float32),  # Y-Axis (rows)
        indexing="xy",
    )

    camera_dirs = np.stack(
        [
            (x + 0.5 - w * 0.5) / focal,
            -(y + 0.5 - h * 0.5) / focal,
            -np.ones_like(x),
        ],
        axis=-1,
    )

    c2w = camtoworlds[:, None, None, :3, :3]
    camera_dirs = camera_dirs[None, Ellipsis, None]
    directions = np.matmul(c2w, camera_dirs)[Ellipsis, 0]
    origins = np.broadcast_to(
        camtoworlds[:, None, None, :3, -1], directions.shape
    )
    norms = np.linalg.norm(directions, axis=-1, keepdims=True)
    viewdirs = directions / norms
    rays = Rays(
        origins=origins, directions=directions, viewdirs=viewdirs
    )
    return rays

# Generate sample data.
vert_pos = torch.rand(n_points, 3, dtype=torch.float32, device=device) * 10.0
vert_pos[:, 2] += 25.0
vert_pos[:, :2] -= 5.0
vert_col = torch.rand(n_points, 3, dtype=torch.float32, device=device)
opacity = torch.rand(n_points, dtype=torch.float32, device=device) * 1
cam_params = torch.tensor(
    [
        0.0,
        0.0,
        0.0,  # Position 0, 0, 0 (x, y, z).
        0.0,
        math.pi,  # Because of the right handed system, the camera must look 'back'.
        0.0,  # Rotation 0, 0, 0 (in axis-angle format).
        5.0,  # Focal length in world size.
        2.0,  # Sensor size in world size (width).
    ],
    dtype=torch.float32,
    device=device,
)

# convert camera format
cam_pos = cam_params[0:3]
cam_R = axis_angle_to_matrix(cam_params[3:6])
cam_focal = cam_params[6] / cam_params[7] * width
cam_T = - torch.matmul(cam_R, cam_pos)
camtoworld = torch.cat([cam_R.t(), -cam_T[:, None]], dim=1)
camtoworld = torch.cat([
    camtoworld, torch.tensor([[0., 0., 0., 1.]], device=camtoworld.device)
], dim=0)

# generate rays for the entire image
rays = generate_rays(
    width, height, cam_focal.cpu(), camtoworld[None, :].cpu().numpy()) 

# samples
n_samples = 64
min_dist = 20
max_dist = 40
t = np.linspace(min_dist, max_dist, n_samples)
print ("t:", t)
samples = rays.origins + rays.viewdirs * t[:, None, None, None]
# print (samples.shape) # [(64, 500, 500, 3)]
samples = torch.from_numpy(samples).to(device)

# distance of the samples and points
p = 6
dists = torch.linalg.norm(samples - vert_pos[:, None, None, None, :], dim=-1)
# print (dists.shape) # [10, 64, 500, 500]
weights = 1.0 / torch.clamp(dists ** p, min=1e-308)

# weights = weights * (dists < radius).float()  # xyz limits: causes aliasing artifacts
viewdirs = torch.from_numpy(rays.viewdirs).to(device)
ray_dists = torch.linalg.norm(
    vert_pos[:, None, None, None] 
    - viewdirs[None, ...] * torch.einsum("bhwi,ni->nbhw", viewdirs, vert_pos)[..., None],
    dim = -1
) # [10, 1, 500, 500]
weights = weights * (ray_dists < radius).float()  # xy limits is better

dinominator = torch.sum(weights, dim=0)

samples_col = torch.sum(
    vert_col[:, None, None, None, :] * weights[..., None], dim=0
) / torch.clamp(dinominator[..., None], min=1e-308)
# TODO: it might not be right to interpolate density!
samples_opy = torch.sum(
    opacity[:, None, None, None] * weights[...], dim=0
) / torch.clamp(dinominator[...], min=1e-308)
print (samples_col.shape, samples_opy.shape)

print ("vert_col", vert_col)
print ("weights", weights[:, :, y, x])
print ("dinominator", dinominator[:, y, x])
print ("samples_col", samples_col[:, y, x])

# for volumetric rendering
rgb = samples_col.permute(1, 2, 0, 3).view(-1, n_samples, 3)
sigma = samples_opy.permute(1, 2, 0).view(-1, n_samples, 1)
z_vals = torch.from_numpy(t[None, :]).to(device)
dirs = torch.from_numpy(rays.viewdirs).to(device).view(-1, 3)

def volumetric_rendering(rgb, sigma, z_vals, dirs, white_bkgd):
    """Volumetric Rendering Function.
    Args:
      rgb: torch.tensor(float32), color, [batch_size, num_samples, 3]
      sigma: torch.tensor(float32), density, [batch_size, num_samples, 1].
      z_vals: torch.tensor(float32), [batch_size, num_samples].
      dirs: torch.tensor(float32), [batch_size, 3].
      white_bkgd: bool.
    Returns:
      comp_rgb: torch.tensor(float32), [batch_size, 3].
      disp: torch.tensor(float32), [batch_size].
      acc: torch.tensor(float32), [batch_size].
      weights: torch.tensor(float32), [batch_size, num_samples]
    """
    eps = 1e-10
    dists = torch.cat(
        [
            z_vals[Ellipsis, 1:] - z_vals[Ellipsis, :-1],
            torch.tensor(
                [1e10], dtype=z_vals.dtype, device=z_vals.device
            ).expand(z_vals[Ellipsis, :1].shape),
        ],
        -1,
    )
    dists = dists * torch.linalg.norm(dirs[Ellipsis, None, :], dim=-1)
    # Note that we're quietly turning sigma from [..., 0] to [...].
    alpha = 1.0 - torch.exp(-sigma[Ellipsis, 0] * dists)
    accum_prod = torch.cat(
        [
            torch.ones_like(alpha[Ellipsis, :1]),
            torch.cumprod(1.0 - alpha[Ellipsis, :-1] + eps, dim=-1),
        ],
        dim=-1,
    )
    weights = alpha * accum_prod

    comp_rgb = (weights[Ellipsis, None] * rgb).sum(dim=-2)
    depth = (weights * z_vals).sum(dim=-1)
    acc = weights.sum(dim=-1)  # Alpha
    # Equivalent to (but slightly more efficient and stable than):
    #  disp = 1 / max(eps, where(acc > eps, depth / acc, 0))
    inv_eps = 1 / eps
    disp = (acc / depth).double()  # torch.where accepts <scaler, double tensor> 
    disp = torch.where((disp > 0) & (disp < inv_eps) & (acc > eps), disp, inv_eps)
    disp = disp.float()
    if white_bkgd:
        comp_rgb = comp_rgb + (1.0 - acc[Ellipsis, None])
    return comp_rgb, disp, acc, weights

comp_rgb, disp, acc, weights = volumetric_rendering(
    rgb, sigma, z_vals, dirs, white_bkgd=True)
img = comp_rgb.view(height, width, 3)
cv2.imwrite("./nerf_naive.jpg", np.uint8(img.cpu().numpy() * 255)[:, :, ::-1])