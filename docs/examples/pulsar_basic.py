#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This example demonstrates the most trivial, direct interface of the pulsar
sphere renderer. It renders and saves an image with 10 random spheres.
Output: basic.png.
"""
import logging
import math
from os import path

import imageio
import torch
from pytorch3d.renderer.points.pulsar import Renderer
import vedo


LOGGER = logging.getLogger(__name__)

def pos_to_col(vert_pos):
    _min = vert_pos.min(dim=0)[0]
    _max = vert_pos.max(dim=0)[0]
    vert_col = (vert_pos - _min) / (_max  - _min)
    return vert_col

def cli():
    """
    Basic example for the pulsar sphere renderer.

    Writes to `basic.png`.
    """
    LOGGER.info("Rendering on GPU...")
    torch.manual_seed(1)
    n_points = 10000
    width = 400
    height = 400
    device = torch.device("cuda")
    # The PyTorch3D system is right handed; in pulsar you can choose the handedness.
    # For easy reproducibility we use a right handed coordinate system here.
    renderer = Renderer(width, height, n_points, right_handed_system=True, n_track=256).to(device)
    # Generate sample data.
    vert_pos = torch.rand(n_points, 3, dtype=torch.float32, device=device) * 10.0
    vert_pos[:, 2] += 25.0
    vert_pos[:, :2] -= 5.0
    vert_col = torch.rand(n_points, 3, dtype=torch.float32, device=device, requires_grad=True)
    opacity = torch.rand(n_points, dtype=torch.float32, device=device) * 1
    vert_rad = torch.ones(n_points, dtype=torch.float32, device=device) * 3
    opacity.requires_grad = True
    cam_params = torch.tensor(
        [
            0.0,
            0.0,
            0.0,  # Position 0, 0, 0 (x, y, z).
            0.0,
            math.pi,  # Because of the right handed system, the camera must look 'back'.
            0.0,  # Rotation 0, 0, 0 (in axis-angle format).
            5.0,  # Focal length in world size.
            2.0,  # Sensor size in world size.
        ],
        dtype=torch.float32,
        device=device,
    )
    # Render.
    image, _ = renderer(
        vert_pos,
        vert_col,
        vert_rad,
        cam_params,
        1.0e-1,  # Renderer blending parameter gamma, in [1., 1e-5].
        max_depth=40.0,  # Maximum depth.
        min_depth=20.0,
        opacity=opacity,
        mode=3,
        return_forward_info=True,
    )

    # x, y = 650, 300
    # print (y, x, image[y, x])
    # image[y, x].sum().backward()
    # print (vert_col.grad)
    # print (opacity.grad)

    LOGGER.info("Writing image to `%s`.", path.abspath("basic.png"))
    imageio.imsave("basic.png", (image.cpu().detach() * 255.0).to(torch.uint8).numpy())
    LOGGER.info("Done.")

    # obj_list = []
    # for i in range(n_points):
    #     pt = vedo.Points(
    #         vert_pos.cpu().numpy()[i:i+1], 
    #         r=vert_rad.cpu().numpy()[i:i+1] * 100
    #     ).c(vert_col.detach().cpu().numpy()[i])
    #     obj_list.append(pt)
    # vedo.show(*obj_list)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cli()
