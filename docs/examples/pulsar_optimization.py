#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This example demonstrates scene optimization with the plain
pulsar interface. For this, a reference image has been pre-generated
(you can find it at `../../tests/pulsar/reference/examples_TestRenderer_test_smallopt.png`).
The scene is initialized with random spheres. Gradient-based
optimization is used to converge towards a faithful
scene representation.
"""
import logging
import math

import cv2
import imageio
import numpy as np
import torch
from pytorch3d.renderer.points.pulsar import Renderer
from torch import nn, optim


LOGGER = logging.getLogger(__name__)
N_POINTS = 200_000
WIDTH = 500
HEIGHT = 500
DEVICE = torch.device("cuda")


class SceneModel(nn.Module):
    """
    A simple scene model to demonstrate use of pulsar in PyTorch modules.

    The scene model is parameterized with sphere locations (vert_pos),
    channel content (vert_col), radiuses (vert_rad), camera position (cam_pos),
    camera rotation (cam_rot) and sensor focal length and width (cam_sensor).

    The forward method of the model renders this scene description. Any
    of these parameters could instead be passed as inputs to the forward
    method and come from a different model.
    """

    def __init__(self):
        super(SceneModel, self).__init__()
        self.gamma = 1.0
        # Points.
        torch.manual_seed(1)
        vert_pos = torch.rand(N_POINTS, 3, dtype=torch.float32) * 10.0
        vert_pos[:, 2] += 25.0
        vert_pos[:, :2] -= 5.0
        self.register_parameter("vert_pos", nn.Parameter(vert_pos, requires_grad=False))
        self.register_parameter(
            "vert_col",
            nn.Parameter(
                torch.ones(N_POINTS, 3, dtype=torch.float32) * 0.5, requires_grad=True
            ),
        )
        self.register_parameter(
            "vert_rad",
            nn.Parameter(
                torch.ones(N_POINTS, dtype=torch.float32) * 0.05, requires_grad=False
            ),
        )
        self.register_parameter(
            "opacity",
            nn.Parameter(
                torch.ones(N_POINTS, dtype=torch.float32) * 1e19, requires_grad=True
            ),
        )
        self.register_buffer(
            "cam_params",
            torch.tensor(
                [0.0, 0.0, 0.0, 0.0, math.pi, 0.0, 5.0, 2.0], dtype=torch.float32
            ),
        )
        # The volumetric optimization works better with a higher number of tracked
        # intersections per ray.
        self.renderer = Renderer(
            WIDTH, HEIGHT, N_POINTS, n_track=1, right_handed_system=True
        )

    def forward(self):
        import random
        return self.renderer.forward(
            self.vert_pos,
            self.vert_col,
            self.vert_rad,
            self.cam_params,
            self.gamma,
            45.0,
            # opacity=torch.nn.functional.softplus(self.opacity),
            mode=2,
            return_forward_info=True,
        )


def cli():
    """
    Scene optimization example using pulsar.
    """
    LOGGER.info("Loading reference...")
    # Load reference.
    ref = (
        torch.from_numpy(
            cv2.resize(
                imageio.imread(
                    "../../tests/pulsar/reference/examples_TestRenderer_test_smallopt.png"
                ),
                (WIDTH, HEIGHT),
            )[:, ::-1, :].copy()
        ).to(torch.float32)
        / 255.0
    ).to(DEVICE)
    # Set up model.
    model = SceneModel().to(DEVICE)
    # Optimizer.
    optimizer = optim.SGD(
        [
            {"params": [model.vert_col], "lr": 1e-3},
            # {"params": [model.vert_rad], "lr": 5e-3},
            # {"params": [model.vert_pos], "lr": 1e-2},
            # {"params": [model.opacity], "lr": 1e29},
        ]
    )
    LOGGER.info("Optimizing...")
    # Optimize.
    for i in range(1500):
        optimizer.zero_grad()
        result, result_info = model()
        # Visualize.
        result_im = (result.cpu().detach().numpy() * 255).astype(np.uint8)
        cv2.imshow("opt", result_im[:, :, ::-1])
        overlay_img = np.ascontiguousarray(
            ((result * 0.5 + ref * 0.5).cpu().detach().numpy() * 255).astype(np.uint8)[
                :, :, ::-1
            ]
        )
        overlay_img = cv2.putText(
            overlay_img,
            "Step %d" % (i),
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
            False,
        )
        cv2.imshow("overlay", overlay_img)
        cv2.waitKey(1)
        result.retain_grad()
        # Update.
        loss = ((result - ref) ** 2).sum()
        LOGGER.info("loss %d: %f", i, loss.item())
        loss.backward()
        print (
            "python sphere grad:", 
            model.vert_col.grad[68599],
            model.vert_col[68599] 
        )
        print (
            "python image grad:",
            result.grad[10, 10],
            result[10, 10],
            ref[10, 10],
        )
        print ("\n")
        optimizer.step()
        # print (result_info[50, 375][4::2].long())
        # exit()
        # Cleanup.
        # with torch.no_grad():
        #     model.vert_col.data = torch.clamp(model.vert_col.data, 0.0, 1.0)
        #     # Remove points.
        #     model.vert_pos.data[model.vert_rad < 0.001, :] = -1000.0
        #     model.vert_rad.data[model.vert_rad < 0.001] = 0.0001
        #     vd = (
        #         (model.vert_col - torch.ones(3, dtype=torch.float32).to(DEVICE))
        #         .abs()
        #         .sum(dim=1)
        #     )
        #     model.vert_pos.data[vd <= 0.2] = -1000.0
    LOGGER.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cli()
