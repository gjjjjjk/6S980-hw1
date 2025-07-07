from jaxtyping import Float,install_import_hook
import torch
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
import cv2

with install_import_hook(("src",), ("beartype", "beartype")):
    from src.geometry import project,homogenize_vectors,transform_world2cam

def render_point_cloud(
    vertices: Float[Tensor, "vertex 3"],
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
    resolution: tuple[int, int] = (256, 256),
) -> Float[Tensor, "*batch height width"]:
    """Create a white canvas with the specified resolution. Then, transform the points
    into camera space, project them onto the image plane, and color the corresponding
    pixels on the canvas black.
    """

    
    device = vertices.device
    B = extrinsics.shape[0]
    H, W = resolution
    batch = []
    xlim: tuple[float, float] = (0, 2.0)
    ylim: tuple[float, float] = (0, 2.0)
    # 1. 转换为齐次坐标 [N, 4]
    vertices_homo = homogenize_vectors(vertices)

    print(vertices_homo.shape)
    # 2. 世界->相机坐标系 [16, B, 4]
    cam_coords = transform_world2cam(vertices_homo, extrinsics)
    
    # 3. 投影到像素坐标 [16, B,2]
    uv = project(cam_coords,intrinsics)  # [16, B, 2]
    
    uv = uv.cpu()

    group,_, _ = uv.shape

    
    # for b in range(group):
    #     alpha: float = 0.5
    #     uv2 = uv[b]
    #     # print(uv2.shape)
    #     fig = plt.figure(figsize=(6, 6))
        
    #     ax = fig.add_subplot(111)  # 2D 坐标系
    #     # ax.set_xlabel("x")  # x 轴标签
    #     # ax.set_ylabel("y")  # y 轴标签
    #     ax.set_xlim(xlim)   # x 轴范围（如 [x_min, x_max]）
    #     ax.set_ylim(ylim)   # y 轴范围
    #     ax.scatter(*uv2.T, alpha=alpha, marker=",", lw=0.5, s=1, color="black")
    #     plt.savefig(f"output/1_projection/view_{b:0>2}.png") 

        # img_tensor = torch.rand(b, H, W, dtype=torch.float32)
    canvas = torch.ones((B, H, W), device=device)
    uv2 = uv[..., ] * 128
    pixel_coords = uv2[..., :2].round().long()  # [batch, valid_points, 2]

    x_valid = (pixel_coords[..., 0] >= 0) & (pixel_coords[..., 0] < W)
    y_valid = (pixel_coords[..., 1] >= 0) & (pixel_coords[..., 1] < H)
    valid_mask = x_valid & y_valid
    
    for b in range(B):
        batch_pixels = pixel_coords[b][valid_mask[b]]  # [N, 2]
        canvas[b, batch_pixels[:, 1], batch_pixels[:, 0]] = 0  # 图像坐标系y向下
    
    return canvas
        
    
