from jaxtyping import Float
import torch
from torch import Tensor


def render_point_cloud(
    vertices: Float[Tensor, "vertex 3"],
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
    resolution: tuple[int, int] = (256, 256),
) -> Float[Tensor, "batch height width"]:
    """Create a white canvas with the specified resolution. Then, transform the points
    into camera space, project them onto the image plane, and color the corresponding
    pixels on the canvas black.
    """

    """渲染3D点云到2D图像
    
    Args:
        vertices: 世界坐标系下的3D点云 [N,3]
        extrinsics: 相机外参矩阵 [B,4,4]
        intrinsics: 相机内参矩阵 [B,3,3]
        resolution: 输出图像分辨率 (H,W)
    
    Returns:
        渲染图像 [B,H,W]（白色背景，黑色点）
    """
    device = vertices.device
    B = extrinsics.shape[0]
    H, W = resolution
    
    # 1. 转换为齐次坐标 [N,4]
    vertices_homo = torch.cat([vertices, torch.ones_like(vertices[:, :1])], dim=-1)
    
    # 2. 世界->相机坐标系 [B,N,4]
    cam_coords = torch.einsum('bij,nj->bni', extrinsics, vertices_homo)
    
    # 3. 投影到像素坐标 [B,N,2]
    uv = torch.einsum('bij,bnj->bni', intrinsics, cam_coords[:, :, :3])  # [B,N,3]
    uv = uv[..., :2] / uv[..., 2:].clamp(min=1e-6)  # 透视除法
    
    # 4. 创建画布 [B,H,W]
    images = torch.ones((B, H, W), device=device)
    
    # 5. 渲染点（将对应像素设为0）
    uv_round = uv.round().long()  # 四舍五入为整数坐标
    valid_x = (uv_round[..., 0] >= 0) & (uv_round[..., 0] < W)
    valid_y = (uv_round[..., 1] >= 0) & (uv_round[..., 1] < H)
    valid = valid_x & valid_y
    
    for b in range(B):
        # 获取当前批次的有效点
        cur_uv = uv_round[b][valid[b]]
        images[b, cur_uv[:, 1], cur_uv[:, 0]] = 0  # 图像坐标系y向下
        
    return images
