from jaxtyping import Float
from torch import Tensor
import torch


def homogenize_points(
    points: Float[Tensor, "*batch dim"],
) -> Float[Tensor, "*batch dim+1"]:
    """Turn n-dimensional points into (n+1)-dimensional homogeneous points."""
    ones = torch.ones_like(points[..., :1])
    return torch.cat([points, ones], dim=-1)
    


def homogenize_vectors(
    points: Float[Tensor, "*batch dim"],
) -> Float[Tensor, "*batch dim+1"]:
    """Turn n-dimensional vectors into (n+1)-dimensional homogeneous vectors."""

    ones = torch.ones_like(points[..., :1])
    return torch.cat([points, ones], dim=-1)
    


def transform_rigid(
    xyz: Float[Tensor, "*#batch 4"],
    transform: Float[Tensor, "*#batch 4 4"],
) -> Float[Tensor, "*batch 4"]:
    """Apply a rigid-body transform to homogeneous points or vectors."""
    # 确保输入维度正确
    assert xyz.size(-1) == 4, "输入必须是齐次坐标"
    assert transform.size(-1) == 4 and transform.size(-2) == 4, "需要4x4变换矩阵"
    
    # 应用变换：x' = T @ x
    # 使用torch.matmul自动处理批次维度
    return torch.matmul(transform, xyz.unsqueeze(-1)).squeeze(-1)
    


def transform_world2cam(
    xyz: Float[Tensor, "*#batch 4"],
    cam2world: Float[Tensor, "*#batch 4 4"],
) -> Float[Tensor, "*batch 4"]:
    """Transform points or vectors from homogeneous 3D world coordinates to homogeneous
    3D camera coordinates.
    """

    # 输入验证
    assert xyz.size(-1) == 4, "输入必须是齐次坐标"
    assert cam2world.size(-1) == 4 and cam2world.size(-2) == 4, "需要4x4变换矩阵"
    
    # 计算世界->相机的变换：world2cam = inv(cam2world)
    world2cam = torch.inverse(cam2world)
    
    # 应用变换：x_cam = world2cam @ x_world
    return torch.matmul(world2cam, xyz.unsqueeze(-1)).squeeze(-1)


def transform_cam2world(
    xyz: Float[Tensor, "*#batch 4"],
    cam2world: Float[Tensor, "*#batch 4 4"],
) -> Float[Tensor, "*batch 4"]:
    """Transform points or vectors from homogeneous 3D camera coordinates to homogeneous
    3D world coordinates.
    """

    # 输入验证
    assert xyz.size(-1) == 4, "输入必须是齐次坐标"
    assert cam2world.size(-1) == 4 and cam2world.size(-2) == 4, "需要4x4变换矩阵"
    
    # 直接应用cam2world变换：x_world = cam2world @ x_cam
    return torch.einsum('...ij,...j->...i', cam2world, xyz)

def project(
    xyz: Float[Tensor, "*#batch 4"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
) -> Float[Tensor, "*batch 2"]:
    """Project homogenized 3D points in camera coordinates to pixel coordinates."""
    
    """将相机坐标系下的3D点投影到像素坐标系
    
    Args:
        xyz: 齐次坐标，可以是点(x,y,z,1)或方向向量(x,y,z,0)
        intrinsics: 相机内参矩阵 [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    
    Returns:
        像素坐标 [...,2] (u,v)
    """
    # 输入验证
    assert xyz.size(-1) == 4, "输入必须是齐次坐标"
    assert intrinsics.size(-1) == 3 and intrinsics.size(-2) == 3, "需要3x3内参矩阵"
    
    # 透视除法 (x/z, y/z)，处理w分量和z分量
    z = xyz[..., 2:3] / xyz[..., 3:4]  # 处理齐次坐标w分量
    uv = xyz[..., :2] / z.clamp(min=1e-6)  # 避免除零
    
    # 应用内参变换 [u,v,1]^T = K @ [x/z, y/z, 1]^T
    return torch.einsum('...ij,...j->...i', intrinsics[..., :2, :], 
                       torch.cat([uv, torch.ones_like(uv[..., :1])], dim=-1))[..., :2]

