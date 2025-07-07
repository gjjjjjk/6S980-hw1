from jaxtyping import Float,jaxtyped
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
    print(transform)
    # 应用变换：x' = T @ x
    # 使用torch.matmul自动处理批次维度
    return torch.matmul(transform, xyz.unsqueeze(-1)).squeeze(-1)
    


def transform_world2cam(
    xyz: Float[Tensor, "*#batch 4"],
    cam2world: Float[Tensor, "*_batch 4 4"],
) -> Float[Tensor, "6 *batch 4"]:
    """Transform points or vectors from homogeneous 3D world coordinates to homogeneous
    3D camera coordinates.
    """

    # 输入验证
    assert xyz.size(-1) == 4, "输入必须是齐次坐标"
    assert cam2world.size(-1) == 4 and cam2world.size(-2) == 4, "需要4x4变换矩阵"
    
    # 计算世界->相机的变换：world2cam = inv(cam2world)
    world2cam = torch.inverse(cam2world)

    # 应用变换：x_cam = world2cam @ x_world
    return  torch.einsum('bij,cj->bci', world2cam, xyz)


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
    intrinsics: Float[Tensor, "*_batch 3 3"],
) -> Float[Tensor, "6 *#batch 2"]:
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
    
    # intrinsics[..., :2 , 0:2] = 1.0/256
    # intrinsics[..., :2 , 2:3] = 0.5/256
    k0 = torch.cat([intrinsics, torch.zeros_like(intrinsics[..., :1])],dim=-1)
    print(xyz.shape)
    print(k0.shape)
    i = torch.einsum('bij,bcj->bci', k0, xyz)
    print(i.shape)
    # i2 = i[...,:2] / i[..., 2:3].clamp(min=1e-6)
    #print(i2.shape)
    i2 = i[..., :2] / i[..., 2:3]
    print(i2.shape)
    return i2

