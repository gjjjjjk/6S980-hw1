from jaxtyping import Float,jaxtyped
from torch import Tensor
import torch
import cv2
import numpy as np
from typing import Tuple
from pathlib import Path
from jaxtyping import Float
from numpy.typing import NDArray
from scipy.linalg import svd
from scipy.optimize import least_squares
import trimesh

def load_dataset(path: Path, grayscale: bool = True
                 ) -> NDArray[np.float32]:
    """ load images to convert tensor"""
    # Get all image paths
    image_paths = list(path.glob("*.*"))  # Adjust pattern if needed (e.g., "*.jpg")
    if not image_paths:
        raise FileNotFoundError(f"No images found in {path}")
    
    images = []
    for img_path in image_paths:
        # Read image (BGR format in OpenCV)
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
        
        # Convert to grayscale if needed (for RGB inputs)
        if not grayscale and len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Normalize to [0, 1] and add to list
        img = img.astype(np.float32)
        images.append(img)
    
    return images

    
def Finding_correspondences_and_generate_F(
        images : NDArray[np.float32]) -> Tuple [Float[Tensor, "*batch 3 3"],
                                                 NDArray[np.float32],
                                                 NDArray[np.float32],
                                                 NDArray[np.float32],
                                                 NDArray[np.float32],
                                                 NDArray[np.float32]]:
    """ use SIFT algorithm to generate keypoints"""

    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    flann = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 5}, {'checks': 50})

    all_kps = []
    all_descs = []
    
    for img in range(images):
        # Detect keypoints and describ 
        kps, descs = sift.detectAndCompute(img, None)
        
        all_kps.append(kps)
        all_descs.append(descs)
        
    for i in range(images - 1):
        desc1 = all_descs[i]
        desc2 = all_descs[i + 1]

        # FLANN匹配
        matches = flann.knnMatch(desc1, desc2, k=2)
    
        # 比率测试筛选优质匹配
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        # 提取匹配点坐标
        src_pts = np.float32([all_kps[i][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([all_kps[i+1][m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        if len(src_pts) >= 8:
            F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC, ransacReprojThreshold=1.0, confidence=0.99)
            inliers_num = mask.ravel().tolist().count(1)  # 统计内点数量
            inlier_matches = [m for i, m in enumerate(good_matches) if mask[i]]
            print(f"基础矩阵:\n{F}")
            print(f"内点数量: {inliers_num}/{len(good_matches)}")
        else:
            print("匹配点不足8对，无法估计基础矩阵！")

    return F, inlier_matches, src_pts, dst_pts, all_kps, all_descs

def Fundamental(F : Float[Tensor, "*batch 3 3"]
                ) -> Tuple[Float[Tensor, "*batch 3 3"],Float[Tensor, "*batch 3 3"]]:
    """ use SVD resolution to generate spin martix R and vector t . to cross 
    product between R and t"""

    # 第一个相机 P1 = [I | 0]
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    
    # 计算极点 e2（F^T 的右零空间）
    U, S, Vt = svd(F.T)
    e2 = Vt[-1, :]  # 最后一列
    e2 = e2 / e2[2]  # 齐次坐标归一化

    # 反对称矩阵 [e2]×
    e2_skew = np.array([
        [0, -e2[2], e2[1]],
        [e2[2], 0, -e2[0]],
        [-e2[1], e2[0], 0]
    ])
    
    # P2 = [[e2]× * F | λ * e2]（λ=1）
    P2 = np.hstack([e2_skew @ F, e2.reshape(3, 1)])
    
    return P1, P2

def triangulate_points(P1, P2, inlier_matches, kps, pts1, pts2) -> NDArray[np.float32]:
    
    points_3d = []
    for m in inlier_matches:
        x1 = np.array([kps[0][m.queryIdx].pt[0], kps[0][m.queryIdx].pt[1], 1])
        x2 = np.array([kps[1][m.trainIdx].pt[0], kps[1][m.trainIdx].pt[1], 1])
        X = cv2.triangulatePoints(P1, P2, x1[:2], x2[:2])
        X /= X[3]  # 齐次坐标归一化
        points_3d.append(X[:3])
    
    for i in range(len(points_3d)):
        res = least_squares(reprojection_error, points_3d[i], args=(pts1[i], pts2[i], P1, P2))
        points_3d[i] = res.x
    
    # tensor_points_3d = torch.tensor(points_3d, dtype=torch.float32) 

    return points_3d    
    
def reprojection_error(X, x1, x2, P1, P2):
    proj1 = P1 @ np.append(X, 1)
    proj1 = proj1[:2] / proj1[2]
    proj2 = P2 @ np.append(X, 1)
    proj2 = proj2[:2] / proj2[2]
    error = np.hstack([x1 - proj1, x2 - proj2])
    return error

def plot_model(points : NDArray[np.float32]):
    # Ball Pivoting算法
    mesh = trimesh.Trimesh(vertices=points)
    mesh = trimesh.repair.broken_faces(mesh)  # 修复可能的面片问题

    # 保存为OBJ
    mesh.export('output.obj')

    # 可视化
    mesh.show()


def closest_points_between_rays(P1, d1, P2, d2):
    """
    计算两条射线之间的最近点。
    
    参数:
        P1, P2: 射线起点的坐标 (np.array, shape=(3,)).
        d1, d2: 射线的方向向量 (np.array, shape=(3,)), 需为单位向量.
    
    返回:
        Q1: 射线1上的最近点.
        Q2: 射线2上的最近点.
        distance: 最近距离.
    """
    # 确保方向向量是单位向量
    d1 = d1 / np.linalg.norm(d1)
    d2 = d2 / np.linalg.norm(d2)
    
    # 计算叉积和差值向量
    d1_cross_d2 = np.cross(d1, d2)
    P1_minus_P2 = P1 - P2
    
    # 处理平行射线（叉积为零向量）
    if np.linalg.norm(d1_cross_d2) < 1e-6:
        # 平行射线，任选起点计算距离
        t = np.dot(P2 - P1, d1) / np.dot(d1, d1)
        Q1 = P1 + t * d1
        Q2 = P2
        distance = np.linalg.norm(Q1 - Q2)
        return Q1, Q2, distance
    
    # 计算参数 s 和 t
    s = (np.cross(d2, P1_minus_P2) @ d1_cross_d2) / (np.linalg.norm(d1_cross_d2) ** 2)
    t = (np.cross(d1, P1_minus_P2) @ d1_cross_d2) / (np.linalg.norm(d1_cross_d2) ** 2)
    
    # 计算最近点
    Q1 = P1 + s * d1
    Q2 = P2 + t * d2
    distance = np.linalg.norm(Q1 - Q2)
    
    return Q1, Q2, distance