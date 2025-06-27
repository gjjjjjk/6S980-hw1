from pathlib import Path
from typing import Literal, TypedDict, Optional
import json
from jaxtyping import Float
from torch import Tensor
import torch
from PIL import Image
import torchvision.transforms as T

def load_image(img_path: Path, target_size: Optional[tuple] = None) -> torch.Tensor:
    """加载图像并转换为PyTorch张量
    
    Args:
        img_path: 图像路径
        target_size: 可选，目标尺寸 (height, width)
    
    Returns:
        Tensor: 标准化后的图像张量 [C,H,W]，值范围[-1,1]
    """
    # 1. 定义预处理流程
    transforms = [
        T.ToTensor(),  # 转为张量并自动缩放到[0,1]
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化到[-1,1]
    ]
    
    # 2. 可选：调整大小
    if target_size:
        transforms.insert(0, T.Resize(target_size))
    
    transform = T.Compose(transforms)
    
    # 3. 加载图像（确保RGB格式）
    with Image.open(img_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return transform(img)

class PuzzleDataset(TypedDict):
    extrinsics: Float[Tensor, "batch 4 4"]
    intrinsics: Float[Tensor, "batch 3 3"]
    images: Float[Tensor, "batch height width"]


def load_dataset(path: Path) -> PuzzleDataset:
    """Load the dataset into the required format."""

    """加载拼图数据集
    
    Args:
        path: 数据集目录路径，应包含：
            - images/ : 拼图块图像目录
            - edges.json : 边缘匹配关系文件
    
    Returns:
        PuzzleDataset 实例
    """
    # 1. 加载所有拼图块图像
    images_dir = path / "images"
    pieces = {}
    for img_file in images_dir.glob("*.png"):  # 假设为PNG格式
        piece_id = img_file.stem  # 去除扩展名的文件名作为ID
        image = load_image(img_file)  # 需要实现图像加载函数
        pieces[piece_id] = image
    
    # 2. 加载边缘关系
    edges_file = path / "edges.json"
    with open(edges_file, 'r') as f:
        edges = json.load(f)  # 格式: {"piece1": ["piece2", "piece3"], ...}
    
    # 3. 转换为数值ID（如果需要）
    piece_to_idx = {p: i for i, p in enumerate(pieces.keys())}
    edges_idx = {
        piece: [piece_to_idx[adj] for adj in adjs] 
        for piece, adjs in edges.items()
    }
    
    return PuzzleDataset(pieces, edges_idx)


def convert_dataset(dataset: PuzzleDataset) -> PuzzleDataset:
    """Convert the dataset into OpenCV-style camera-to-world format. As a reminder, this
    format has the following specification:

    - The camera look vector is +Z.
    - The camera up vector is -Y.
    - The camera right vector is +X.
    - The extrinsics are in camera-to-world format, meaning that they transform points
      in camera space to points in world space.

    """

    """将数据集转换为OpenCV风格的相机坐标系格式
    
    规范说明：
    - 相机朝向：+Z轴方向
    - 相机上方向：-Y轴方向
    - 相机右方向：+X轴方向
    - 外参矩阵格式：camera-to-world（将相机空间的点转换到世界空间）
    
    Args:
        dataset: 原始数据集（假设使用其他坐标系格式）
    
    Returns:
        转换后的新PuzzleDataset实例
    """
    # 1. 创建坐标系转换矩阵（从当前格式到OpenCV格式）
    # 假设原始数据使用Y-up的右手坐标系（如Blender风格）
    convert_matrix = torch.tensor([
        [1,  0,  0, 0],  # X保持不变
        [0, -1,  0, 0],  # Y反向（Y-up → -Y-up）
        [0,  0, -1, 0],  # Z反向（Z-forward → Z-backward）
        [0,  0,  0, 1]   # 齐次坐标
    ], dtype=torch.float32)
    
    # 2. 转换所有外参矩阵
    converted_extrinsics = {}
    for piece_id, extrinsic in dataset.extrinsics.items():
        # 应用转换：T_opencv = T_original @ convert_matrix
        converted_extrinsics[piece_id] = extrinsic @ convert_matrix
    
    # 3. 返回新数据集（保持其他数据不变）
    return PuzzleDataset(
        pieces=dataset.pieces,
        edges=dataset.edges,
        extrinsics=converted_extrinsics,
        intrinsics=dataset.intrinsics
    )


def quiz_question_1() -> Literal["w2c", "c2w"]:
    """In what format was your puzzle dataset?"""

    raise NotImplementedError("This is your homework.")


def quiz_question_2() -> Literal["+x", "-x", "+y", "-y", "+z", "-z"]:
    """In your puzzle dataset's format, what was the camera look vector?"""

    raise NotImplementedError("This is your homework.")


def quiz_question_3() -> Literal["+x", "-x", "+y", "-y", "+z", "-z"]:
    """In your puzzle dataset's format, what was the camera up vector?"""

    raise NotImplementedError("This is your homework.")


def quiz_question_4() -> Literal["+x", "-x", "+y", "-y", "+z", "-z"]:
    """In your puzzle dataset's format, what was the camera right vector?"""

    raise NotImplementedError("This is your homework.")


def explanation_of_problem_solving_process() -> str:
    """Please return a string (a few sentences) to describe how you solved the puzzle.
    We'll only grade you on whether you provide a descriptive answer, not on how you
    solved the puzzle (brute force, deduction, etc.).
    """

    raise NotImplementedError("This is your homework.")
