from pathlib import Path
from typing import Literal, TypedDict, Optional
import json
from jaxtyping import Float
from torch import Tensor
import torch
from PIL import Image
import torchvision.transforms as T
from torch_sparse import cat

class PuzzleDataset(TypedDict):
    extrinsics: Float[Tensor, "batch 4 4"]
    intrinsics: Float[Tensor, "batch 3 3"]
    images: Float[Tensor, "batch height width"]
    piece_id: list[str] 



def load_dataset(path: Path) -> PuzzleDataset:
    # 1. 加载所有拼图块图像
    # images_dir = path / "images"
    # with open(images_dir, 'r') as f:

    # 2. 加载内参和外参
    meta_data = path / "metadata.json"
    with open(meta_data, 'r') as f:
        meta = json.load(f)  # 格式: {"piece1": ["piece2", "piece3"], ...}
    
    piece_ids = meta.get("piece_id", [f"piece_{i}" for i in range(len(meta["intrinsics"]))])
    
    extrinsics_dict = torch.tensor(meta["extrinsics"], dtype=torch.float32)
    print(extrinsics_dict.shape)
    # print(extrinsics_dict.values().layout)
    # l = list(extrinsics_dict.values())
    # torch.stack(l, dim=0)

    # sparse_list = [sanitize_sparse(t) for t in extrinsics_dict.values()]
    # extrinsics_tensor = cat(sparse_list, dim=0) 
    # extrinsics_tensor = torch.sparse.sum(torch.stack(list(extrinsics_dict.values())), dim=0)
    # extrinsics_tensor = torch.stack([t.to_dense() if t.is_sparse else t 
    #                         for t in extrinsics_dict.values()], dim=0) 

    intrinsics_dict = torch.tensor(meta["intrinsics"], dtype=torch.float32)
    # intrisics_tensor = torch.stack([t.to_dense() if t.is_sparse else t 
                            # for t in intrinsics_dict.values()], dim=0) 

    data : PuzzleDataset = {
        "intrinsics" : intrinsics_dict,
        "extrinsics" : extrinsics_dict,
        "piece_id"   : piece_ids
    }
    
    return data


def convert_dataset(dataset: PuzzleDataset) -> PuzzleDataset:
    """Convert the dataset into OpenCV-style camera-to-world format. As a reminder, this
    format has the following specification:

    - The camera look vector is +Z.
    - The camera up vector is -Y.
    - The camera right vector is +X.
    - The extrinsics are in camera-to-world format, meaning that they transform points
      in camera space to points in world space.

    """
    
    convert_matrix = torch.tensor([
        [1,  0,  0, 0],  # X保持不变
        [0, -1,  0, 0],  # Y反向（Y-up → -Y-up）
        [0,  0, -1, 0],  # Z反向（Z-forward → Z-backward）
        [0,  0,  0, 1]   # 齐次坐标
    ], dtype=torch.float32)
    
    
    converted_extrinsics = torch.einsum('bij,jk->bik', dataset["extrinsics"], convert_matrix)
    print(converted_extrinsics)
   
    return PuzzleDataset(
        extrinsics=converted_extrinsics,
        intrinsics=dataset["intrinsics"]
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
