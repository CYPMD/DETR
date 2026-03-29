from typing import Tuple, Dict

from PIL import Image

import numpy as np

import torch

import torchvision
from torchvision.transforms import v2

from src.data.constants import (
    IMAGENET_MEAN,
    IMAGENET_STD
)


class TrainTransform(object):
    def __init__(self, size: Tuple[int, int]) -> None:
        self.size = size
        
        self.transforms = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomZoomOut(fill=[123.0, 117.0, 104.0]),
            v2.RandomIoUCrop(),
            v2.RandomPhotometricDistort(),
            v2.Resize(size=size),
            v2.SanitizeBoundingBoxes(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD
            ),
        ])

    def __call__(self, img: Image.Image, target: Dict):
        width, height = img.size
        if isinstance(target["boxes"], torch.Tensor):
            target["boxes"] = torchvision.tv_tensors.BoundingBoxes(
                target["boxes"], format="CXCYWH", canvas_size=(height, width)
            )
        img, target = self.transforms(img, target)
        return img, target


class ValidationTransform(object):
    def __init__(self, size: Tuple[int, int]) -> None:
        self.size = size
        
        self.transforms = v2.Compose([
            v2.Resize(size=size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD
            ),
        ])

    def __call__(self, img: Image.Image, target: Dict):
        width, height = img.size
        if isinstance(target["boxes"], torch.Tensor):
            target["boxes"] = torchvision.tv_tensors.BoundingBoxes(
                target["boxes"], format="CXCYWH", canvas_size=(height, width)
            )
        img, target = self.transforms(img, target)
        return img, target


def denormalize(img: torch.Tensor) -> np.ndarray:
    """Expects RGB tensor of shape [3, H, W], returns normalized numpy array of shape [H, W, 3].
    
    Args:
    -----
    img : RGB tensor of shape [3, H, W]
    """ 
    img = img.cpu().permute(1, 2, 0).numpy()
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    img = img * std + mean
    return np.clip(img, 0, 1)


def cxcywh_to_xywh(box: np.ndarray) -> np.ndarray:
    """Expects single bounding box.
    
    Args:
    -----
    box : Bounding box of shape [4] 
    """ 
    box = box.copy()

    cx, cy = box[0], box[1]
    w, h = box[2], box[3]

    x = cx - w / 2
    y = cy - h / 2

    box[0], box[1] = x, y

    return box


def box_to_img_scale(box: np.ndarray, img_size: Tuple[int, int]) -> np.ndarray:
    """Expects single bouding box.
    
    Args:
    -----
    box : Bounding box of shape [4] 
    img_size : Original image width and height as tuple.
    clone : If boxes should be cloned.
    """ 
    box = box.copy() 

    width, height = img_size

    box[0] *= width
    box[1] *= height
    box[2] *= width
    box[3] *= height

    return box


def boxes_to_img_scale(
        boxes: torch.Tensor, 
        img_size: Tuple[int, int],
        clone: bool=False
) -> np.ndarray:
    """Expects multiple bouding boxes.

    Args:
    -----
    boxes : Bounding boxes of shape [N, 4] 
    img_size : Original image width and height as tuple.
    clone : If boxes should be cloned.
    """ 
    if clone: boxes = boxes.clone() 

    width, height = img_size

    boxes[:, 0] *= width
    boxes[: , 1] *= height
    boxes[:, 2] *= width
    boxes[:, 3] *= height

    return boxes


def box_to_img_xywh_and_scale(
        box: np.ndarray | torch.Tensor, 
        img_size: Tuple[int, int]
) -> np.ndarray:
    """Expects single bouding box. Normalizes box to xywh and converts to image scale.
    
    Args:
    -----
    box : Bounding box of shape [4] (np.ndarray or torch.Tensor)
    img_size : Original image width and height as tuple.
    """ 
    if isinstance(box, torch.Tensor):
        box = box.detach().cpu().numpy() 
    
    box = cxcywh_to_xywh(box)
    box = box_to_img_scale(box, img_size)

    return box


def get_xywh(box: np.ndarray) -> Tuple[int, int, int, int]:
    x, y, w, h = box.tolist()
    x, y, w, h = int(x), int(y), int(w), int(h)
    return x, y, w, h