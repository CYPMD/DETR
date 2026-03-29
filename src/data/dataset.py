import copy

from typing import List, Dict, Tuple, Callable

from PIL import Image

import torch
from torch.utils.data import Dataset

import torchvision
from torchvision.datasets import VOCDetection

from src.data.constants import (
    CLASSES,
    CLASS_TO_IDX,
    CLASS_IDX_TO_NAME, 
)


class VOC(Dataset):
    def __init__(
            self, 
            root: str,
            transform: Callable | None=None, 
            image_set: str="train",
            download: bool=True
    ) -> None:
        super().__init__()

        self.voc = VOCDetection(root, year="2012", image_set=image_set, download=download)

        self.root = root
        self.download = download
        self.transform = transform

        self.classes = None
        self.class_to_idx = None
        self.class_idx_to_name = None
        self.__post_init__()
    
    def __post_init__(self) -> None:
        classes = [x.lower() for x in CLASSES] 
        
        assert len(classes) == 20, (
            f"VOC should have 20 classes, but there are only: {len(classes)}"
        )

        class_to_idx = {class_name : class_id
                             for class_id, class_name in enumerate(classes)
        }

        class_idx_to_name = {class_id : class_name
                                  for class_name, class_id in class_to_idx.items()
        }

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.class_idx_to_name = class_idx_to_name

    def __len__(self) -> int:
        return len(self.voc)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, ...]:
        if isinstance(index, torch.Tensor):
            index = index.item()
        
        # Load image and detection targets
        img, target_ann = self.voc[index]

        # Cache objects in image
        ann = target_ann["annotation"]
        objects = ann["object"]
        
        # Boxformat in cxcywh 
        boxes, labels = self._extract_boxes_and_labels(objects)

        target = {"boxes" : boxes, "labels" : labels} 
        
        if self.transform is not None:
            img, target = self.transform(img, target)

        if isinstance(img, Image.Image):
            width_new, height_new = img.size
        elif isinstance(img, torch.Tensor):
            _, height_new, width_new = img.shape
        else:
            raise ValueError(f"Unknown image type: {type(img)}")

        # Normalize bounding boxes to [0, 1]^4
        target["boxes"] = torch.as_tensor(target["boxes"], dtype=torch.float32) 
        target["boxes"][:, 0] /= width_new
        target["boxes"][:, 1] /= height_new
        target["boxes"][:, 2] /= width_new
        target["boxes"][:, 3] /= height_new

        return img, target
    
    def _extract_boxes_and_labels(
            self, 
            objects: List[Dict]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extracts labels and bounding boxes from VOC annoations, converts boxes to `cxcywh`.""" 
        if isinstance(objects, dict):
            objects = [objects] 
        
        boxes, labels = [], []
        for obj in objects: 
            if bool(int(obj["difficult"])):
                continue

            bbox = obj["bndbox"]
            xmin, xmax = float(bbox["xmin"]), float(bbox["xmax"])
            ymin, ymax = float(bbox["ymin"]), float(bbox["ymax"])
            
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_to_idx[obj["name"]])

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        boxes = torchvision.ops.box_convert(boxes, "xyxy", "cxcywh")

        return boxes, labels