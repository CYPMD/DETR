from typing import Tuple, Dict, List

import numpy as np

from scipy.optimize import linear_sum_assignment

import torch
import torch.nn as nn

import torchvision


class GIoULoss(nn.Module):
    def __init__(self, reduction: str="sum", eps: float=1e-6) -> None:
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, b: torch.Tensor, b_pred: torch.Tensor) -> torch.Tensor:
        area_b = self._get_area(b)                                      # [B, N]
        area_b_pred = self._get_area(b_pred)                            # [B, N]
        area_inter = self._get_area_intersection(b, b_pred)             # [B, N]
        area_union = area_b + area_b_pred - area_inter                  # [B, N]

        largest_bbox = self._get_largest_bbox(b, b_pred)                # [B, N, 4]
        area_largest = self._get_area(largest_bbox)                     # [B, N]

        iou = area_inter / (area_union + self.eps)
        penalty = ((area_largest-area_union) / (area_largest+self.eps)) # [B, N]

        loss = 1 - iou + penalty                                        # [B, N]

        if self.reduction == "sum":
            return loss.sum()                                           # [1]
        if self.reduction == "mean":
            return loss.mean()                                          # [1]
        return loss

    def _get_area(self, b: torch.Tensor) -> torch.Tensor:
        """Returns the area of a bounding box."""
        return b[:, :, 2].clamp(min=0) * b[:, :, 3].clamp(min=0)
 
    def _get_area_intersection(self, b: torch.Tensor, b_pred: torch.Tensor) -> torch.Tensor:
        """Returns the intersection area between b and b_pred.""" 
        b_cx, b_pred_cx = b[:, :, 0], b_pred[:, :, 0]
        b_cy, b_pred_cy = b[:, :, 1], b_pred[:, :, 1]
        b_w, b_pred_w = b[:, :, 2], b_pred[:, :, 2]
        b_h, b_pred_h = b[:, :, 3], b_pred[:, :, 3]

        b_x1, b_pred_x1 = b_cx - b_w / 2, b_pred_cx - b_pred_w / 2
        b_y1, b_pred_y1 = b_cy - b_h / 2, b_pred_cy - b_pred_h / 2
        b_x2, b_pred_x2 = b_x1 + b_w, b_pred_x1 + b_pred_w 
        b_y2, b_pred_y2 = b_y1 + b_h , b_pred_y1 + b_pred_h

        width = torch.min(b_x2, b_pred_x2) - torch.max(b_x1, b_pred_x1)
        height = torch.min(b_y2, b_pred_y2) - torch.max(b_y1, b_pred_y1)

        width = width.clamp(min=0)
        height = height.clamp(min=0)

        return width * height

    def _get_largest_bbox(self, b: torch.Tensor, b_pred: torch.Tensor) -> torch.Tensor:
        """Returns the largest box containing both b and b_pred.""" 
        b_cx, b_pred_cx = b[:, :, 0], b_pred[:, :, 0]
        b_cy, b_pred_cy = b[:, :, 1], b_pred[:, :, 1]
        b_w, b_pred_w = b[:, :, 2], b_pred[:, :, 2]
        b_h, b_pred_h = b[:, :, 3], b_pred[:, :, 3] 

        b_x1, b_pred_x1 = b_cx - b_w / 2, b_pred_cx - b_pred_w / 2
        b_y1, b_pred_y1 = b_cy - b_h / 2, b_pred_cy - b_pred_h / 2
        b_x2, b_pred_x2 = b_x1 + b_w, b_pred_x1 + b_pred_w 
        b_y2, b_pred_y2 = b_y1 + b_h , b_pred_y1 + b_pred_h

        # Compute largest bounding box containing both
        x1 = torch.min(b_x1, b_pred_x1)
        y1 = torch.min(b_y1, b_pred_y1)
        x2 = torch.max(b_x2, b_pred_x2)
        y2 = torch.max(b_y2, b_pred_y2)
        
        w = (x2 - x1).clamp(min=0)
        h = (y2 - y1).clamp(min=0)
        cx = x1 + w / 2
        cy = y1 + h / 2

        return torch.stack([cx, cy, w, h], dim=-1)


class BBOXLoss(nn.Module):
    def __init__(
            self, 
            lambda_l1: float, 
            lambda_giou: float, 
            reduction_l1: str="sum", 
            reduction_giou: str="sum",
    ) -> None:
        super().__init__()
        self.lambda_l1  = lambda_l1
        self.lambda_giou = lambda_giou
        self.reduction_l1 = reduction_l1
        self.reduction_iou = reduction_giou

        self.criterion_l1 = nn.L1Loss(reduction=reduction_l1)
        self.criterion_giou = GIoULoss(reduction=reduction_giou)

    def forward(self, b: torch.Tensor, b_pred: torch.Tensor) -> torch.Tensor:
        loss_l1 = self.lambda_l1 * self.criterion_l1(b, b_pred)
        loss_iou = self.lambda_giou * self.criterion_giou(b, b_pred)
        return loss_l1 + loss_iou


class HungarianAlgorithm(object):
    def __init__(self, lambda_l1: float, lambda_giou: float) -> None:
        self.lambda_l1 = lambda_l1
        self.lambda_giou = lambda_giou

    def __call__(
            self, 
            logits: torch.Tensor, 
            labels: torch.Tensor, 
            boxes_pred: torch.Tensor, 
            boxes: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self.match(logits, labels, boxes_pred, boxes)

    @torch.no_grad()
    def match(
            self, 
            logits: torch.Tensor, 
            labels: torch.Tensor, 
            boxes_pred: torch.Tensor, 
            boxes: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Classification cost
        probs = torch.softmax(logits, dim=-1)           # [num_queries, n_classes]    
        cost_class = -probs[:, labels]                  # [num_queries, M]

        # Cost L1 dist
        cost_l1 = torch.cdist(
            boxes_pred, boxes, p=1
        )                                               # [num_queries, M]

        boxes_xyxy = torchvision.ops.box_convert(
            boxes, "cxcywh", "xyxy"
        )                                               # [M, 4]
        boxes_pred_xyxy = torchvision.ops.box_convert(
            boxes_pred, "cxcywh", "xyxy"
        )                  

        # Cost GIou
        cost_giou = -torchvision.ops.generalized_box_iou(
            boxes_pred_xyxy, boxes_xyxy
        )                                               # [num_queries, M]

        # Compute total cost (has shape: [num_queries, M])
        total_cost = (
            cost_class + self.lambda_l1 * cost_l1 + self.lambda_giou * cost_giou
        )                                               # [num_queries, M]

        # Hungarian algorithm 
        row_indices, col_indices = linear_sum_assignment(
            total_cost.detach().cpu().numpy()
        ) 
        return row_indices, col_indices


class DETRLoss(nn.Module):
    def __init__(
            self, 
            lambda_l1: float, 
            lambda_giou: float, 
            num_classes: int,
            empty_set_weight: float=0.1
    ) -> None:
        super().__init__()
        self.lambda_l1  = lambda_l1
        self.lambda_giou = lambda_giou
        self.num_classes = num_classes
        self.empty_set_id = num_classes - 1

        weights = torch.ones((self.num_classes,), dtype=torch.float32)
        weights[self.empty_set_id] = empty_set_weight
        self.register_buffer("class_weights", weights)

        self.hungarian = HungarianAlgorithm(lambda_l1, lambda_giou) 
        self.criterion_classes = nn.CrossEntropyLoss(reduction="sum", weight=self.class_weights)
        self.criterion_bbox = BBOXLoss(lambda_l1, lambda_giou, "sum", "sum")

    def forward(
            self, 
            logits: torch.Tensor,
            boxes_pred: torch.Tensor,
            targets: List[Dict],
    ) -> torch.Tensor:
        # logits have shape:                                # [B, num_queries, n_classes]
        # boxes_pred have shape:                            # [B, num_queries, 4]
        
        # Just some shortcut constants
        num_queries = logits.size(1)
        empty_set_id = self.empty_set_id
        device = logits.device

        # Compute bounding-box and cross-entropy loss 
        loss_bbox = torch.tensor(0.0, device=logits.device)
        loss_ce = torch.tensor(0.0, device=logits.device)

        N_targets = 0
        for i, target in enumerate(targets):
            # Let M be the number of ground truth labels in the image
            boxes = target["boxes"].to(logits.device)       # [M, 4]
            labels = target["labels"].to(logits.device)     # [M]

            b_pred = boxes_pred[i]                          # [num_queries, 4]
            y_pred = logits[i, :, :]                        # [num_queries, n_classes]

            # Shape of [num_queries] with empty_set_id as elements
            target_classes = torch.full((num_queries,), empty_set_id, dtype=torch.long, device=device)

            # No targets in this image
            if boxes.size(0) == 0:
                loss_ce = loss_ce + self.criterion_classes(y_pred, target_classes)
                continue

            # Hungarian algorithm, NOTE: M == num_matches
            row_indices, col_indices = self.hungarian(y_pred, labels, b_pred, boxes)

            # Select matching labels
            target_classes[row_indices] = labels[col_indices]

            # Select matching boxes
            matched_target_boxes = boxes[col_indices]        # [num_matches, 4]
            matched_pred_boxes = b_pred[row_indices]         # [num_matches, 4]

            # Compute bbox loss
            loss_bbox = loss_bbox + self.criterion_bbox(
                matched_target_boxes.unsqueeze(0),           # [1, num_matches_4]
                matched_pred_boxes.unsqueeze(0)              # [1, num_matches, 4]
            )

            # Compute cross-entropy loss
            loss_ce = loss_ce + self.criterion_classes(
                y_pred,                                      # [num_queries, n_classes]
                target_classes                               # [num_queries]
            )

            N_targets += boxes.size(0)

        N_targets = max(N_targets, 1)
        loss_bbox /= N_targets
        loss_ce /= N_targets

        return loss_ce + loss_bbox, loss_bbox, loss_ce

    def match(
            self, 
            logits: torch.Tensor, 
            labels: torch.Tensor, 
            boxes_pred: torch.Tensor, 
            boxes: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Computes bi-partite matching using hungarian algorithm."""
        return self.hungarian(logits, labels, boxes_pred, boxes)