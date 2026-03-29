from typing import Tuple


import torch
import torchvision


def extract_boxes(
        logits: torch.Tensor, 
        boxes: torch.Tensor, 
        score_thresh: float, 
        nmf_iou_thresh: float|None=None
) -> Tuple[torch.Tensor, ...]:
    logits = logits.squeeze(0).cpu()
    boxes = boxes.squeeze(0).cpu()

    probs = torch.softmax(logits, dim=-1)
    probs_fg = probs[:, :-1]
    scores, labels = probs_fg.max(dim=-1) 

    keep = scores >= score_thresh

    boxes = boxes[keep] 
    scores = scores[keep]
    labels = labels[keep]

    if nmf_iou_thresh is not None and len(boxes) > 0:
        boxes_xyxy = torchvision.ops.box_convert(
            boxes, in_fmt="xywh", out_fmt="xyxy"
        )
        
        keep = torchvision.ops.batched_nms(boxes_xyxy, scores, labels, nmf_iou_thresh)
        # keep = torchvision.ops.nms(boxes_xyxy, scores, nmf_iou_thresh)

        boxes = boxes[keep] 
        scores = scores[keep]
        labels = labels[keep]

    return scores, labels, boxes


def extract_boxes_with_attn(
        logits: torch.Tensor, 
        boxes: torch.Tensor, 
        attn_weights: torch.Tensor,
        score_thresh: float,
        nmf_iou_thresh: float|None=None
) -> Tuple[torch.Tensor, ...]:
    logits = logits.squeeze(0).cpu()
    boxes = boxes.squeeze(0).cpu()

    probs = torch.softmax(logits, dim=-1)
    probs_fg = probs[:, :-1]
    scores, labels = probs_fg.max(dim=-1) 

    keep = scores >= score_thresh

    boxes = boxes[keep] 
    scores = scores[keep]
    labels = labels[keep]
    attn_weights = attn_weights[keep]

    if nmf_iou_thresh is not None and len(boxes) > 0:
        boxes_xyxy = torchvision.ops.box_convert(
            boxes, in_fmt="xywh", out_fmt="xyxy"
        )
        
        keep = torchvision.ops.batched_nms(boxes_xyxy, scores, labels, nmf_iou_thresh)
        # keep = torchvision.ops.nms(boxes_xyxy, scores, nmf_iou_thresh)

        boxes = boxes[keep] 
        scores = scores[keep]
        labels = labels[keep]
        attn_weights = attn_weights[keep]

    return scores, labels, boxes, attn_weights