from typing import Dict, Tuple, List

from torchmetrics import detection

import torch
from torch.utils.data import DataLoader

from src.model.detr import DETR
from src.model.loss import DETRLoss
from src.data.transforms import boxes_to_img_scale


class DETREvaluator:
    def __init__(
            self, 
            detr: DETR,
            criterion: DETRLoss,
            device: torch.device,
            score_thresh: float=0.01
    
    ) -> None:
        self.detr = detr
        self.criterion = criterion
        self.device = device
        self.score_thresh = score_thresh

        self.detr.to(self.device)
        self.criterion.to(self.device)
    
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.detr.eval()

        metric = detection.MeanAveragePrecision(box_format="cxcywh", iou_type="bbox")
        
        total_loss = 0.0
        total_loss_ce = 0.0
        total_loss_bbox = 0.0

        for n, (imgs, targets) in enumerate(dataloader):
            imgs = imgs.to(self.device)

            # Predict stuff 
            logits, boxes_pred = self.detr(imgs)

            # Compute matching loss
            loss, loss_bbox, loss_ce = self.criterion(logits, boxes_pred, targets)
            
            # Update losses
            total_loss += (loss.item() - total_loss) / (n + 1)
            total_loss_ce += (loss_ce.item() - total_loss_ce) / (n + 1)
            total_loss_bbox += (loss_bbox.item() - total_loss_bbox) / (n + 1)

            # Update class and box metrics
            preds_lst, targets_lst = self._prepare_map_inputs(imgs, logits, boxes_pred, targets)
            metric.update(preds_lst, targets_lst)

        result = metric.compute()
        
        stats = {
            "loss": total_loss,
            "loss_ce": total_loss_ce,
            "loss_bbox": total_loss_bbox,
            "map": float(result["map"].item()),
            "map50": float(result["map_50"].item()),
        }

        return stats

    def _prepare_map_inputs(
            self, 
            imgs: torch.Tensor,                                 # [B, 3, W_0, H_0]
            logits: torch.Tensor,                               # [B, num_queries, n_classes]
            boxes_pred: torch.Tensor,                           # [B, num_queries, 4]
            targets: List[Dict[str, torch.Tensor]]
        ) -> Tuple[List, List]:
        self.detr.eval() 

        probs = torch.softmax(logits, dim=-1)                   # [B, num_queries, n_classes]
        probs_foreground = probs[:, :, :-1]                     # [B, num_queries, n_classes - 1] 
        scores_all, labels_pred_all = torch.max(
            probs_foreground, dim=-1
        )                                                       # [B, num_queries], [B, num_queries]

        preds_lst, targets_lst = [], []
        for i, target in enumerate(targets):
            # Cache image shape 
            _, height, width = imgs[i].shape
            
            # Ground truth labels and boxes 
            b = target["boxes"].to(self.device)                 # [M, 4]
            y = target["labels"].to(self.device)                # [M]

           # Predicted boxes, labels and scores
            b_pred = boxes_pred[i]                              # [num_queries, 4]
            y_pred = labels_pred_all[i]                         # [num_queries]
            y_pred_scores = scores_all[i]                       # [num_queries]

            # Rescale boxes to absolute image scale
            b = boxes_to_img_scale(b, (width, height), False)
            b_pred = boxes_to_img_scale(b_pred, (width, height), False)

            # Only keep strong predictions
            keep = y_pred_scores >= self.score_thresh           # [num_queries]

            preds_lst.append({
                "boxes": b_pred[keep],                          # [n_keep, 4]
                "labels": y_pred[keep],                         # [n_keep]
                "scores": y_pred_scores[keep]                   # [n_keep]
            })

            targets_lst.append({
                "boxes" : b,                                    # [M, 4]
                "labels" : y                                    # [M]
            })

        return preds_lst, targets_lst