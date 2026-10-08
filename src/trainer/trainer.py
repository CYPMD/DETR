from typing import Dict

import os
import json

import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model.detr import DETR
from src.model.loss import DETRLoss
from src.trainer.evaluator import DETREvaluator


class DETRTrainer:
    def __init__(
            self, 
            detr: DETR, 
            n_epochs: int,
            learning_rate: float,
            learning_rate_backbone: float,
            weight_decay: float,
            weight_decay_backbone: float,
            lr_schedule: float,
            lambda_l1: float,
            lambda_giou: float,
            device: str,
            clip_grad_norm: float|None=None,
            score_tresh: float=0.01,
            decay_lr_every: int=50,
            eval_every: int=5,
            save_every: int=5,
            verbose: bool=True,
            learning_rate_encoder: float|None=None,
            weight_decay_encoder: float|None=None,
            output_dir: str|None=None,
            run_config: Dict|None=None,
    ) -> None:
        # Detect device 
        self.device = torch.device(device)

        # Detection transformer model
        self.detr = detr
        self.detr.to(self.device)

        encoder_parameters = list(self.detr.pretrained_parameters())
        encoder_ids = {id(p) for p in encoder_parameters}
        encoder_lr = learning_rate_backbone
        encoder_wd = weight_decay_backbone
        if self.detr.architecture == "vit_b_32":
            encoder_lr = learning_rate_encoder if learning_rate_encoder is not None else 1e-5
            encoder_wd = weight_decay_encoder if weight_decay_encoder is not None else weight_decay_backbone

        # The pretrained feature encoder gets its own learning rate. Decoder,
        # new projections, queries and heads use the main learning rate.
        groups = [
            {
                "params": [p for p in encoder_parameters if p.requires_grad],
                "lr": encoder_lr,
                "weight_decay": encoder_wd,
            },
            {
                "params": [
                    p for p in self.detr.parameters()
                    if p.requires_grad and id(p) not in encoder_ids
                ],
                "lr": learning_rate,
                "weight_decay": weight_decay,
            },
        ]
        self.optimizer = torch.optim.AdamW([group for group in groups if group["params"]])

        # DETR loss
        self.criterion = DETRLoss(lambda_l1, lambda_giou, detr.n_classes)
        self.criterion.to(self.device)

        # Lr scheduler
        lmbda = lambda epoch: lr_schedule
        self.scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
            self.optimizer,
            lr_lambda=lmbda 
        )
        
        # Hyperparameters
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.clip_grad_norm = clip_grad_norm
        self.score_tresh = score_tresh

        # Logging and evaluation settings
        self.decay_lr_every = decay_lr_every
        self.eval_every = eval_every
        self.save_every = save_every
        self.verbose = verbose
        self.has_val = False
        self.output_dir = output_dir or os.path.join("runs", detr.architecture)
        self.run_config = dict(run_config or {})
        self.epoch = 0

        self.evaluator = DETREvaluator(
            self.detr, self.criterion, self.device, self.score_tresh
        )

        self.n_params = sum(
            p.numel() for p in self.detr.parameters() if p.requires_grad
        )

        # Keep track of performance 
        self.stats_train = {
            "epoch": [], "loss": [], "loss_ce": [], "loss_bbox": [], "map": [], "map50": []
        }
        self.stats_val = {
            "epoch": [], "loss": [], "loss_ce": [], "loss_bbox": [], "map": [], "map50": []
        }

    def train(self, train_loader: DataLoader, val_loader: DataLoader|None=None) -> None:
        if val_loader is not None: self.has_val = True 
        for epoch in range(1, self.n_epochs + 1): 
            self.epoch = epoch
            
            self.train_one_epoch(train_loader, epoch=epoch)

            if epoch % self.eval_every == 0:
                self.evaluate(train_loader, epoch, True)
                if self.has_val: self.evaluate(val_loader, epoch, False)

            if epoch % self.save_every == 0:
                self._checkpoint()

            if epoch % self.decay_lr_every == 0:
                self.scheduler.step()

            if self.verbose and epoch % self.eval_every == 0:
                self._print_stats(epoch)
            
        self.evaluate(train_loader, epoch, True)
        if self.has_val: self.evaluate(val_loader, epoch, False)
        self._checkpoint()

    def train_one_epoch(self, dataloader: DataLoader, epoch: int|None=None) -> None:
        self.detr.train()
        n_aux = len(self.detr.transformer.decoder.layers) - 1 
        description = "Training" if epoch is None else f"Epoch {epoch}/{self.n_epochs} | train"
        n_batches = len(dataloader)

        with tqdm(
            dataloader,
            desc=description,
            unit="batch",
            dynamic_ncols=True,
            mininterval=1.0,
            disable=not self.verbose,
        ) as progress:
            for step, (imgs, targets) in enumerate(progress, start=1):
                imgs = imgs.to(self.device)               # [B, 3, W_0, H_0]

                # Predict stuff
                logits, boxes_pred = self.detr(imgs)      # [N, num_queries, n_classes], [N, num_queries, 4]

                # Compute loss
                loss, _, _ = self.criterion(logits, boxes_pred, targets)

                # Compute auxiliary loss
                loss_aux_total = 0.0
                for n in range(n_aux):
                    x_dec = self.detr.transformer.decoder.outs_[n]
                    logits_aux = self.detr.proj_class(x_dec)
                    boxes_aux = self.detr.proj_bbox(x_dec)
                    loss_aux, _, _ = self.criterion(logits_aux, boxes_aux, targets)
                    loss_aux_total += loss_aux

                # Compute final loss
                loss = loss + loss_aux_total

                # Backpropagation
                self.optimizer.zero_grad()

                loss.backward()

                if self.clip_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.detr.parameters(), self.clip_grad_norm)

                self.optimizer.step()

                # Sample the displayed loss periodically to limit CUDA synchronization.
                if self.verbose and (step == 1 or step % 20 == 0 or step == n_batches):
                    progress.set_postfix(
                        loss=f"{loss.detach().item():.4f}",
                        refresh=False,
                    )

    def evaluate(self, dataloader: DataLoader, epoch: int, train_set: bool = False) -> None:
        split = "train" if train_set else "val"
        history = self.evaluator.evaluate(
            dataloader,
            description=f"Epoch {epoch}/{self.n_epochs} | eval {split}",
            verbose=self.verbose,
        )
        self._append_stats(history, epoch, train_set)

    def _append_stats(self, history: Dict[str, list], epoch: int, train_set: bool = False) -> None:
        stats = self.stats_train if train_set else self.stats_val
        stats["epoch"].append(epoch)
        stats["loss"].append(history["loss"])
        stats["loss_ce"].append(history["loss_ce"])
        stats["loss_bbox"].append(history["loss_bbox"])
        stats["map"].append(history["map"])
        stats["map50"].append(history["map50"])
 
    def _checkpoint(self) -> None:
        if self.verbose:
            tqdm.write("Saving checkpoint and metric reports...")

        save_dir = self.output_dir
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, "checkpoint.pt")
        torch.save({
            "format_version": 1,
            "epoch": self.epoch,
            "model_config": self.detr.model_config,
            "run_config": self.run_config,
            "model_state_dict": self.detr.state_dict(),
        }, file_path)
        # Retain a raw state dict for existing inference notebooks.
        torch.save(self.detr.state_dict(), os.path.join(save_dir, "model_weights.pt"))
        with open(os.path.join(save_dir, "model_config.json"), "w") as handle:
            json.dump(self.detr.model_config, handle, indent=2)
        
        file_name = os.path.join(save_dir, "train_metrics.csv")
        pd.DataFrame.from_dict(self.stats_train).to_csv(file_name, index=False)

        if self.has_val: 
            file_name = os.path.join(save_dir, "val_metrics.csv")
            pd.DataFrame.from_dict(self.stats_val).to_csv(file_name, index=False)

        if self.verbose:
            tqdm.write(f"Checkpoint saved: {file_path}")

    def _print_stats(self, epoch: int) -> None:
        if self.has_val: 
            print(
                f"Epoch: {epoch}\t"
                f"Loss (train): {self.stats_train['loss'][-1]:.4f}\t"
                f"Loss (val): {self.stats_val['loss'][-1]:.4f}\t"
                f"mAP (train): {self.stats_train['map'][-1]:.4f}\t"
                f"mAP (val): {self.stats_val['map'][-1]:.4f}\t"
                f"mAP50 (train): {self.stats_train['map50'][-1]:.4f}\t"
                f"mAP50 (val): {self.stats_val['map50'][-1]:.4f}\t"
            )
        else:
            print(
                f"Epoch: {epoch}\t"
                f"Loss (train): {self.stats_train['loss'][-1]:.4f}\t"
                f"mAP (train): {self.stats_train['map'][-1]:.4f}\t"
                f"mAP50 (train): {self.stats_train['map50'][-1]:.4f}\t"
            )
