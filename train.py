from argparse import Namespace, ArgumentParser
from typing import Any, List, Tuple, Dict

import numpy as np

import torch
from torch.utils.data import DataLoader


from src import DETR, DETRTrainer
from src import VOC, TrainTransform, ValidationTransform


def parse_args() -> Namespace:
    parser = ArgumentParser(description="DETR training")

    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--encoder_layers", type=int, default=4)
    parser.add_argument("--encoder_heads", type=int, default=8)
    parser.add_argument("--decoder_layers", type=int, default=4)
    parser.add_argument("--decoder_heads", type=int, default=8)
    parser.add_argument("--n_classes", type=int, default=21)            # n_classes + 1
    parser.add_argument("--num_queries", type=int, default=25)
    parser.add_argument("--ff_dim", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_tokens", type=float, default=400)        # (img_size[0] / 32) * (img_size[1] / 32)

    parser.add_argument("--n_epochs", type=int, default=300)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--learning_rate_backbone", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--weight_decay_backbone", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lambda_l1", type=float, default=5.0)
    parser.add_argument("--lambda_giou", type=float, default=2.0)
    parser.add_argument("--lr_schedule", type=float, default=0.5)
    parser.add_argument("--clip_grad_norm", type=float, default=0.5)
    parser.add_argument("--score_thresh", type=float, default=0.01)
    parser.add_argument("--decay_lr_every", type=int, default=100)
    
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--root", type=str, default="./voc")
    parser.add_argument("--download", type=bool, default=True)
    parser.add_argument("--img_size", type=tuple, default=(640, 640))

    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--verbose", default=True)
    return parser.parse_args()


def collate_fn(batch: Any) -> Tuple[torch.Tensor, List[Dict]]:
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(targets)


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    args = parse_args() 
    set_seeds(args.seed) 
    
    detr = DETR(
        args.d_model, 
        args.encoder_layers,
        args.encoder_heads, 
        args.decoder_layers, 
        args.decoder_heads, 
        args.n_classes, 
        args.num_queries,
        args.ff_dim,
        args.dropout,
        args.max_tokens
    )

    pin_mem = True if args.num_workers > 0 else False
    
    train_transform = TrainTransform(args.img_size) 
    val_transform = ValidationTransform(args.img_size)

    train_set = VOC(root=args.root, image_set="train", transform=train_transform)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
        collate_fn=collate_fn,
    )

    val_set = VOC(root=args.root, image_set="val", transform=val_transform)
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
        collate_fn=collate_fn,
    )

    trainer = DETRTrainer(
        detr=detr,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        learning_rate_backbone=args.learning_rate_backbone,
        weight_decay=args.weight_decay,
        weight_decay_backbone=args.weight_decay_backbone,
        lr_schedule=args.lr_schedule,
        lambda_l1=args.lambda_l1,
        lambda_giou=args.lambda_giou,
        device=args.device,
        clip_grad_norm=args.clip_grad_norm,
        score_tresh=args.score_thresh,
        decay_lr_every=args.decay_lr_every,
        eval_every=args.eval_every,
        save_every=args.save_every,
        verbose=args.verbose,
    ) 

    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()