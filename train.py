from argparse import Namespace, ArgumentParser
from time import perf_counter
from typing import Any, List, Tuple, Dict
from pathlib import Path
import random

import numpy as np

import torch
from torch.utils.data import DataLoader


from src import DETR, DETRTrainer, load_weights, load_decoder_weights
from src import VOC, TrainTransform, ValidationTransform


def parse_args(argv=None) -> Namespace:
    parser = ArgumentParser(description="DETR training")

    parser.add_argument("--model", choices=("resnet50", "vit_b_32"), default="resnet50")
    parser.add_argument("--no_pretrained", action="store_true", help="Random feature-encoder initialization; also useful offline")
    parser.add_argument("--freeze_encoder", action="store_true", help="Freeze ViT, or the ResNet backbone in baseline mode")
    initializers = parser.add_mutually_exclusive_group()
    initializers.add_argument("--weights", type=str, help="Strictly load complete detection weights; starts a new optimizer")
    initializers.add_argument("--init_decoder_from", type=str, help="Warm-start matching queries, decoder and detection heads")
    parser.add_argument("--output_dir", type=str, default=None, help="Default: runs/<model>; use a unique directory per experiment")

    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--encoder_layers", type=int, default=6)
    parser.add_argument("--encoder_heads", type=int, default=8)
    parser.add_argument("--decoder_layers", type=int, default=6)
    parser.add_argument("--decoder_heads", type=int, default=8)
    parser.add_argument("--n_classes", type=int, default=21, help="Total outputs INCLUDING no-object: VOC=21")
    parser.add_argument("--num_queries", type=int, default=100)
    parser.add_argument("--ff_dim", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_tokens", type=int, default=400, help="ResNet positional capacity; ignored by ViT")

    parser.add_argument("--n_epochs", type=int, default=150)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--learning_rate_backbone", type=float, default=5e-5)
    parser.add_argument("--learning_rate_encoder", type=float, default=1e-5, help="ViT learning rate; baseline uses learning_rate_backbone")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--weight_decay_backbone", type=float, default=1e-4)
    parser.add_argument("--weight_decay_encoder", type=float, default=1e-4, help="ViT weight decay")
    parser.add_argument("--batch_size", type=int, default=4)            # Limited by my compute
    parser.add_argument("--lambda_l1", type=float, default=5.0)
    parser.add_argument("--lambda_giou", type=float, default=2.0)
    parser.add_argument("--lr_schedule", type=float, default=0.1)
    parser.add_argument("--clip_grad_norm", type=float, default=0.1)
    parser.add_argument("--score_thresh", type=float, default=0.01)
    parser.add_argument("--decay_lr_every", type=int, default=100)
    
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--root", type=str, default="./voc")
    parser.add_argument(
        "--download",
        action="store_true",
        default=False,
        help="Download and extract VOC archives. Omit when the dataset is already extracted.",
    )
    parser.add_argument("--img_size", type=int, nargs=2, default=(640, 640), metavar=("HEIGHT", "WIDTH"))

    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--verbose", default=True)  # Preserve the existing argument.
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    args.img_size = tuple(args.img_size)
    args.verbose = str(args.verbose).lower() not in {"false", "0", "no"} and not args.quiet
    if any(size <= 0 for size in args.img_size):
        parser.error("--img_size dimensions must be positive")
    if args.model == "vit_b_32" and any(size % 32 for size in args.img_size):
        parser.error("ViT-B/32 requires --img_size dimensions divisible by 32")
    if args.model == "resnet50":
        tokens = ((args.img_size[0] + 31) // 32) * ((args.img_size[1] + 31) // 32)
        if args.max_tokens < tokens:
            parser.error(f"This image size needs --max_tokens >= {tokens}")
    for name in ("n_epochs", "batch_size", "save_every", "eval_every", "decay_lr_every"):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be positive")
    if args.output_dir is None:
        args.output_dir = str(Path("runs") / args.model)
    if args.freeze_encoder and args.no_pretrained and not args.weights:
        parser.error("Do not freeze a randomly initialized encoder; use pretrained weights or --weights")
    return args


def collate_fn(batch: Any) -> Tuple[torch.Tensor, List[Dict]]:
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(targets)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    args = parse_args() 
    startup_started = perf_counter()

    def log_startup(message: str) -> None:
        if args.verbose:
            elapsed = perf_counter() - startup_started
            print(f"[startup +{elapsed:.1f}s] {message}", flush=True)

    log_startup("Setting random seeds...")
    set_seeds(args.seed) 
    
    log_startup(f"Building {args.model} DETR (pretrained weights may download if not cached)...")
    if args.model == "vit_b_32":
        log_startup("ViT uses 32x32 patches, 12 encoder layers, 12 heads and width 768; d_model controls the decoder.")
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
        args.max_tokens,
        architecture=args.model,
        pretrained=not args.no_pretrained and args.weights is None,
        freeze_encoder=args.freeze_encoder,
    )
    if args.weights:
        load_weights(detr, args.weights)
        log_startup("Loaded all detection weights; starting a new optimizer.")
    elif args.init_decoder_from:
        load_decoder_weights(detr, args.init_decoder_from)
        log_startup("Loaded queries, decoder and detection heads from the supplied checkpoint.")
    log_startup("Model ready.")

    pin_mem = True if args.num_workers > 0 else False
    
    train_transform = TrainTransform(args.img_size) 
    val_transform = ValidationTransform(args.img_size)

    log_startup(f"Loading training data from {args.root} (download={args.download})...")
    train_set = VOC(
        root=args.root,
        image_set="train",
        transform=train_transform,
        download=args.download,
    )
    log_startup(f"Training dataset ready: {len(train_set):,} images.")
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
        collate_fn=collate_fn,
    )

    log_startup(f"Loading evaluation data from {args.root} (download={args.download})...")
    val_set = VOC(
        root=args.root,
        image_set="val",
        transform=val_transform,
        download=args.download,
    )
    log_startup(f"Evaluation dataset ready: {len(val_set):,} images.")
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
        collate_fn=collate_fn,
    )

    log_startup(f"Initializing trainer on {args.device}...")
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
        learning_rate_encoder=args.learning_rate_encoder,
        weight_decay_encoder=args.weight_decay_encoder,
        output_dir=args.output_dir,
        run_config=vars(args),
    ) 

    log_startup(
        f"Starting training: {args.n_epochs} epochs, "
        f"{len(train_loader):,} batches per epoch, {args.num_workers} data workers."
    )
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
