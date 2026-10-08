"""TorchVision ViT-B/32 adapted to return spatial tokens for detection."""

import math
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ViT_B_32_Weights, vit_b_32


class ViTPatchEncoder(nn.Module):
    """Encode RGB patches without a CNN feature backbone or classification head.

    Images must be normalized with the existing ImageNet mean/std and have
    positive height/width divisible by 32. Rectangular images are supported.
    Keep CLS inside the pretrained encoder, then remove it from decoder memory.
    """

    patch_size = 32

    def __init__(self, pretrained: bool = True, frozen: bool = False) -> None:
        super().__init__()
        weights = ViT_B_32_Weights.IMAGENET1K_V1 if pretrained else None
        self.vit = vit_b_32(weights=weights)
        self.vit.heads = nn.Identity()
        self.hidden_dim = self.vit.hidden_dim
        self.frozen = frozen
        if frozen:
            self.vit.requires_grad_(False)
            self.vit.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self.vit.eval()
        return self

    def spatial_positions(self, height: int, width: int) -> torch.Tensor:
        """Interpolate the learned 7x7 patch grid; preserve the CLS position.

        Keep the original Parameter so gradients and saved weights retain their
        normal form. Recompute for each input grid, without changing checkpoints.
        """
        position = self.vit.encoder.pos_embedding
        cls_position, patch_positions = position[:, :1], position[:, 1:]
        side = math.isqrt(patch_positions.shape[1])
        if side * side != patch_positions.shape[1]:
            raise ValueError("Expected a square pretrained ViT positional grid")
        if (height, width) == (side, side):
            return position

        grid = patch_positions.reshape(1, side, side, self.hidden_dim).permute(0, 3, 1, 2)
        # Float interpolation also works when the model has been cast to fp16.
        grid = F.interpolate(
            grid.float(), size=(height, width), mode="bicubic", align_corners=True,
        ).to(position.dtype)
        grid = grid.flatten(2).transpose(1, 2)
        return torch.cat((cls_position, grid), dim=1)

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("ViT expects normalized RGB images with shape [B, 3, H, W]")
        height, width = images.shape[-2:]
        if height < 32 or width < 32 or height % 32 or width % 32:
            raise ValueError(
                f"ViT-B/32 requires H and W to be positive multiples of 32; got {height}x{width}. "
                "Set --img_size HEIGHT WIDTH, for example --img_size 640 640."
            )

        # Conv2d(kernel_size=stride=32) is exactly a shared linear projection
        # of non-overlapping flattened patches, not a CNN feature backbone.
        patches = self.vit.conv_proj(images).flatten(2).transpose(1, 2)
        cls = self.vit.class_token.expand(images.shape[0], -1, -1).to(patches.dtype)
        tokens = torch.cat((cls, patches), dim=1)
        positions = self.spatial_positions(height // 32, width // 32).to(tokens.dtype)

        # Equivalent to TorchVision's encoder forward with resized positions.
        # Avoid its classifier forward, which asserts the native 224x224 size.
        tokens = self.vit.encoder.dropout(tokens + positions)
        tokens = self.vit.encoder.layers(tokens)
        tokens = self.vit.encoder.ln(tokens)
        return tokens[:, 1:], positions[:, 1:].expand(images.shape[0], -1, -1)
