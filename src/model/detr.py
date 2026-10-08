from typing import Tuple
from itertools import chain

import torch
import torch.nn as nn

from torchvision.models import resnet50, ResNet50_Weights

from src.model.vit import ViTPatchEncoder


class ResNet50Backbone(nn.Module):
    """
    Pretrained ResNet-50 backbone.
    
    Reference:
    ----------
    Deep Residual Learning for Image Recognition, He et al., 2015
    https://arxiv.org/abs/1512.03385
    """ 
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()

        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        resnet = resnet50(weights=weights)

        # [N, 3, H_0, W_0] -> [N, 2048, H, W]
        self.resnet50 = nn.Sequential(
            *list(resnet.children())[:-2]
        )

        self._freeze_batch_norm()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_feat = self.resnet50(x)
        return x_feat

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze_batch_norm()
        return self

    def _freeze_batch_norm(self) -> None:
        # Freeze all BatchNorm layers
        for m in self.resnet50.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False


class SpatialToSequence(nn.Module):
    """Transforms image features into a sequence suitable for transformers.""" 
    def __init__(self, in_channels: int, d_model: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.conv = nn.Conv2d(in_channels, d_model, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape 
        x_seq = self.conv(x).view(B, self.d_model, H * W)
        x_seq = x_seq.permute(0, 2, 1)
        return x_seq


class Queries(nn.Module):
    """Learnable query."""
    def __init__(self, num_queries: int, d_model: int) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, num_queries, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, _ = x.shape
        q = self.queries.expand(B, -1, -1)
        return q


class PositionalEncoding(nn.Module):
    """Positional encoding for rows and columns.""" 
    def __init__(self, max_tokens: int=400, d_model: int=256) -> None:
        super().__init__()
        self.pos = nn.Parameter(torch.randn(1, max_tokens, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input image x has shape:                          # [B, W * H, d_model]
        B, L, d_model = x.shape

        if L > self.pos.shape[1]:
            raise ValueError(
                f"ResNet produced {L} tokens, but max_tokens={self.pos.shape[1]}. "
                "Increase --max_tokens or reduce --img_size."
            )

        pos = self.pos[:, :L, :].expand(B, L, d_model)      # [B, W * H, d_model]

        return pos


class FeedForward(nn.Module):
    """Feed-Forward layer.""" 
    def __init__(self, d_model: int, ff_dim: int, dropout: float) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
            # [B, num_queries, d_model] -> [B, num_queries, ff_dim]
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # [B, num_queries, ff_dim] -> [B, num_queries, d_model]
            nn.Linear(ff_dim, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class EncoderLayer(nn.Module):
    """Encoder layer.""" 
    def __init__(
            self, 
            n_heads: int, 
            d_model: int, 
            ff_dim: int, 
            dropout: float
    ) -> None:
        super().__init__()

        # [B, num_queries, d_model] -> [B, num_queries, d_model]
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.layer_norm_1 = nn.LayerNorm(d_model)

        # [B, num_queries, d_model] -> [B, num_queries, d_model]
        self.ff = FeedForward(d_model, ff_dim, dropout)
        self.layer_norm_2 = nn.LayerNorm(d_model)

    def forward(self, x_feat: torch.Tensor, pos_enc: torch.Tensor) -> torch.Tensor:
        # Input x has shape:                        # [B, num_queries, d_model]                         
        v = x_feat                                  # [B, num_queries, d_model]
        k = x_feat + pos_enc                        # [B, num_queries, d_model]
        q = x_feat + pos_enc                        # [B, num_queries, d_model]

        # Self attention layer & norm
        x_attn, _ = self.attn(q, k, v)              # [B, num_queries, d_model]
        x_norm = self.layer_norm_1(x_feat + x_attn) # [B, num_queries, d_model]

        # FFN layer & norm
        x_ff = self.ff(x_norm)                      # [B, num_queries, d_model]
        x_norm = self.layer_norm_2(x_ff + x_norm)   # [B, num_queries, d_model]
        
        return x_norm


class Encoder(nn.Module):
    """Encoder.""" 
    def __init__(
            self, 
            n_layers: int, 
            n_heads: int, 
            d_model: int, 
            ff_dim: int, 
            dropout: float
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(n_heads, d_model, ff_dim, dropout) 
            for _ in range(n_layers)
        ])
    
    def forward(self, x_feat: torch.Tensor, pos_enc: torch.Tensor) -> torch.Tensor:
        for encoder in self.layers:
            x_feat = encoder(x_feat, pos_enc)
        return x_feat


class DecoderLayer(nn.Module):
    """Decoder layer.""" 
    def __init__(
            self, 
            n_heads: int, 
            d_model: int, 
            ff_dim: int, 
            dropout: float
    ) -> None:
        super().__init__()

        # [B, num_queries, d_model] -> [B, num_queries, d_model]
        self.attn_1 = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.layer_norm_1 = nn.LayerNorm(d_model)

        self.attn_2 = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.layer_norm_2 = nn.LayerNorm(d_model)

        # [B, num_queries, d_model] -> [B, num_queries, d_model]
        self.ff = FeedForward(d_model, ff_dim, dropout)
        self.layer_norm_3 = nn.LayerNorm(d_model)

        # Manually set this to true in self.detr...
        self.save_attention_weight_ = False     # For visualization
        self.attention_weight_ = None           # The attention weight

    def forward(
            self, 
            x_enc: torch.Tensor, 
            x_dec: torch.Tensor, 
            pos_enc: torch.Tensor, 
            query: torch.Tensor,
    ) -> torch.Tensor:
        v = x_dec                                             # [B, num_queries, d_model]
        k = x_dec + query                                     # [B, num_queries, d_model]
        q = x_dec + query                                     # [B, num_queries, d_model]

        # Self-attention
        x_dec_attn, _ = self.attn_1(q, k, v)                  # [B, num_queries, d_model]
        x_dec_norm = self.layer_norm_1(x_dec_attn + x_dec)    # [B, num_queries, d_model]

        v = x_enc                                             # [B, H * W, d_model]
        k = x_enc + pos_enc                                   # [B, H * W, d_model]
        q = x_dec_norm + query                                # [B, num_queries, d_model]

        # Cross-attention
        cross_attn, cross_attn_weights = self.attn_2(q, k, v) # [B, num_queries, d_model]
        cross_norm = self.layer_norm_2(cross_attn+x_dec_norm) # [B, num_queries, d_model]

        # FFN layer
        cross_ff = self.ff(cross_norm)                        # [B, num_queries, d_model]
        cross_norm = self.layer_norm_3(cross_ff + cross_norm) # [B, num_queries, d_model]

        if self.save_attention_weight_:
            self.attention_weight_ = cross_attn_weights       # [B, num_heads, H * W]

        return cross_norm


class Decoder(nn.Module):
    """Decoder."""
    def __init__(
            self, 
            n_layers: int, 
            n_heads: int, 
            d_model: int, 
            ff_dim: int, 
            dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(n_heads, d_model, ff_dim, dropout) 
            for _ in range(n_layers)
        ])

        self.outs_ = []                 # For auxilary loss
        self.attention_weights_ = []    # For visualization (visualize decoder attention)

    def forward(
            self, 
            x_enc: torch.Tensor, 
            x_dec: torch.Tensor, 
            pos_enc: torch.Tensor, 
            queries: torch.Tensor,
    ) -> torch.Tensor:
        self.outs_ = [] 
        self.attention_weights_ = []
        for decoder in self.layers:
            x_dec = decoder(x_enc, x_dec, pos_enc, queries)

            self.outs_.append(x_dec)

            if decoder.save_attention_weight_:
                self.attention_weights_.append(decoder.attention_weight_)
        
        return x_dec


class Transformer(nn.Module):
    """Transformer model.""" 
    def __init__(
            self, 
            encoder_layers: int,
            encoder_heads: int, 
            decoder_layers: int,
            decoder_heads: int, 
            d_model: int, 
            ff_dim: int, 
            dropout: float
    ) -> None:
        super().__init__()

        self.encoder = Encoder(
            encoder_layers, encoder_heads, d_model, ff_dim, dropout
        )
        self.decoder = Decoder(
            decoder_layers, decoder_heads, d_model, ff_dim, dropout
        )

    def forward(
            self, 
            x_enc: torch.Tensor, 
            x_dec: torch.Tensor, 
            pos_enc: torch.Tensor, 
            queries: torch.Tensor
    ) -> torch.Tensor:
        x_enc = self.encoder(x_enc, pos_enc)
        x_dec = self.decoder(x_enc, x_dec, pos_enc, queries)
        return x_dec


class ViTTransformer(nn.Module):
    """Pretrained image encoder followed by the repository's DETR decoder."""

    def __init__(
            self, decoder_layers: int, decoder_heads: int, d_model: int,
            ff_dim: int, dropout: float, pretrained: bool, freeze_encoder: bool,
    ) -> None:
        super().__init__()
        self.encoder = ViTPatchEncoder(pretrained=pretrained, frozen=freeze_encoder)
        self.input_proj = nn.Linear(self.encoder.hidden_dim, d_model)
        self.position_proj = nn.Linear(self.encoder.hidden_dim, d_model, bias=False)
        self.decoder = Decoder(decoder_layers, decoder_heads, d_model, ff_dim, dropout)

    def forward(self, images: torch.Tensor, queries: Queries) -> torch.Tensor:
        memory, positions = self.encoder(images)
        memory = self.input_proj(memory)
        positions = self.position_proj(positions)
        query = queries(memory)
        return self.decoder(memory, torch.zeros_like(query), positions, query)


class DETR(nn.Module):
    """
    Detection Transformer.
    
    Reference:
    ----------
    End-to-End Object Detection with Transformers, Carion et al., 2020
    https://arxiv.org/abs/2005.12872 
    """
    def __init__(
            self,
            d_model: int,
            encoder_layers: int,
            encoder_heads: int,
            decoder_layers: int,
            decoder_heads: int,
            n_classes: int,
            num_queries: int,
            ff_dim: int=2048,
            dropout: float=0.1,
            max_tokens: int=400,
            *,
            architecture: str="resnet50",
            pretrained: bool=True,
            freeze_encoder: bool=False,
    ) -> None:
        """
        Detection Transformer

        n_classes INCLUDES no-object: foreground IDs are 0..n_classes-2,
        and the no-object ID is n_classes-1 (VOC: 21 outputs).

        architecture="resnet50" preserves the original parameter names.
        architecture="vit_b_32" replaces both ResNet and the DETR encoder.
        encoder_layers, encoder_heads and max_tokens apply only to ResNet.
        
        Reference:
        ---------- 
        End-to-End Object Detection with Transformers, Carion et al., 2020
        https://arxiv.org/abs/2005.12872 
        """ 
        super().__init__()
        if architecture not in {"resnet50", "vit_b_32"}:
            raise ValueError(f"Unknown architecture: {architecture}")
        if d_model <= 0 or decoder_heads <= 0 or d_model % decoder_heads:
            raise ValueError("d_model must be positive and divisible by decoder_heads")
        if decoder_layers < 1 or n_classes < 2 or num_queries < 1:
            raise ValueError("Need decoder_layers >= 1, n_classes >= 2 and num_queries >= 1")
        if architecture == "resnet50":
            if encoder_layers < 1 or encoder_heads <= 0 or d_model % encoder_heads:
                raise ValueError("Need encoder_layers >= 1 and d_model divisible by encoder_heads")
            if not isinstance(max_tokens, int) or max_tokens < 1:
                raise ValueError("max_tokens must be a positive integer")

        self.architecture = architecture
        self.n_classes = n_classes
        self.num_queries = num_queries
        self.feature_grid_size = None
        self.model_config = dict(
            d_model=d_model, encoder_layers=encoder_layers, encoder_heads=encoder_heads,
            decoder_layers=decoder_layers, decoder_heads=decoder_heads,
            n_classes=n_classes, num_queries=num_queries, ff_dim=ff_dim,
            dropout=dropout, max_tokens=max_tokens, architecture=architecture,
            pretrained=pretrained, freeze_encoder=freeze_encoder,
        )

        if architecture == "resnet50":
            self.backbone = ResNet50Backbone(pretrained=pretrained)
            self.spatial_to_sequence = SpatialToSequence(2048, d_model)
            self.transformer = Transformer(
                encoder_layers, encoder_heads, decoder_layers, decoder_heads,
                d_model, ff_dim, dropout,
            )
            self.pos_enc = PositionalEncoding(max_tokens, d_model)
            if freeze_encoder:
                self.backbone.requires_grad_(False)
        else:
            self.transformer = ViTTransformer(
                decoder_layers, decoder_heads, d_model, ff_dim, dropout,
                pretrained=pretrained, freeze_encoder=freeze_encoder,
            )

        # [] -> [B, num_queries, d_model] 
        self.queries = Queries(num_queries, d_model) 

        # [B, num_queries, d_model] -> [B, num_queries, n_classes] 
        self.proj_class = nn.Linear(d_model, n_classes)
        
        self.proj_bbox = nn.Sequential(
            # [B, num_queries, d_model] -> [B, num_queries, d_model] 
            nn.Linear(d_model, d_model),
            nn.ReLU(),

            # [B, num_queries, d_model] -> [B, num_queries, d_model] 
            nn.Linear(d_model, d_model),
            nn.ReLU(),

            # [B, num_queries, d_model] -> [B, num_queries, 4] 
            nn.Linear(d_model, 4),
            nn.Sigmoid()
        )

        self._init_params()

    def forward(self, x_img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.architecture == "resnet50":
            x_feat = self.backbone(x_img)
            self.feature_grid_size = tuple(x_feat.shape[-2:])
            x_feat = self.spatial_to_sequence(x_feat)
            pos_enc = self.pos_enc(x_feat)
            queries = self.queries(x_feat)
            x_tr = self.transformer(x_feat, torch.zeros_like(queries), pos_enc, queries)
        else:
            x_tr = self.transformer(x_img, self.queries)
            self.feature_grid_size = (x_img.shape[-2] // 32, x_img.shape[-1] // 32)

        x_logits = self.proj_class(x_tr)                # [B, num_queries, n_classes]
        x_bbox = self.proj_bbox(x_tr)                   # [B, num_queries, 4]
        return x_logits, x_bbox

    def pretrained_parameters(self):
        """Parameters assigned the smaller feature-encoder learning rate."""
        if self.architecture == "resnet50":
            return self.backbone.parameters()
        return self.transformer.encoder.parameters()
    
    def _init_params(self) -> None:
        if self.architecture == "resnet50":
            modules = chain(
                self.transformer.modules(), self.proj_bbox.modules(),
                self.proj_class.modules(), self.spatial_to_sequence.modules(),
            )
        else:
            # Do not walk transformer.modules(): that would overwrite pretrained ViT!
            modules = chain(
                self.transformer.decoder.modules(), self.transformer.input_proj.modules(),
                self.transformer.position_proj.modules(), self.proj_bbox.modules(),
                self.proj_class.modules(),
            )
        for m in modules:
            if isinstance(m, nn.Linear): 
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d): 
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)