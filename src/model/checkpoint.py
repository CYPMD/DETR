"""Load self-describing checkpoints or explicitly configured legacy weights."""

from collections.abc import Mapping

import torch

from src.model.detr import DETR


def read_checkpoint(path):
    # Only tensor/primitive checkpoints from this repository are supported.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Expected a state dict or a checkpoint mapping")
    state = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Checkpoint contains no model weights")
    if not all(isinstance(key, str) and torch.is_tensor(value) for key, value in state.items()):
        raise ValueError("Expected a raw state dict or the model_state_dict checkpoint key")
    return checkpoint, state


def load_weights(model: DETR, path) -> None:
    """Strictly load complete detection weights; never silently skip layers."""
    checkpoint, state = read_checkpoint(path)
    config = checkpoint.get("model_config")
    if config and config["architecture"] != model.architecture:
        raise ValueError("Checkpoint architecture differs; use --init_decoder_from to transfer the decoder")
    if config:
        fields = ["d_model", "decoder_layers", "decoder_heads", "ff_dim", "num_queries", "n_classes"]
        if model.architecture == "resnet50":
            fields += ["encoder_layers", "encoder_heads", "max_tokens"]
        mismatches = [key for key in fields if config[key] != model.model_config[key]]
        if mismatches:
            raise ValueError(f"Model configuration differs: {', '.join(mismatches)}")
    model.load_state_dict(state, strict=True)


def load_decoder_weights(model: DETR, path) -> None:
    """Warm-start the compatible queries, decoder and detection heads only.

    Requires matching d_model, decoder layers/heads, ff_dim, queries and classes.
    A legacy raw state dict cannot verify attention head count; supply the same
    decoder_heads setting used to train it.
    """
    checkpoint, state = read_checkpoint(path)
    config = checkpoint.get("model_config")
    if config:
        fields = ("d_model", "decoder_layers", "decoder_heads", "ff_dim", "num_queries", "n_classes")
        mismatches = [key for key in fields if config[key] != model.model_config[key]]
        if mismatches:
            raise ValueError(f"Decoder configuration differs: {', '.join(mismatches)}")
    prefixes = ("transformer.decoder.", "queries.", "proj_class.", "proj_bbox.")
    expected = {key: value for key, value in model.state_dict().items() if key.startswith(prefixes)}
    incoming = {key: value for key, value in state.items() if key.startswith(prefixes)}
    if incoming.keys() != expected.keys():
        raise ValueError("Decoder checkpoint keys differ; match decoder_layers and model settings")
    bad_shapes = [key for key in expected if expected[key].shape != incoming[key].shape]
    if bad_shapes:
        raise ValueError(f"Decoder checkpoint shapes differ: {', '.join(bad_shapes)}")
    # Other keys are deliberately absent: the new encoder retains ViT weights.
    model.load_state_dict(incoming, strict=False)


def load_model(path, device="cpu") -> DETR:
    """Reconstruct a saved model without downloading pretrained weights."""
    checkpoint, state = read_checkpoint(path)
    if "model_config" not in checkpoint:
        raise ValueError("Legacy raw weights require DETR(..., pretrained=False) and load_weights(model, path)")
    config = dict(checkpoint["model_config"])
    config["pretrained"] = False
    model = DETR(**config)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
