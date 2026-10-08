"""Offline checks with real TorchVision blocks; no downloaded weights/dataset.

Most tests use a reduced-width ViT to exercise the exact adapter cheaply.
RUN_FULL_VIT_TESTS=1 also tests the unmodified ViT-B/32 architecture.
"""

import os

import pytest
import torch
from torch import nn
from torchvision.models.vision_transformer import VisionTransformer

from src import DETR, DETRLoss, DETRTrainer, load_model, load_weights, load_decoder_weights
import src.model.vit as vit_module
from train import parse_args


def settings(**overrides):
    config = dict(
        d_model=32, encoder_layers=1, encoder_heads=4, decoder_layers=2,
        decoder_heads=4, n_classes=4, num_queries=5, ff_dim=64,
        dropout=0.0, max_tokens=400, architecture="vit_b_32", pretrained=False,
    )
    config.update(overrides)
    return config


@pytest.fixture
def tiny_vit(monkeypatch):
    created = []

    def build(*, weights):
        vit = VisionTransformer(
            image_size=224, patch_size=32, num_layers=2, num_heads=4,
            hidden_dim=32, mlp_dim=64, dropout=0.0, attention_dropout=0.0,
        )
        # Record values BEFORE DETR initialization to detect accidental reinit.
        reference = {k: v.clone() for k, v in vit.state_dict().items() if not k.startswith("heads.")}
        created.append((weights, reference))
        return vit

    monkeypatch.setattr(vit_module, "vit_b_32", build)
    return created


def targets():
    return [
        {"boxes": torch.tensor([[0.4, 0.5, 0.2, 0.3]]), "labels": torch.tensor([1])},
        {"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.long)},
    ]


def trainer_for(model, tmp_path):
    return DETRTrainer(
        model, n_epochs=1, learning_rate=1e-4, learning_rate_backbone=5e-5,
        weight_decay=1e-4, weight_decay_backbone=1e-4, lr_schedule=0.1,
        lambda_l1=5.0, lambda_giou=2.0, device="cpu", verbose=False,
        learning_rate_encoder=1e-5, output_dir=str(tmp_path),
    )


def test_pretrained_encoder_is_never_reinitialized(tiny_vit):
    model = DETR(**settings(pretrained=True))
    requested, before = tiny_vit[0]
    assert requested == vit_module.ViT_B_32_Weights.IMAGENET1K_V1
    after = model.transformer.encoder.vit.state_dict()
    assert before.keys() == after.keys()
    assert all(torch.equal(before[key], after[key]) for key in before)
    assert not hasattr(model, "backbone")
    assert not hasattr(model, "spatial_to_sequence")
    assert isinstance(model.transformer.encoder.vit.heads, nn.Identity)


def test_native_resolution_matches_torchvision_encoder(tiny_vit):
    model = DETR(**settings()).eval()
    encoder = model.transformer.encoder
    image = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        actual, positions = encoder(image)
        patches = encoder.vit.conv_proj(image).flatten(2).transpose(1, 2)
        tokens = torch.cat((encoder.vit.class_token, patches), dim=1)
        expected = encoder.vit.encoder(tokens)[:, 1:]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(positions, encoder.vit.encoder.pos_embedding[:, 1:])


@pytest.mark.parametrize("height,width", [(64, 96), (224, 224), (640, 640), (640, 384)])
def test_patch_shapes_attention_and_outputs(tiny_vit, height, width):
    model = DETR(**settings()).eval()
    for layer in model.transformer.decoder.layers:
        layer.save_attention_weight_ = True
    with torch.no_grad():
        logits, boxes = model(torch.randn(1, 3, height, width))
    assert logits.shape == (1, 5, 4)
    assert boxes.shape == (1, 5, 4)
    assert torch.isfinite(logits).all() and torch.isfinite(boxes).all()
    assert ((boxes >= 0) & (boxes <= 1)).all()
    assert model.feature_grid_size == (height // 32, width // 32)
    expected_tokens = (height // 32) * (width // 32)
    assert all(w.shape == (1, 5, expected_tokens) for w in model.transformer.decoder.attention_weights_)
    assert len(model.transformer.decoder.outs_) == 2


def test_real_detection_loss_backward_and_auxiliary_outputs(tiny_vit):
    model = DETR(**settings())
    logits, boxes = model(torch.randn(2, 3, 64, 96))
    criterion = DETRLoss(5.0, 2.0, 4)
    loss = criterion(logits, boxes, targets())[0]
    for intermediate in model.transformer.decoder.outs_[:-1]:
        loss = loss + criterion(model.proj_class(intermediate), model.proj_bbox(intermediate), targets())[0]
    assert torch.isfinite(loss)
    loss.backward()
    parameters = [
        model.transformer.encoder.vit.conv_proj.weight,
        model.transformer.encoder.vit.encoder.pos_embedding,
        model.transformer.encoder.vit.encoder.layers[0].self_attention.in_proj_weight,
        model.transformer.input_proj.weight, model.transformer.position_proj.weight,
        model.queries.queries, model.proj_class.weight, model.proj_bbox[0].weight,
    ]
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in parameters)


@pytest.mark.parametrize("frozen", [False, True])
def test_training_step_optimizer_groups_and_freezing(tiny_vit, tmp_path, frozen):
    model = DETR(**settings(freeze_encoder=frozen))
    trainer = trainer_for(model, tmp_path)
    groups = trainer.optimizer.param_groups
    optimizer_ids = [id(p) for group in groups for p in group["params"]]
    assert len(optimizer_ids) == len(set(optimizer_ids))
    assert set(optimizer_ids) == {id(p) for p in model.parameters() if p.requires_grad}
    if frozen:
        assert len(groups) == 1
    else:
        assert groups[0]["lr"] == 1e-5 and groups[1]["lr"] == 1e-4
        assert {id(p) for p in groups[0]["params"]} == {id(p) for p in model.pretrained_parameters()}
    encoder_before = model.transformer.encoder.vit.conv_proj.weight.detach().clone()
    decoder_before = model.proj_class.weight.detach().clone()
    trainer.train_one_epoch([(torch.randn(2, 3, 64, 96), targets())], epoch=1)
    assert not torch.equal(decoder_before, model.proj_class.weight)
    if frozen:
        assert not model.transformer.encoder.vit.training
        assert all(p.grad is None and not p.requires_grad for p in model.pretrained_parameters())
        assert torch.equal(encoder_before, model.transformer.encoder.vit.conv_proj.weight)
    else:
        assert not torch.equal(encoder_before, model.transformer.encoder.vit.conv_proj.weight)


def test_checkpoint_round_trip_and_decoder_transfer(tiny_vit, tmp_path):
    model = DETR(**settings()).eval()
    trainer = trainer_for(model, tmp_path)
    trainer._checkpoint()
    restored = load_model(tmp_path / "checkpoint.pt")
    assert tiny_vit[-1][0] is None  # Loading does not download pretraining weights.
    image = torch.randn(1, 3, 64, 96)
    with torch.no_grad():
        before, after = model(image), restored(image)
    for expected, actual in zip(before, after):
        torch.testing.assert_close(expected, actual)
    load_weights(restored, tmp_path / "model_weights.pt")
    destination = DETR(**settings())
    encoder_before = destination.transformer.encoder.vit.conv_proj.weight.detach().clone()
    load_decoder_weights(destination, tmp_path / "checkpoint.pt")
    torch.testing.assert_close(destination.proj_class.weight, model.proj_class.weight)
    torch.testing.assert_close(destination.queries.queries, model.queries.queries)
    assert torch.equal(encoder_before, destination.transformer.encoder.vit.conv_proj.weight)
    mismatched = DETR(**settings(decoder_heads=2))
    with pytest.raises(ValueError, match="decoder_heads"):
        load_decoder_weights(mismatched, tmp_path / "checkpoint.pt")
    with pytest.raises(ValueError, match="decoder_heads"):
        load_weights(mismatched, tmp_path / "checkpoint.pt")


def test_legacy_resnet_weights_and_cross_architecture_warm_start(tiny_vit, tmp_path):
    baseline = DETR(**settings(architecture="resnet50"))
    path = tmp_path / "legacy.pt"
    torch.save(baseline.state_dict(), path)
    restored = DETR(**settings(architecture="resnet50"))
    load_weights(restored, path)
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in baseline.state_dict().items())
    new_model = DETR(**settings())
    encoder_before = new_model.transformer.encoder.vit.conv_proj.weight.detach().clone()
    load_decoder_weights(new_model, path)
    assert torch.equal(new_model.queries.queries, baseline.queries.queries)
    assert torch.equal(new_model.proj_class.weight, baseline.proj_class.weight)
    assert torch.equal(encoder_before, new_model.transformer.encoder.vit.conv_proj.weight)


def test_invalid_patch_sizes_and_cli_settings(tiny_vit):
    model = DETR(**settings())
    with pytest.raises(ValueError, match="multiples of 32"):
        model(torch.randn(1, 3, 65, 96))
    with pytest.raises(SystemExit):
        parse_args(["--model", "vit_b_32", "--img_size", "640", "650"])
    with pytest.raises(SystemExit):
        parse_args(["--model", "resnet50", "--img_size", "800", "1344"])
    args = parse_args(["--model", "vit_b_32", "--img_size", "640", "384"])
    assert args.img_size == (640, 384) and args.output_dir == os.path.join("runs", "vit_b_32")
    assert parse_args([]).model == "resnet50"


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("RUN_FULL_VIT_TESTS") != "1", reason="Opt in to full ViT-B/32 CPU/memory use")
def test_full_vit_b32_forward_and_backward():
    model = DETR(**settings())
    assert model.transformer.encoder.hidden_dim == 768
    assert len(model.transformer.encoder.vit.encoder.layers) == 12
    logits, boxes = model(torch.randn(1, 3, 64, 96))
    (logits.square().mean() + boxes.mean()).backward()
    assert torch.isfinite(model.transformer.encoder.vit.conv_proj.weight.grad).all()
    model.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad():
        logits, boxes = model(torch.randn(1, 3, 640, 640))
    assert logits.shape == (1, 5, 4) and boxes.shape == (1, 5, 4)
    assert model.feature_grid_size == (20, 20)


@pytest.mark.pretrained
@pytest.mark.skipif(os.environ.get("RUN_PRETRAINED_VIT_TESTS") != "1", reason="Opt in to pretrained weight download")
def test_official_pretrained_weights_are_preserved():
    model = DETR(**settings(pretrained=True)).eval()
    # TorchVision migrates legacy MLP key names while loading the official
    # checkpoint. Compare its loaded parameters, not the pre-migration keys.
    reference = vit_module.vit_b_32(weights=vit_module.ViT_B_32_Weights.IMAGENET1K_V1).state_dict()
    actual = model.transformer.encoder.vit.state_dict()
    expected_keys = {key for key in reference if not key.startswith("heads.")}
    assert actual.keys() == expected_keys
    assert all(torch.equal(actual[key], reference[key]) for key in expected_keys)
    del reference
    with torch.no_grad():
        logits, boxes = model(torch.randn(1, 3, 640, 640))
    assert torch.isfinite(logits).all() and torch.isfinite(boxes).all()
    assert model.feature_grid_size == (20, 20)
