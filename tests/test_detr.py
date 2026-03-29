import pytest
import torch

from src import DETR



@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def detr_model(device):
    model = DETR(
        d_model=256,
        encoder_layers=2,
        encoder_heads=2,
        decoder_layers=2,
        decoder_heads=2,
        n_classes=91,
        num_queries=100,
        ff_dim=2048,
        dropout=0.1,
    ).to(device)
    return model


class TestDETR:
    def test_forward_output_shapes(self, detr_model, device):
        x = torch.randn(4, 3, 128, 128, device=device)

        logits, bbox = detr_model(x)

        assert logits.shape == (4, 100, 91)  # n_classes + 1
        assert bbox.shape == (4, 100, 4)

    def test_bbox_range(self, detr_model, device):
        x = torch.randn(2, 3, 128, 128, device=device)

        _, bbox = detr_model(x)

        assert torch.all(bbox >= 0.0)
        assert torch.all(bbox <= 1.0)

    def test_outputs_are_finite(self, detr_model, device):
        x = torch.randn(2, 3, 128, 128, device=device)

        logits, bbox = detr_model(x)

        assert torch.isfinite(logits).all()
        assert torch.isfinite(bbox).all()

    def test_backward_pass(self, detr_model, device):
        x = torch.randn(2, 3, 128, 128, device=device)

        logits, bbox = detr_model(x)
        loss = logits.mean() + bbox.mean()
        loss.backward()

        grads = [p.grad for p in detr_model.parameters() if p.requires_grad]
        assert any(g is not None for g in grads)

    def test_batch_size_one(self, detr_model, device):
        x = torch.randn(1, 3, 128, 128, device=device)

        logits, bbox = detr_model(x)

        assert logits.shape == (1, 100, 91)
        assert bbox.shape == (1, 100, 4)