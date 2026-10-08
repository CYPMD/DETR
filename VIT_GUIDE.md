# Train ResNet DETR and ViT-B/32 DETR in one repository

The default remains the repository's original ResNet-50 DETR. Select the new
model with `--model vit_b_32`. It replaces **both** the ResNet feature backbone
and the original DETR encoder with a pretrained TorchVision ViT-B/32. The same
object queries, decoder, detection heads, Hungarian matching, auxiliary losses,
VOC data pipeline, evaluator, and visualizer are used by both variants.

## Install or apply the changes

The complete `cypmd-detr/` folder in the download is reconstructed from the
attached repository text with these changes applied. Dataset files, trained
weights, and README image assets were not present in that text and are not
included. You can use this folder as your training repository.

If keeping your existing checkout, copy the files listed in `CHANGES.md` to the
same relative paths. Alternatively, the bundle's `changes.patch` contains all
additions and edits; from the root of your existing checkout run:

```bash
git apply --check /path/to/changes.patch
git apply /path/to/changes.patch
```

Do not apply the patch to the already updated folder. If your checkout has
additional edits, use the file list and diff to merge them instead of replacing
your local work blindly.

Use Python 3.10 or later. Keep/install a matching PyTorch and TorchVision pair
for your hardware, following the [official installer](https://pytorch.org/get-started/locally/).
Then, from the repository root:

```bash
python -m pip install -r requirements.txt
```

This implementation uses TorchVision directly; it does not require `timm` or
Hugging Face Transformers. The first pretrained ViT run downloads the standard
ImageNet-1K weights if they are not already in the PyTorch cache. A failed
download raises an error; it never silently trains a random encoder.

## Train either architecture

Regular DETR, with the original architecture and defaults:

```bash
python train.py --model resnet50 --root ./voc \
  --img_size 640 640 --num_queries 100 \
  --output_dir runs/resnet50_experiment
```

Pretrained ViT-B/32 DETR (start with batch size 1 if GPU memory is limited):

```bash
python train.py --model vit_b_32 --root ./voc \
  --img_size 640 640 --num_queries 100 --batch_size 1 \
  --learning_rate 1e-4 --learning_rate_encoder 1e-5 \
  --output_dir runs/vit_b32_experiment
```

Both commands assume VOC is already extracted at `./voc`. Add `--download` to
request dataset downloading, as in the original repo. For a controlled
comparison, set the same image size, query count, batch size, seed, decoder
settings, augmentations, evaluation settings and training budget in both runs.
The encoder architectures and pretraining differ, so this is not an equal
parameter-count comparison. If your earlier run used 25 queries, use
`--num_queries 25` in both commands.

The learning rates above are starting settings, not tuned detection results.
ViT-B/32 has greater encoder compute/memory requirements than this ResNet
baseline. Lower the batch size or image size if needed. `--freeze_encoder`
freezes the entire ViT (including its patch and position embeddings), leaving
the projections and detection decoder trainable. In baseline mode it freezes
the ResNet backbone. `--no_pretrained` enables a random-initialization ablation;
omit it for pretrained training.

## What the ViT model does

For a normalized `640 x 640` RGB image:

| Stage | Tensor shape |
| --- | --- |
| Input image | `[B, 3, 640, 640]` |
| 32 x 32 patch projection | `[B, 400, 768]` |
| Add CLS and learned spatial positions | `[B, 401, 768]` |
| Pretrained ViT encoder: 12 layers, 12 attention heads | `[B, 401, 768]` |
| Remove CLS, project encoder output to `d_model=256` | `[B, 400, 256]` |
| Existing DETR decoder, with 100 object queries | `[B, 100, 256]` |
| Class logits and normalized `cx, cy, w, h` boxes | `[B, 100, n_classes]`, `[B, 100, 4]` |

The only convolution in the ViT encoder is its pretrained patch projection:
`Conv2d(3, 768, kernel_size=32, stride=32)`. Mathematically this is a shared
linear map on flattened, non-overlapping RGB patches, matching the supplied
diagram. There is no ResNet, feature pyramid, or extra DETR encoder in this path.

TorchVision's learned `7 x 7` patch-position grid is interpolated with bicubic
interpolation to the current patch grid. The CLS position is preserved, CLS
participates in pretrained self-attention, and only spatial patch tokens reach
the detector. The ImageNet classification head is removed. A separate learned
projection of the interpolated patch positions supplies positional information
to the existing decoder's cross-attention keys. The original learned grid
remains a trainable parameter, so interpolation also works at inference time.

All pretrained patch-projection, attention, MLP, normalization, CLS, and position
parameters are kept intact at construction. DETR initialization is restricted
to new decoder/projection/detection layers. The encoder is fine-tuned by default.

Use the existing detection transforms and ImageNet normalization. Do not replace
them with the classification weights' center-crop transform: it would require
corresponding box transformations. Normalization is applied by the data
transforms, not again inside the model.

## Configuration rules

| Setting | ResNet-50 DETR | ViT-B/32 DETR |
| --- | --- | --- |
| `--model` | `resnet50` (default) | `vit_b_32` |
| Encoder width/layers/heads | `d_model`, `encoder_layers`, `encoder_heads` | Fixed at 768 / 12 / 12 by pretrained ViT |
| `--d_model`, `--decoder_layers`, `--decoder_heads`, `--ff_dim`, `--dropout` | As in the original model | Configure the DETR decoder and new projections; do not change ViT dimensions/dropout |
| `--learning_rate` | Custom transformer and heads | New projections, queries, decoder and heads |
| Feature encoder rate | `--learning_rate_backbone` | `--learning_rate_encoder` |
| Feature encoder weight decay | `--weight_decay_backbone` | `--weight_decay_encoder` |
| `--max_tokens` | Must cover `ceil(H/32) * ceil(W/32)` | Ignored; position grid is resized |
| `--img_size` | Two integers, **height width** | Two integers, **height width**, each a multiple of 32 |
| `--n_classes` | Total class outputs, including no-object | Same convention |

For Pascal VOC, `--n_classes 21` means 20 foreground classes and no-object at
index 20. Do not add another output class. Boxes stay normalized to `[0,1]` in
`cxcywh` format, and `forward()` returns exactly `(logits, boxes)` as before.

Rectangular inputs such as `--img_size 640 384` work. ViT rejects dimensions
that are not multiples of 32 rather than dropping edge pixels. The current
collate function stacks equally resized images; arbitrary ragged batches and
padding masks are not introduced. A one-patch stride of 32 can limit spatial
detail for small objects, so measure accuracy rather than assuming an improvement.

## Use your already trained DETR

Existing ResNet checkpoints retain the same state-dict keys and shapes. With the
same architecture settings, loading your old raw `.pt` file remains supported:

```python
from src import DETR, load_weights

model = DETR(
    d_model=256, encoder_layers=6, encoder_heads=8,
    decoder_layers=6, decoder_heads=8, n_classes=21,
    num_queries=100, ff_dim=2048, dropout=0.1, max_tokens=400,
    architecture="resnet50", pretrained=False,
)
load_weights(model, "old_resnet_weights.pt")
```

Match `num_queries` and all architecture settings to your actual old run.
`pretrained=False` avoids an unnecessary ResNet download before loading the
complete detector. The `pretrained` switch has no effect on the weights loaded
from your checkpoint.

To reuse the learned queries, decoder, class head and box head in the ViT model:

```bash
python train.py --model vit_b_32 --root ./voc \
  --img_size 640 640 --num_queries 100 --batch_size 1 \
  --init_decoder_from /path/to/old_resnet_weights.pt \
  --output_dir runs/vit_b32_decoder_transfer
```

This keeps the new encoder's ImageNet-pretrained weights. Decoder layer count,
width, head count, feed-forward width, query count, and class count must match
the old model. Missing or shape-incompatible decoder weights produce an error.
Legacy raw weights do not encode attention head count, so you must supply the
original `--decoder_heads` value. Feature representations change when switching
encoders; decoder transfer is an experiment, not a guarantee of faster convergence.
For a fresh decoder comparison, omit `--init_decoder_from`.

## Saved models and inference

Each output directory contains:

| File | Contents |
| --- | --- |
| `checkpoint.pt` | Model weights, model configuration, run arguments and epoch |
| `model_weights.pt` | Raw state dict for existing inference notebooks |
| `model_config.json` | Constructor arguments |
| `train_metrics.csv`, `val_metrics.csv` | Original metrics (`val` only if used) |

These are model checkpoints, not complete optimizer/RNG-resume snapshots. Using
`--weights path/to/checkpoint.pt` or `--weights path/to/model_weights.pt` loads
the entire detector strictly and starts a **new** optimizer/schedule. Set the
matching model/configuration flags. Use a separate `--output_dir` for each
experiment; another run in the same directory replaces these files.

New self-describing checkpoints can be loaded for inference without a download:

```python
from src import load_model, DETRVisualizer

model = load_model("runs/vit_b32_experiment/checkpoint.pt", device="cuda")
visualizer = DETRVisualizer(model, img_size=(640, 640), device="cuda")
# Use the existing visualizer methods as before.
```

When calling the model directly, supply RGB float tensors normalized by the
existing transforms, use `model.eval()` and `torch.no_grad()`, and use the same
image preprocessing as during training. Decoder attention maps use the actual
patch grid and support rectangular images.

## Checks

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
RUN_FULL_VIT_TESTS=1 python -m pytest -q
# Optional: also check authentic pretrained weights (downloads if uncached).
RUN_PRETRAINED_VIT_TESTS=1 python -m pytest -q -m pretrained
```

Normal tests require no downloads. Most new tests use real TorchVision encoder
blocks at reduced width to keep them fast. The opt-in integration test constructs
the actual 12-layer, 768-dimensional ViT-B/32 with random weights, checks
backpropagation, and tests a 640 x 640 forward pass. The separate pretrained
check compares every retained encoder tensor against the official checkpoint.
`VALIDATION.md` records what
was executed when this bundle was prepared. Model accuracy still requires a
training/evaluation run on your dataset.

## Upstream references

- [TorchVision ViT-B/32 and pretrained normalization](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vit_b_32.html)
- [TorchVision ViT source and positional interpolation](https://docs.pytorch.org/vision/stable/_modules/torchvision/models/vision_transformer.html)

These references describe the upstream encoder. The patch-token adapter,
decoder projections and repository integration are implemented in this bundle.
