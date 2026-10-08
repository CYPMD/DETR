# Validation performed

Prepared on 2026-10-08 from the attached repository text.

Test environment: Python 3.12, PyTorch 2.2.2+cpu, TorchVision 0.17.2+cpu,
NumPy 1.26.4, TorchMetrics 1.3.2 and pytest 8.3.5. Tests used CPU with two
OpenMP/MKL threads. OpenCV Headless 4.11.0.86 was used for imports in this
non-GUI environment; the repository requirements retain regular OpenCV for
the original webcam/video UI.

## Passed

19 distinct pytest cases passed across three targeted runs:

| Run | Result |
| --- | --- |
| Offline suite | 17 passed; full-size integration test skipped |
| `RUN_FULL_VIT_TESTS=1 python -m pytest -q -m integration` | Full-size ViT-B/32 test passed |
| `RUN_PRETRAINED_VIT_TESTS=1 python -m pytest -q -m pretrained` | Authentic pretrained-weight test passed |

The pretrained test was added after the offline suite ran. In the final test
collection there are 19 cases; a default run opts out of the two expensive
integration/pretrained cases. The runs above exercised all 19 cases.

The checks cover:

- Original DETR output shapes, normalized box range, finite values and gradients.
- Actual TorchVision encoder blocks with rectangular and square inputs,
  including 640 x 640 and 640 x 384.
- Exact equivalence to TorchVision's encoder computation at its native 224 x 224
  resolution, after removing CLS from the output.
- Decoder auxiliary outputs and attention-map dimensions (no CLS in memory).
- Hungarian matching and the existing detection loss with a nonempty target
  image and an empty-target image in the same batch.
- Nonzero finite gradients through patch projection, positional interpolation,
  attention, new projections, queries, and detection heads.
- An AdamW training step, correct parameter-group coverage/rates, and frozen
  encoder behavior.
- Saving/loading detector weights, reconstructing a model without downloading,
  legacy raw-state-dict loading, and cross-architecture decoder transfer.
- Rejection of incompatible decoder settings and invalid image dimensions.
- The full 12-layer, 768-dimensional ViT-B/32: backward pass on 64 x 96 and
  forward pass on 640 x 640.
- The actual `ViT_B_32_Weights.IMAGENET1K_V1` checkpoint: every retained encoder
  tensor equals an independently loaded TorchVision model after DETR
  initialization; the pretrained detector gives finite outputs at 640 x 640.

An additional direct comparison loaded the **original attached** implementation's
state dict into the updated ResNet model with `strict=True`. Both class and box
outputs were bit-identical on a 64 x 96 input in evaluation mode with matching
settings. This comparison used the original source, not just another instance
of the updated class.

All Python source files passed syntax parsing. The CLI help was checked. The
patch was applied to a fresh reconstruction of the attachment and compared to
the delivered source tree before packaging.

## Not measured

No Pascal VOC training run, mAP comparison, CUDA throughput test, GPU memory
benchmark, or interactive webcam test was performed. Dataset files and the
user's trained checkpoint were not attached. Existing README accuracy results
belong to the original ResNet experiment and do not establish ViT performance.
The downloaded pretrained encoder weights and test checkpoints are not bundled;
the code obtains official encoder weights through TorchVision when required.
