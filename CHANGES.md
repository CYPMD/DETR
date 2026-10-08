# Files to add or replace

All paths below are relative to the repository root. The bundled `changes.patch`
contains the same changes. The full `cypmd-detr/` folder is already updated.

## New files

| File | Purpose |
| --- | --- |
| `src/model/vit.py` | Pretrained ViT-B/32 patch encoder, position interpolation, spatial token output, optional freezing |
| `src/model/checkpoint.py` | Strict detector loading, self-describing model loading, and validated decoder transfer |
| `tests/test_vit_detr.py` | Offline integration/gradient/checkpoint tests plus opt-in full ViT and real pretrained checks |
| `pytest.ini` | Test collection and optional integration/pretraining markers |
| `requirements-dev.txt` | Test dependencies |
| `VIT_GUIDE.md` | Architecture, training commands, checkpoint migration, inference and limitations |
| `VALIDATION.md` | Executed checks and what was not measured |
| `CHANGES.md` | This file |

## Modified files

| File | Change |
| --- | --- |
| `src/model/detr.py` | Select `resnet50` or `vit_b_32`; preserve the baseline state-dict layout; add width/position projections and parameter grouping API; prevent ViT reinitialization; report feature-grid size |
| `src/model/__init__.py` | Export checkpoint helpers |
| `src/__init__.py` | Export checkpoint helpers through the existing public package |
| `src/trainer/trainer.py` | Generalize optimizer parameter groups; save configuration with weights; keep each experiment's outputs together |
| `train.py` | Add model/weight/freeze/encoder-rate/output options; fix two-integer image-size parsing and integer max_tokens; validate configurations |
| `src/visualize/visualizer.py` | Use height/width consistently and reshape attention using the actual encoder grid |
| `requirements.txt` | Populate the previously empty dependency list |
| `tests/test_detr.py` | Disable downloads in baseline tests |
| `README.md` | Link the ViT guide; add model selection examples; correct the original example's output shape and token capacity |

The dataset, transforms, loss, matching algorithm, and evaluator implementations
are unchanged. The existing decoder implementation and detection-head structure
are reused. Original trained ResNet weights can still be loaded with matching
settings; the new encoder does not make full ResNet and ViT checkpoints
interchangeable. Use the explicit decoder-transfer option for that experiment.

## Behavior changes to account for

- Training outputs now go to `runs/<model>/` by default, or `--output_dir`.
- `--img_size` accepts `HEIGHT WIDTH`, such as `--img_size 640 640`.
- `--max_tokens` is now parsed as an integer.
- ViT ignores the baseline's `encoder_layers`, `encoder_heads` and `max_tokens`.
- `n_classes` still includes the final no-object class; only its misleading
  documentation was corrected.
- Checkpoints include model/run configuration, but not optimizer/scheduler/RNG
  state. `--weights` starts a new training run initialized from detector weights.
