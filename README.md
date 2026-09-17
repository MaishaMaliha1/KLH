# Beyond Gradients: Curvature-Aware Part-Level Explanations for Vision Models

Reference implementation of the KL-Hessian attribution framework from our CVPRW 2026 paper.

KL-Hessian is a training-free, part-level attribution framework for frozen vision classifiers. It combines symmetric KL divergence in probability space with directional curvature to measure both class-belief redistribution and local nonlinear amplification.

## Method

For each image part, the implementation:

1. restricts the explained-class gradient to that part and normalizes it;
2. estimates a Hessian-vector product with symmetric finite differences;
3. masks and normalizes the second-order direction;
4. blends first- and second-order directions;
5. probes the image in positive and negative directions;
6. measures the resulting belief shift with symmetric KL divergence;
7. computes directional curvature; and
8. applies a nonnegative curvature amplifier.


## Supported vision models

The model registry contains the model families described in the paper:

- ViT-B/16
- ConvNeXtV2-B
- SwinV2-B
- DINO ViT-B/16 for the retinal case study

Aliases:

```text
vit_b16
convnextv2_b
swinv2_b
dino_vit_b16
```

The registry is explicit and fails if the required `timm` entry is unavailable rather than silently substituting another checkpoint.

## Part generation

The package includes:

- DINOv2 token clustering with k-means for general images;
- MediaPipe Face Mesh regions for faces;
- OpenPifPaf-based animal-pose regions;
- YOLOv8 vehicle localization followed by DINOv2 subdivision;
- uniform-grid regions; and
- direct loading of externally prepared masks.

Generated masks are converted to non-overlapping regions before attribution when necessary.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Optional segmentation dependencies:

```bash
python -m pip install -e '.[parts]'
```

Development installation:

```bash
python -m pip install -e '.[dev]'
```

## Configuration

Natural-image attribution defaults:

```yaml
attribution:
  epsilon: 0.20
  delta: 0.06
  curvature_blend: 0.40
  curvature_weight: 1.0
```

Retinal configuration:

```yaml
attribution:
  epsilon: 0.35
  delta: 0.12
  curvature_blend: 0.35
  curvature_weight: 1.0
```

Region counts used in the provided configurations follow the paper's reported settings: 12 for CIFAR-10/ImageNet-1K, 14 for FFHQ, and 10 for IDRiD.

## Command-line usage

```bash
klh path/to/image.jpg \
    --config configs/default.yaml \
    --output outputs/example.json
```

With precomputed masks:

```bash
klh path/to/image.jpg \
    --config configs/default.yaml \
    --masks path/to/masks.pt \
    --output outputs/example.json
```

A general single-image runner is also available:

```bash
python scripts/evaluate.py path/to/image.jpg \
    --model vit_b16 \
    --parts dino \
    --output outputs/example.json
```

## Python API

```python
from klh import AttributionConfig, KLHessianAttributor, create_model

bundle = create_model("vit_b16")
config = AttributionConfig(
    epsilon=0.20,
    delta=0.06,
    curvature_blend=0.40,
    curvature_weight=1.0,
)
attributor = KLHessianAttributor(bundle.model, config)
result = attributor.attribute(image_tensor, masks)
print(result.scores)
```

## Testing

```bash
python -m pip install -e '.[dev]'
pytest
```

GitHub Actions runs the test suite and Ruff checks on pushes and pull requests.


## Citation

If you use this code in your research or build upon this work, please cite:

```bibtex
@InProceedings{Maliha_2026_CVPR,
    author    = {Maliha, Maisha and Hougen, Dean F.},
    title     = {Beyond Gradients: Curvature-Aware Part-Level Explanations for Vision Models},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2026},
    pages     = {562-571}
}
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
