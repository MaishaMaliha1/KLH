import torch
from torch import nn

from klh.attribution import KLHessianAttributor
from klh.config import AttributionConfig


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3 * 4 * 4, 3)

    def forward(self, x):
        return self.linear(x.flatten(1))


class RightHalfClassifier(nn.Module):
    def forward(self, x):
        s = x[:, :, :, 2:].sum(dim=(1, 2, 3))
        return torch.stack([s, -s, 0.5 * s], dim=1)


def test_attribution_shapes_and_finiteness():
    model = TinyClassifier().eval()
    image = torch.linspace(-1.0, 1.0, 3 * 4 * 4).reshape(1, 3, 4, 4)
    masks = torch.zeros(2, 4, 4)
    masks[0, :, :2] = 1
    masks[1, :, 2:] = 1

    result = KLHessianAttributor(model, AttributionConfig()).attribute(image, masks)
    assert result.scores.shape == (2,)
    assert result.kl_shift.shape == (2,)
    assert result.curvature.shape == (2,)
    assert result.amplifier.shape == (2,)
    assert torch.isfinite(result.scores).all()
    assert (result.scores >= 0).all()


def test_zero_gradient_region_has_zero_score():
    model = RightHalfClassifier().eval()
    image = torch.ones(1, 3, 4, 4)
    mask = torch.zeros(1, 4, 4)
    mask[0, :, :2] = 1
    result = KLHessianAttributor(model).attribute(image, mask, class_index=0)
    assert result.scores[0].abs() < 1e-10
