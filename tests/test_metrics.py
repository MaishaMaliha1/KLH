import torch
from torch import nn

from klh.metrics import part_faithfulness, part_scores_to_pixel_map, pixel_faithfulness


class SumClassifier(nn.Module):
    def forward(self, x):
        s = x.flatten(1).sum(dim=1)
        return torch.stack([s, -s], dim=1)


def test_part_score_projection():
    scores = torch.tensor([1.0, 2.0])
    masks = torch.zeros(2, 4, 4)
    masks[0, :, :2] = 1
    masks[1, :, 2:] = 1
    projected = part_scores_to_pixel_map(scores, masks, smoothing_kernel=1)
    assert torch.all(projected[:, :2] == 1.0)
    assert torch.all(projected[:, 2:] == 2.0)


def test_faithfulness_returns_finite_values():
    model = SumClassifier().eval()
    image = torch.ones(1, 3, 4, 4)
    baseline = torch.zeros_like(image)
    masks = torch.zeros(2, 4, 4)
    masks[0, :, :2] = 1
    masks[1, :, 2:] = 1
    scores = torch.tensor([2.0, 1.0])
    part = part_faithfulness(model, image, baseline, masks, scores, 0)
    pixel_map = part_scores_to_pixel_map(scores, masks, smoothing_kernel=1)
    pixel = pixel_faithfulness(model, image, baseline, pixel_map, 0, steps=4)
    for result in (part, pixel):
        assert torch.isfinite(torch.tensor(result.insertion))
        assert torch.isfinite(torch.tensor(result.deletion))
        assert torch.isfinite(torch.tensor(result.aopc))
