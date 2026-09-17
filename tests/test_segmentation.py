import torch

from klh.segmentation import GridParts


def test_grid_parts_cover_image_without_overlap():
    image = torch.zeros(3, 16, 16)
    masks = GridParts(4, 4)(image)
    assert masks.shape == (16, 16, 16)
    assert torch.all(masks.sum(dim=0) == 1)
