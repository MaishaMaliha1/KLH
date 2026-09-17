import torch

from klh.evaluation import jaccard_topk, spearman_rank


def test_rank_metrics():
    a = torch.tensor([0.1, 0.7, 0.2, 0.9])
    b = torch.tensor([0.1, 0.6, 0.3, 0.8])
    assert spearman_rank(a, b) > 0.7
    assert jaccard_topk(a, b, 2) == 1.0
