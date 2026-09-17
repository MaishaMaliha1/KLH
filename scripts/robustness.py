from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from klh.evaluation import jaccard_topk, spearman_rank


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two saved part-score vectors")
    parser.add_argument("reference", type=Path)
    parser.add_argument("perturbed", type=Path)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    a = torch.tensor(json.loads(args.reference.read_text())["scores"], dtype=torch.float32)
    b = torch.tensor(json.loads(args.perturbed.read_text())["scores"], dtype=torch.float32)
    print(f"spearman={spearman_rank(a, b):.6f}")
    print(f"topk_jaccard={jaccard_topk(a, b, args.top_k):.6f}")


if __name__ == "__main__":
    main()
