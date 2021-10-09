from collections import defaultdict
from typing import Tuple

import torch
import util


def load_glove_embeddings(dim: int = 300) -> dict[str, torch.Tensor]:
    embeddings = {}

    with open(f'glove.6B/glove.6B.{dim}d.txt', 'rt') as f:
        for line in util.mytqdm(f, desc='glove'):
            split = line.split(' ')
            token = split[0]
            tensor = torch.tensor([float(val) for val in split[1:]])
            embeddings[token] = tensor

    return embeddings


def load_embeddings_tensor_and_token_to_idx_dict(
        dim: int = 300
) -> Tuple[torch.Tensor, dict[str, int]]:

    embeddings = load_glove_embeddings(dim)

    token_to_idx: dict[str, int] = defaultdict(int)
    tensors: list[torch.Tensor] = []

    # add tensor for unk token
    tensors.append(torch.zeros((1, dim), dtype=torch.float))

    for i, (token, tensor) in enumerate(embeddings.items()):
        token_to_idx[token] = i + 1
        tensors.append(tensor.unsqueeze(0))

    return torch.cat(tensors), token_to_idx

