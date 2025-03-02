from typing import Tuple
import os

import torch

import torch_geometric.data


def scatter_reduce(
    input: torch.Tensor,
    index: torch.Tensor,
    num_bins: int,
    reduction: str,
    broadcast_index: bool = False,
    gather: bool = False,
) -> torch.Tensor:
    if broadcast_index:
        scatter_index = index.view(-1, 1).expand_as(input)
        zeros = input.new_zeros(num_bins, input.size(-1))
    else:
        scatter_index = index
        zeros = input.new_zeros(num_bins)

    output = zeros.scatter_reduce(
        dim=0, index=scatter_index, src=input, reduce=reduction, include_self=False
    )

    if gather:
        output = output[index]

    return output


def get_device(
    cuda_device_name: str | None = "cuda",
    use_cuda_tf32_matmul: bool = True,
    use_cuda_expandable_segments: bool = True,
) -> torch.device:
    if torch.cuda.is_available():
        assert cuda_device_name is not None
        torch.backends.cuda.matmul.allow_tf32 = use_cuda_tf32_matmul
        if use_cuda_expandable_segments:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        return torch.device(cuda_device_name)

    return torch.device("cpu")


def get_graph_stats(
    dataset: torch_geometric.data.Dataset, num_samples: int
) -> Tuple[float, float, float]:
    total_num_nodes = 0
    total_num_existing_edges = 0

    for data in dataset:
        total_num_nodes += data.num_nodes
        total_num_existing_edges += (
            data.added_edge_mask.logical_not().long().sum().item()
        )

    expected_num_nodes = total_num_nodes / len(dataset)
    expected_num_existing_edges = total_num_existing_edges / len(dataset)

    total_num_edges = 0
    for _ in range(num_samples):
        for data in dataset:
            total_num_edges += data.num_edges

    expected_num_edges = total_num_edges / (num_samples * len(dataset))
    expected_num_added_edges = expected_num_edges - expected_num_existing_edges

    return (expected_num_nodes, expected_num_existing_edges, expected_num_added_edges)
