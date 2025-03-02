from typing import Final, Tuple
from abc import ABC, abstractmethod
import math

import torch

import torch_geometric
import torch_geometric.data
import torch_geometric.transforms
import torch_geometric.utils

import numpy as np
import scipy.sparse.linalg


class CastFloat(torch_geometric.transforms.BaseTransform):
    def __init__(self, dtype: torch.dtype):
        super().__init__()
        self.dtype: Final = dtype

    def __call__(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        for key, value in data:
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                value = value.to(dtype=self.dtype)
                data[key] = value

        return data


class AddZeroX(torch_geometric.transforms.BaseTransform):
    def __init__(self, dtype: torch.dtype = torch.long):
        super().__init__()
        self.dtype: Final = dtype

    def __call__(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        data.x = torch.zeros(1, dtype=self.dtype, device=data.edge_index.device).expand(
            data.num_nodes
        )
        return data


class AddZeroEdgeAttr(torch_geometric.transforms.BaseTransform):
    def __init__(self, dtype: torch.dtype = torch.long):
        super().__init__()
        self.dtype: Final = dtype

    def __call__(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        data.edge_attr = torch.zeros(
            1, dtype=self.dtype, device=data.edge_index.device
        ).expand(data.num_edges)
        return data


class AppendZerosToEdgeAttr(torch_geometric.transforms.BaseTransform):
    def __init__(self, num_zeros: int) -> None:
        assert num_zeros >= 1
        self.num_zeros: Final = num_zeros

    def __call__(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        edge_attr = data.edge_attr
        assert edge_attr.ndim <= 2
        if edge_attr.ndim == 1:
            edge_attr = edge_attr.unsqueeze(1)

        zeros = torch.zeros(1, dtype=edge_attr.dtype).expand(
            edge_attr.size(0), self.num_zeros
        )
        data.edge_attr = torch.cat((edge_attr, zeros), dim=1)
        return data


class AddDegree(torch_geometric.transforms.BaseTransform):
    def __init__(
        self,
        is_directed: bool,
        degree_attr_name: str = "degree",
        device: torch.device | None = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.is_directed: Final = is_directed
        self.degree_attr_name: Final = degree_attr_name
        self.device: Final = device

    def __call__(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        original_device = data.edge_index.device
        edge_index = data.edge_index.to(device=self.device)
        num_nodes = data.num_nodes

        (out_degree, in_degree) = self._get_degree(
            edge_index=edge_index, num_nodes=num_nodes, is_directed=self.is_directed
        )
        degree = torch.stack((out_degree, in_degree), dim=1)
        data[self.degree_attr_name] = degree.to(device=original_device)

        return data

    @staticmethod
    def _get_degree(
        edge_index: torch.Tensor, num_nodes: int, is_directed: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out_degree = edge_index[0].bincount(minlength=num_nodes)
        if is_directed:
            in_degree = edge_index[1].bincount(minlength=num_nodes)
        else:
            in_degree = out_degree

        return (out_degree, in_degree)

    @classmethod
    def get_degree_from_data(
        cls,
        data: torch_geometric.data.Data,
        degree_attr_name: str | None,
        is_directed: bool,
        edge_index: torch.Tensor | None = None,
        num_nodes: int | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if edge_index is None:
            edge_index = data.edge_index

        if num_nodes is None:
            num_nodes = data.num_nodes

        if degree_attr_name is not None:
            out_degree = data[degree_attr_name][:, 0]
            in_degree = data[degree_attr_name][:, 1]
        else:
            (out_degree, in_degree) = cls._get_degree(
                edge_index=edge_index, num_nodes=num_nodes, is_directed=is_directed
            )

        return (out_degree, in_degree)


class AddEdges(torch_geometric.transforms.BaseTransform, ABC):
    @abstractmethod
    def _get_added_edge_index(
        self, data: torch_geometric.data.Data
    ) -> torch.Tensor | None:
        raise NotImplementedError()

    def __init__(
        self,
        added_edge_attr: torch.Tensor | None,
        added_edge_mask_attr_name: str | None,
    ) -> None:
        super().__init__()
        self.added_edge_attr: Final = added_edge_attr
        self.mask_attr_name: Final = added_edge_mask_attr_name

    def __call__(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        added_edge_index = self._get_added_edge_index(data=data)
        data = self._add_edge_index(data=data, added_edge_index=added_edge_index)
        data = self._add_edge_attr(data=data, added_edge_index=added_edge_index)
        data = self._add_edge_mask(data=data, added_edge_index=added_edge_index)
        return data

    def _add_edge_index(
        self, data: torch_geometric.data.Data, added_edge_index: torch.Tensor | None
    ) -> torch_geometric.data.Data:
        if added_edge_index is None:
            return data

        edge_index = data.edge_index
        original_device = edge_index.device

        data.edge_index = torch.cat((edge_index, added_edge_index), dim=1).to(
            device=original_device
        )

        return data

    def _add_edge_attr(
        self, data: torch_geometric.data.Data, added_edge_index: torch.Tensor | None
    ) -> torch_geometric.data.Data:
        if self.added_edge_attr is None or added_edge_index is None:
            return data

        existing_edge_attr = data.edge_attr
        original_device = existing_edge_attr.device

        added_edge_attr_size = list(existing_edge_attr.shape)
        added_edge_attr_size[0] = added_edge_index.size(1)
        added_edge_attr = self.added_edge_attr.expand(added_edge_attr_size).to(
            device=original_device
        )
        data.edge_attr = torch.cat((existing_edge_attr, added_edge_attr))

        return data

    def _add_edge_mask(
        self, data: torch_geometric.data.Data, added_edge_index: torch.Tensor | None
    ) -> torch.Tensor:
        if self.mask_attr_name is None:
            return data

        edge_index = data.edge_index
        original_device = edge_index.device

        if added_edge_index is None:
            num_existing_edges = edge_index.size(1)
            data[self.mask_attr_name] = torch.zeros(
                num_existing_edges, dtype=torch.bool, device=original_device
            )
        else:
            num_added_edges = added_edge_index.size(1)
            num_existing_edges = edge_index.size(1) - num_added_edges
            original_edge_mask = torch.zeros(
                num_existing_edges, dtype=torch.bool, device=original_device
            )
            added_edge_mask = torch.ones(
                num_added_edges, dtype=torch.bool, device=original_device
            )
            data[self.mask_attr_name] = torch.cat((original_edge_mask, added_edge_mask))

        return data

    @staticmethod
    def _remove_self_loops(
        added_head: torch.Tensor, added_tail: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        is_not_self_loop = added_head.eq(added_tail).logical_not_()
        added_head = added_head.masked_select(is_not_self_loop)
        added_tail = added_tail.masked_select(is_not_self_loop)
        return (added_head, added_tail)


class AddRandomEdges(AddEdges):
    def __init__(
        self,
        num_added_edges_per_node: int,
        sample_regular_graph: bool = True,
        add_directed_edges: bool = False,
        added_edge_attr: torch.Tensor | None = None,
        remove_self_loops: bool = True,
        remove_multiple_edges: bool = True,
        added_edge_mask_attr_name: str | None = None,
        device: torch.device | None = torch.device("cpu"),
    ) -> None:
        assert num_added_edges_per_node >= 0

        super().__init__(
            added_edge_attr=added_edge_attr,
            added_edge_mask_attr_name=added_edge_mask_attr_name,
        )
        self.num_added_edges_per_node: Final = num_added_edges_per_node
        self.sample_regular_graph: Final = sample_regular_graph
        self.add_directed_edges: Final = add_directed_edges
        self.remove_self_loops: Final = remove_self_loops
        self.remove_multiple_edges: Final = remove_multiple_edges
        self.device: Final = device

    def _get_added_edge_index(
        self, data: torch_geometric.data.Data
    ) -> torch.Tensor | None:
        num_nodes = data.num_nodes
        if num_nodes <= 1 or self.num_added_edges_per_node == 0:
            return None

        edge_index = data.edge_index.to(device=self.device)
        num_added_edges_per_node = min(num_nodes - 1, self.num_added_edges_per_node)
        num_added_edges = num_added_edges_per_node * num_nodes
        if self.sample_regular_graph:
            added_head = torch.arange(
                num_added_edges, dtype=edge_index.dtype, device=self.device
            ).remainder_(num_nodes)

            random_permutations = []
            for _ in range(num_added_edges_per_node):
                random_permutations.append(
                    torch.randperm(
                        num_nodes, dtype=edge_index.dtype, device=self.device
                    )
                )

            added_tail = torch.cat(random_permutations)
        else:
            random_index_size = (2 * num_added_edges,)
            random_index = torch.randint(
                low=0,
                high=num_nodes,
                size=random_index_size,
                dtype=edge_index.dtype,
                device=self.device,
            )
            (added_head, added_tail) = random_index.tensor_split(sections=2)

        if self.remove_self_loops:
            (added_head, added_tail) = self._remove_self_loops(
                added_head=added_head, added_tail=added_tail
            )

        added_edge_index = torch.stack((added_head, added_tail))

        if self.remove_multiple_edges:
            added_edge_index = added_edge_index.unique(sorted=False, dim=1)

        if not self.add_directed_edges:
            added_edge_index = torch.cat(
                (added_edge_index, added_edge_index.flip(0)), dim=1
            )

        return added_edge_index


class AddAllEdges(AddEdges):
    def __init__(
        self,
        added_edge_attr: torch.Tensor | None = None,
        remove_self_loops: bool = False,
        added_edge_mask_attr_name: str | None = None,
        device: torch.device | None = torch.device("cpu"),
    ) -> None:
        super().__init__(
            added_edge_attr=added_edge_attr,
            added_edge_mask_attr_name=added_edge_mask_attr_name,
        )
        self.remove_self_loops: Final = remove_self_loops
        self.device: Final = device

    def _get_added_edge_index(
        self, data: torch_geometric.data.Data
    ) -> torch.Tensor | None:
        num_nodes = data.num_nodes
        num_edges = num_nodes * num_nodes

        index = torch.arange(num_edges, dtype=data.edge_index.dtype, device=self.device)
        added_head = index.floor_divide(num_nodes)
        added_tail = index.remainder_(num_nodes)

        if self.remove_self_loops:
            (added_head, added_tail) = self._remove_self_loops(
                added_head=added_head, added_tail=added_tail
            )

        return torch.stack((added_head, added_tail))


class EncodingGenerator(ABC):
    @abstractmethod
    def to_node_encoding(self, device: torch.device | None = None) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def to_edge_encoding(
        self, edge_index: torch.Tensor, device: torch.device | None = None
    ) -> torch.Tensor:
        raise NotImplementedError()


class GenerateEncoding(torch_geometric.transforms.BaseTransform):
    def __init__(
        self,
        mode: str,
        encoding_attr_name: str,
        generator_attr_name: str,
        remove_generator: bool = False,
        device: torch.device | None = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.mode: Final = mode
        self.encoding_attr_name: Final = encoding_attr_name
        self.generator_attr_name: Final = generator_attr_name
        self.remove_generator: Final = remove_generator
        self.device: Final = device

    def __call__(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        generator = data[self.generator_attr_name]
        if self.mode == "node":
            encoding = generator.to_node_encoding(device=self.device)
        elif self.mode == "edge":
            edge_index = data.edge_index.to(device=self.device)
            encoding = generator.to_edge_encoding(edge_index, device=self.device)
        else:
            raise ValueError(f"invalid mode: {self.mode}")

        data[self.encoding_attr_name] = encoding.to(device=data.edge_index.device)

        if self.remove_generator:
            del data[self.generator_attr_name]

        return data


class RandomWalkProbabilities(EncodingGenerator):
    def __init__(
        self,
        random_walk_probabilities: torch.Tensor,
        include_reverse_random_walk: bool = True,
    ) -> None:
        self.random_walk_probabilities: Final = random_walk_probabilities
        self.include_reverse_random_walk: Final = include_reverse_random_walk

    def to_node_encoding(self, device: torch.device | None = None) -> torch.Tensor:
        random_walk_probabilities = self.random_walk_probabilities.to(device=device)
        return random_walk_probabilities.diagonal(dim1=0, dim2=1).transpose(0, 1)

    def to_edge_encoding(
        self, edge_index: torch.Tensor, device: torch.device | None = None
    ) -> torch.Tensor:
        random_walk_probabilities = self.random_walk_probabilities.to(device=device)
        (head_index, tail_index) = edge_index
        random_walk_probability = random_walk_probabilities[head_index, tail_index]
        if self.include_reverse_random_walk:
            reverse_random_walk_probability = random_walk_probabilities[
                tail_index, head_index
            ]
            random_walk_probability = torch.cat(
                (random_walk_probability, reverse_random_walk_probability), dim=-1
            )

        return random_walk_probability


class AddUndirectedRandomWalkProbabilities(torch_geometric.transforms.BaseTransform):
    def __init__(
        self,
        num_random_walk_steps: int,
        use_symmetric_normalization: bool = False,
        include_reverse_random_walk: bool = False,
        degree_attr_name: str | None = None,
        generator_attr_name: str = "random_walk_probabilities",
        compute_dtype: torch.dtype = torch.float32,
        output_dtype: torch.dtype = torch.float16,
        use_sparse_operations: bool = True,
        device: torch.device | None = torch.device("cpu"),
    ) -> None:
        assert num_random_walk_steps >= 1
        assert not (include_reverse_random_walk and use_symmetric_normalization)
        assert compute_dtype.is_floating_point
        assert output_dtype.is_floating_point

        super().__init__()
        self.random_walk_len: Final = num_random_walk_steps
        self.use_symmetric_normalization: Final = use_symmetric_normalization
        self.include_reverse_random_walk: Final = include_reverse_random_walk
        self.degree_attr_name: Final = degree_attr_name
        self.generator_attr_name: Final = generator_attr_name
        self.compute_dtype: Final = compute_dtype
        self.output_dtype: Final = output_dtype
        self.use_sparse_operations: Final = use_sparse_operations
        self.device: Final = device

    def __call__(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        original_device = data.edge_index.device
        edge_index = data.edge_index.to(device=self.device)
        num_nodes = data.num_nodes

        (degree, _) = AddDegree.get_degree_from_data(
            data=data,
            degree_attr_name=self.degree_attr_name,
            is_directed=True,
            edge_index=edge_index,
            num_nodes=num_nodes,
        )
        random_walk_probabilities = self._get_random_walk_probabilities(
            edge_index=edge_index, degree=degree, num_nodes=num_nodes
        ).to(dtype=self.output_dtype, device=original_device)

        data[self.generator_attr_name] = RandomWalkProbabilities(
            random_walk_probabilities=random_walk_probabilities,
            include_reverse_random_walk=self.include_reverse_random_walk,
        )

        return data

    def _get_random_walk_matrix_ref(
        self, edge_index: torch.Tensor, num_nodes: int
    ) -> torch.Tensor:
        adjacency_matrix = (
            torch_geometric.utils.to_dense_adj(edge_index, max_num_nodes=num_nodes)
            .squeeze(0)
            .to(dtype=self.compute_dtype)
        )
        degree = adjacency_matrix.sum(dim=-1).clamp(min=1).view(-1, 1)
        transition_matrix = adjacency_matrix / degree

        random_walk_matrix_list = [transition_matrix]
        for _ in range(self.random_walk_len - 1):
            next_random_walk_matrix = torch.mm(
                random_walk_matrix_list[-1], transition_matrix
            )
            random_walk_matrix_list.append(next_random_walk_matrix)

        random_walk_matrix = torch.stack(random_walk_matrix_list, dim=-1)
        assert random_walk_matrix.shape == (num_nodes, num_nodes, self.random_walk_len)

        return random_walk_matrix

    def _get_random_walk_probabilities(
        self, edge_index: torch.Tensor, degree: torch.Tensor, num_nodes: int
    ) -> torch.Tensor:
        sparse_transition_matrix = self._get_sparse_transition_matrix(
            edge_index=edge_index,
            degree=degree,
            num_nodes=num_nodes,
            use_symmetric_normalization=self.use_symmetric_normalization,
        )
        dense_transition_matrix = sparse_transition_matrix.to_dense()
        if self.use_sparse_operations:
            transition_matrix = sparse_transition_matrix.to_sparse_csr()
        else:
            transition_matrix = dense_transition_matrix

        random_walk_probability_list = [dense_transition_matrix]
        for _ in range(self.random_walk_len - 1):
            next_random_walk_probability = torch.mm(
                transition_matrix, random_walk_probability_list[-1]
            )
            random_walk_probability_list.append(next_random_walk_probability)

        random_walk_probabilities = torch.stack(random_walk_probability_list, dim=-1)
        assert random_walk_probabilities.shape == (
            num_nodes,
            num_nodes,
            self.random_walk_len,
        )

        return random_walk_probabilities

    def _get_sparse_transition_matrix(
        self,
        edge_index: torch.Tensor,
        degree: torch.Tensor,
        num_nodes: int,
        use_symmetric_normalization: bool,
    ) -> torch.Tensor:
        degree = degree.to(dtype=self.compute_dtype, device=self.device)
        if use_symmetric_normalization:
            degree_rsqrt = degree.clamp(min=1).rsqrt()
            edge_weight = degree_rsqrt[edge_index[0]] * degree_rsqrt[edge_index[1]]
        else:
            edge_weight = degree.clamp(min=1).reciprocal()[edge_index[0]]

        transition_matrix_size = (num_nodes, num_nodes)
        return torch.sparse_coo_tensor(
            indices=edge_index, values=edge_weight, size=transition_matrix_size
        )


class RandomWalkMatrixEigenpairs(EncodingGenerator):
    def __init__(
        self,
        eigenvalues: torch.Tensor,
        eigenvectors: torch.Tensor,
        num_random_walk_steps: int,
        dtype: torch.dtype | None,
        device: torch.device | None = None,
    ) -> None:
        assert num_random_walk_steps >= 1
        if dtype is None:
            dtype = torch.float32

        super().__init__()
        self.eigenvectors: Final = eigenvectors.to(dtype=dtype, device=device)
        self.eigenvalue_powers: Final = self._get_eigenvalue_powers(
            eigenvalues=eigenvalues, num_random_walk_steps=num_random_walk_steps
        ).to(dtype=dtype, device=device)

    @staticmethod
    def _get_eigenvalue_powers(
        eigenvalues: torch.Tensor, num_random_walk_steps: int | None
    ) -> torch.Tensor:
        if num_random_walk_steps is None:
            num_random_walk_steps = eigenvalues.size(-1)

        powers = [eigenvalues]
        for _ in range(num_random_walk_steps - 1):
            powers.append(powers[-1] * eigenvalues)

        return torch.stack(powers, dim=-1)

    def to_node_encoding(self, device: torch.device | None = None) -> torch.Tensor:
        eigenvectors = self.eigenvectors.to(device=device)
        eigenvector_product = eigenvectors.square()
        return self._get_encoding(
            eigenvector_product=eigenvector_product, device=device
        )

    def to_edge_encoding(
        self, edge_index: torch.Tensor, device: torch.device | None = None
    ) -> torch.Tensor:
        eigenvectors = self.eigenvectors.to(device=device)
        eigenvector_product = (
            eigenvectors[..., edge_index[0], :] * eigenvectors[..., edge_index[1], :]
        )
        return self._get_encoding(
            eigenvector_product=eigenvector_product, device=device
        )

    def _get_encoding(
        self, eigenvector_product: torch.Tensor, device: torch.device | None
    ) -> torch.Tensor:
        eigenvalue_powers = self.eigenvalue_powers.to(device=device)
        return torch.matmul(eigenvector_product, eigenvalue_powers)


class RandomWalkMatrixComplexEigenpairs(RandomWalkMatrixEigenpairs):
    def __init__(
        self,
        forward_eigenvalues: torch.Tensor,
        forward_eigenvectors: torch.Tensor,
        backward_eigenvalues: torch.Tensor,
        backward_eigenvectors: torch.Tensor,
        num_random_walk_steps: int,
        dtype: torch.dtype | None,
        device: torch.device | None = None,
    ) -> None:
        if dtype is None:
            dtype = torch.complex64

        eigenvalues = torch.stack((forward_eigenvalues, backward_eigenvalues))
        eigenvectors = torch.stack((forward_eigenvectors, backward_eigenvectors))
        super().__init__(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            num_random_walk_steps=num_random_walk_steps,
            dtype=dtype,
            device=device,
        )

    def to_node_encoding(self, device: torch.device | None = None) -> torch.Tensor:
        encoding = super().to_node_encoding(device=device)
        return encoding.real.view(-1, 2 * encoding.size(-1))

    def to_edge_encoding(
        self, edge_index: torch.Tensor, device: torch.device | None = None
    ) -> torch.Tensor:
        encoding = super().to_edge_encoding(edge_index=edge_index, device=device)
        return encoding.real.view(-1, 2 * encoding.size(-1))


class AlternatingRandomWalkMatrixEigenpairs(RandomWalkMatrixEigenpairs):
    def __init__(
        self,
        singular_values: torch.Tensor,
        left_singular_vectors: torch.Tensor,
        right_singular_vectors: torch.Tensor,
        num_random_walk_steps: int,
        dtype: torch.dtype | None,
        device: torch.device | None = None,
    ) -> None:
        eigenvalues = singular_values.square().unsqueeze(0)
        eigenvectors = torch.stack((left_singular_vectors, right_singular_vectors))
        super().__init__(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            num_random_walk_steps=num_random_walk_steps,
            dtype=dtype,
            device=device,
        )

    def to_node_encoding(self, device: torch.device | None = None) -> torch.Tensor:
        encoding = super().to_node_encoding(device=device)
        return encoding.view(-1, 2 * encoding.size(-1))

    def to_edge_encoding(
        self, edge_index: torch.Tensor, device: torch.device | None = None
    ) -> torch.Tensor:
        encoding = super().to_edge_encoding(edge_index=edge_index, device=device)
        return encoding.view(-1, 2 * encoding.size(-1))


class AddDecomposedRandomWalk(torch_geometric.transforms.BaseTransform):
    def __init__(
        self,
        mode: str,
        max_num_components: int,
        num_random_walk_steps: int | None = None,
        add_global_node: bool = False,
        degree_attr_name: str | None = None,
        generator_attr_name: str = "decomposed_random_walk",
        compute_dtype: torch.dtype = torch.float32,
        output_dtype: torch.dtype | None = None,
        eigensolver_tolerance: float = 1e-7,
        eigensolver_max_num_iters: int | None = None,
    ) -> None:
        assert mode in ["undirected", "undirectified", "complex", "alternating"]
        assert max_num_components >= 1
        assert num_random_walk_steps is None or num_random_walk_steps >= 1
        assert compute_dtype.is_floating_point
        assert output_dtype is None or (
            output_dtype.is_complex
            if mode == "complex"
            else output_dtype.is_floating_point
        )
        assert eigensolver_tolerance >= 0.0
        assert eigensolver_max_num_iters is None or eigensolver_max_num_iters >= 1

        super().__init__()
        self.mode: Final = mode
        self.max_num_components: Final = max_num_components
        self.num_random_walk_steps: Final = num_random_walk_steps
        self.add_global_node: Final = add_global_node
        self.degree_attr_name: Final = degree_attr_name
        self.generator_attr_name: Final = generator_attr_name
        self.compute_dtype: Final = compute_dtype
        self.output_dtype: Final = output_dtype
        self.eigensolver_tolerance: Final = eigensolver_tolerance
        self.eigensolver_max_num_iters: Final = eigensolver_max_num_iters
        self.device: Final = torch.device("cpu")

        if self.num_random_walk_steps is None:
            self.num_random_walk_steps = self.max_num_components

    def __call__(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        original_device = data.edge_index.device
        edge_index = data.edge_index.to(device=self.device)
        num_nodes = data.num_nodes
        is_directed = self.mode != "undirected"
        (out_degree, in_degree) = AddDegree.get_degree_from_data(
            data=data,
            degree_attr_name=self.degree_attr_name,
            edge_index=edge_index,
            num_nodes=num_nodes,
            is_directed=is_directed,
        )
        if self.add_global_node:
            (edge_index, out_degree, in_degree, num_nodes) = self._add_global_node(
                edge_index=edge_index,
                out_degree=out_degree,
                in_degree=in_degree,
                num_nodes=num_nodes,
            )

        if self.mode == "undirected":
            encoding = self._get_undirected_eigenpairs(
                edge_index=edge_index,
                degree=out_degree,
                num_nodes=num_nodes,
                original_device=original_device,
            )
        elif self.mode == "undirectified":
            encoding = self._get_undirectified_eigenpairs(
                edge_index=edge_index,
                out_degree=out_degree,
                in_degree=in_degree,
                num_nodes=num_nodes,
                original_device=original_device,
            )
        elif self.mode == "complex":
            encoding = self._get_complex_eigenpairs(
                edge_index=edge_index,
                out_degree=out_degree,
                in_degree=in_degree,
                num_nodes=num_nodes,
                original_device=original_device,
            )
        elif self.mode == "alternating":
            encoding = self._get_alternating_eigenpairs(
                edge_index=edge_index,
                out_degree=out_degree,
                in_degree=in_degree,
                num_nodes=num_nodes,
                original_device=original_device,
            )
        else:
            raise ValueError(f"invalid mode: {self.mode}")

        data[self.generator_attr_name] = encoding
        return data

    @staticmethod
    def _add_global_node(
        edge_index: torch.Tensor,
        out_degree: torch.Tensor,
        in_degree: torch.Tensor,
        num_nodes: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        num_nodes_tensor = torch.tensor([num_nodes], dtype=edge_index.dtype)
        global_node_edge_index = torch.stack(
            (
                num_nodes_tensor.expand(num_nodes),
                torch.arange(num_nodes, dtype=edge_index.dtype),
            )
        )
        edge_index = torch.cat(
            (edge_index, global_node_edge_index, global_node_edge_index.flip(0)), dim=1
        )
        out_degree = torch.cat((out_degree + 1, num_nodes_tensor))
        in_degree = torch.cat((in_degree + 1, num_nodes_tensor))
        num_nodes += 1
        return (edge_index, out_degree, in_degree, num_nodes)

    def _get_undirected_eigenpairs(
        self,
        edge_index: torch.Tensor,
        degree: torch.Tensor,
        num_nodes: int,
        original_device: torch.device,
    ) -> RandomWalkMatrixEigenpairs:
        # decompose the symmetrically normalized adjacency matrix
        degree = degree.to(dtype=self.compute_dtype, device=self.device)
        degree_rsqrt = degree.clamp(min=1).rsqrt()
        edge_weight = degree_rsqrt[edge_index[0]] * degree_rsqrt[edge_index[1]]
        (eigenvalues, eigenvectors) = self._get_eigenpairs(
            edge_index=edge_index,
            edge_weight=edge_weight,
            num_nodes=num_nodes,
            is_symmetric=True,
        )

        return RandomWalkMatrixEigenpairs(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            num_random_walk_steps=self.num_random_walk_steps,
            dtype=self.output_dtype,
            device=original_device,
        )

    def _get_undirectified_eigenpairs(
        self,
        edge_index: torch.Tensor,
        out_degree: torch.Tensor,
        in_degree: torch.Tensor,
        num_nodes: int,
        original_device: torch.device,
    ) -> RandomWalkMatrixEigenpairs:
        edge_index = torch.cat((edge_index, edge_index.flip(0)), dim=1)
        degree = out_degree + in_degree
        return self._get_undirected_eigenpairs(
            edge_index=edge_index,
            degree=degree,
            num_nodes=num_nodes,
            original_device=original_device,
        )

    def _get_complex_eigenpairs(
        self,
        edge_index: torch.Tensor,
        out_degree: torch.Tensor,
        in_degree: torch.Tensor,
        num_nodes: int,
        original_device: torch.device,
    ) -> RandomWalkMatrixComplexEigenpairs:
        # decompose the forward random walk transition matrix
        out_degree = out_degree.to(dtype=self.compute_dtype, device=self.device)
        forward_edge_weight = out_degree.clamp(min=1).reciprocal()[edge_index[0]]
        (forward_eigenvalues, forward_eigenvectors) = self._get_eigenpairs(
            edge_index=edge_index,
            edge_weight=forward_edge_weight,
            num_nodes=num_nodes,
            is_symmetric=False,
        )

        # decompose the backward random walk transition matrix
        in_degree = in_degree.to(dtype=self.compute_dtype, device=self.device)
        backward_edge_weight = in_degree.clamp(min=1).reciprocal()[edge_index[1]]
        flipped_edge_index = edge_index.flip(0)
        (backward_eigenvalues, backward_eigenvectors) = self._get_eigenpairs(
            edge_index=flipped_edge_index,
            edge_weight=backward_edge_weight,
            num_nodes=num_nodes,
            is_symmetric=False,
        )

        return RandomWalkMatrixComplexEigenpairs(
            forward_eigenvalues=forward_eigenvalues,
            forward_eigenvectors=forward_eigenvectors,
            backward_eigenvalues=backward_eigenvalues,
            backward_eigenvectors=backward_eigenvectors,
            num_random_walk_steps=self.num_random_walk_steps,
            dtype=self.output_dtype,
            device=original_device,
        )

    def _get_alternating_eigenpairs(
        self,
        edge_index: torch.Tensor,
        out_degree: torch.Tensor,
        in_degree: torch.Tensor,
        num_nodes: int,
        original_device: torch.device,
    ) -> AlternatingRandomWalkMatrixEigenpairs:
        # decompose the alternating direction random walk transition matrix
        out_degree = out_degree.to(dtype=self.compute_dtype, device=self.device)
        in_degree = in_degree.to(dtype=self.compute_dtype, device=self.device)
        out_degree_rsqrt = out_degree.clamp(min=1).rsqrt()
        in_degree_rsqrt = in_degree.clamp(min=1).rsqrt()
        edge_weight = out_degree_rsqrt[edge_index[0]] * in_degree_rsqrt[edge_index[1]]
        (singular_values, left_singular_vectors, right_singular_vectors) = (
            self._get_svd(
                edge_index=edge_index, edge_weight=edge_weight, num_nodes=num_nodes
            )
        )

        return AlternatingRandomWalkMatrixEigenpairs(
            singular_values=singular_values,
            left_singular_vectors=left_singular_vectors,
            right_singular_vectors=right_singular_vectors,
            num_random_walk_steps=self.num_random_walk_steps,
            dtype=self.output_dtype,
            device=original_device,
        )

    def _get_eigenpairs(
        self,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        num_nodes: int,
        is_symmetric: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        transition_matrix = self._to_scipy_sparse_matrix(
            edge_index=edge_index, edge_weight=edge_weight, num_nodes=num_nodes
        )
        if is_symmetric:
            num_components = min(num_nodes - 1, self.max_num_components)
            (eigenvalues, eigenvectors) = scipy.sparse.linalg.eigsh(
                A=transition_matrix,
                k=num_components,
                which="LM",
                tol=self.eigensolver_tolerance,
            )
        else:
            num_components = min(num_nodes - 2, self.max_num_components)
            (eigenvalues, eigenvectors) = scipy.sparse.linalg.eigs(
                A=transition_matrix,
                k=num_components,
                which="LM",
                tol=self.eigensolver_tolerance,
            )

        eigenvalues = torch.from_numpy(eigenvalues)
        assert eigenvalues.shape == (num_components,)
        eigenvectors = torch.from_numpy(eigenvectors)
        assert eigenvectors.shape == (num_nodes, num_components)
        if self.add_global_node:
            eigenvectors = eigenvectors[:-1, :]

        return (eigenvalues, eigenvectors)

    def _get_svd(
        self, edge_index: torch.Tensor, edge_weight: torch.Tensor, num_nodes: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        transition_matrix = self._to_scipy_sparse_matrix(
            edge_index=edge_index, edge_weight=edge_weight, num_nodes=num_nodes
        )
        num_components = min(num_nodes - 1, self.max_num_components)
        eigensolver_tolerance = math.sqrt(self.eigensolver_tolerance)
        (left_singular_vectors, singular_values, right_singular_vectors) = (
            scipy.sparse.linalg.svds(
                A=transition_matrix,
                k=num_components,
                which="LM",
                tol=eigensolver_tolerance,  # squared internally
                maxiter=self.eigensolver_max_num_iters,
            )
        )

        singular_values = torch.from_numpy(np.ascontiguousarray(singular_values))
        assert singular_values.shape == (num_components,)
        left_singular_vectors = torch.from_numpy(
            np.ascontiguousarray(left_singular_vectors)
        )
        assert left_singular_vectors.shape == (num_nodes, num_components)
        right_singular_vectors = torch.from_numpy(
            np.ascontiguousarray(right_singular_vectors.transpose())
        )
        assert right_singular_vectors.shape == (
            num_nodes,
            num_components,
        )

        return (singular_values, left_singular_vectors, right_singular_vectors)

    @staticmethod
    def _to_scipy_sparse_matrix(
        edge_index: torch.Tensor, edge_weight: torch.Tensor, num_nodes: int
    ) -> scipy.sparse.csr_matrix:
        return torch_geometric.utils.to_scipy_sparse_matrix(
            edge_index=edge_index, edge_attr=edge_weight, num_nodes=num_nodes
        ).tocsr()
