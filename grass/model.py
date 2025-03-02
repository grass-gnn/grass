from typing import Final, Tuple, TypedDict
import math

import torch
import torch.nn as nn

from grass.nn import KaimingLinear, GatedLinearUnit
from grass.norm import FixedDistributionBatchNorm
from grass.gnn import GRASSModule, GRASS


class GRASSModelConfigDict(TypedDict):
    # data
    node_input_type: str
    node_input_dim: int
    edge_input_type: str
    edge_input_dim: int
    # dimensions
    num_layers: int
    dim: int
    output_dim: int
    hidden_dim: int
    # rewiring
    num_added_edges_per_node: int | None
    # encoding
    node_encoding_dim: int | None
    edge_encoding_dim: int | None
    add_degree_encoding: bool
    # attention
    alternate_edge_direction: bool
    attention_edge_removal_rate: float
    # normalization
    norm_type: str | None
    residual_scale: float
    # pooling
    pooling_type: str | None
    expected_num_nodes: float | None
    expected_num_existing_edges: float | None
    expected_num_added_edges: float | None
    # output
    task_head_type: str
    task_head_hidden_dim: int | None
    # performance
    enable_checkpointing: bool


class OGBMoleculeEmbedding(nn.Module):
    def __init__(
        self, embedding_type: str, embedding_dim: int, augment_dim: int = 0
    ) -> None:
        from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

        assert embedding_dim >= 1

        super().__init__()

        if embedding_type == "atom":
            feature_dims = get_atom_feature_dims()
        elif embedding_type == "bond":
            feature_dims = get_bond_feature_dims()
        else:
            raise ValueError(f"invalid OGB molecule embedding type: {embedding_type}")

        self.embedding_list: Final = torch.nn.ModuleList()
        for num_embeddings in feature_dims:
            num_embeddings += augment_dim
            embedding = torch.nn.Embedding(
                num_embeddings=num_embeddings, embedding_dim=embedding_dim
            )
            self.embedding_list.append(embedding)

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, gain: float = 1.0) -> None:
        bound = gain * math.sqrt(3.0 / len(self.embedding_list))
        for embedding in self.embedding_list:
            embedding.reset_parameters()
            nn.init.uniform_(embedding.weight, a=-bound, b=bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = 0
        for index, embedding in enumerate(self.embedding_list):
            output = output + embedding(x[..., index])

        return output


class NodeHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int | None) -> None:
        super().__init__()

        self.head: Final = None
        if hidden_dim is None:
            self.head = KaimingLinear(input_dim=input_dim, output_dim=output_dim)
        else:
            self.head = GatedLinearUnit(
                input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim
            )

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        self.head.reset_parameters()

    def forward(self, x: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class EdgeHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int | None) -> None:
        super().__init__()

        self.head: Final = None
        if hidden_dim is None:
            self.head = KaimingLinear(input_dim=input_dim, output_dim=output_dim)
        else:
            self.head = GatedLinearUnit(
                input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim
            )

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        self.head.reset_parameters()

    def forward(self, x: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        return self.head(edge_attr)


class PoolingHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int | None,
        has_added_edges: bool,
        expected_num_nodes: float | None = None,
        expected_num_existing_edges: float | None = None,
        expected_num_added_edges: float | None = None,
    ) -> None:
        assert input_dim >= 1
        assert output_dim >= 1
        assert hidden_dim is None or hidden_dim >= 1
        assert expected_num_nodes is None or expected_num_nodes > 0.0
        assert expected_num_existing_edges is None or expected_num_existing_edges > 0.0
        assert expected_num_added_edges is None or expected_num_added_edges > 0.0

        super().__init__()

        self.node_pooling_scale: Final = self._get_pooling_scale(
            expected_size=expected_num_nodes
        )
        self.existing_edge_pooling_scale: Final = self._get_pooling_scale(
            expected_size=expected_num_existing_edges
        )
        self.added_edge_pooling_scale: Final = self._get_pooling_scale(
            expected_size=expected_num_added_edges
        )

        if has_added_edges:
            head_input_dim = 3 * input_dim
        else:
            head_input_dim = 2 * input_dim

        self.head: Final = None
        if hidden_dim is None:
            self.head = KaimingLinear(input_dim=head_input_dim, output_dim=output_dim)
        else:
            self.head = GatedLinearUnit(
                input_dim=head_input_dim, output_dim=output_dim, hidden_dim=hidden_dim
            )

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        self.head.reset_parameters()

    @staticmethod
    def _get_pooling_scale(expected_size: float | None) -> float:
        if expected_size is None:
            return 1.0

        assert expected_size > 0.0
        return 1.0 / expected_size

    def forward(self, x: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        pooled_node_feature = x
        if self.node_pooling_scale is not None:
            pooled_node_feature = pooled_node_feature * self.node_pooling_scale

        pooled_existing_edge_feature = edge_attr[0]
        if self.existing_edge_pooling_scale is not None:
            pooled_existing_edge_feature = (
                pooled_existing_edge_feature * self.existing_edge_pooling_scale
            )

        if edge_attr.size(0) >= 2:
            pooled_added_edge_feature = edge_attr[1]
            if self.added_edge_pooling_scale is not None:
                pooled_added_edge_feature = (
                    pooled_added_edge_feature * self.added_edge_pooling_scale
                )
        else:
            pooled_added_edge_feature = pooled_existing_edge_feature.new_zeros()

        pooled_graph_feature = torch.cat(
            (
                pooled_node_feature,
                pooled_existing_edge_feature,
                pooled_added_edge_feature,
            ),
            dim=-1,
        )

        return self.head(pooled_graph_feature)


class GRASSModel(GRASSModule):
    def __init__(self, config: GRASSModelConfigDict) -> None:
        super().__init__()
        self.config: Final = config
        self.dim: Final = config["dim"]
        self.pooling_type: Final = config["pooling_type"]
        self.enable_checkpointing: Final = config["enable_checkpointing"]
        self.node_encoding_dim: Final = config["node_encoding_dim"]
        self.edge_encoding_dim: Final = config["edge_encoding_dim"]
        self.add_degree_encoding: Final = config["add_degree_encoding"]

        self.node_input_layer: Final
        self.node_input_norm: Final
        (self.node_input_layer, self.node_input_norm) = (
            self._get_node_input_layer_and_norm(config)
        )
        self.node_encoding_linear: Final
        self.node_encoding_norm: Final
        (self.node_encoding_linear, self.node_encoding_norm) = (
            self._get_node_encoding_linear_and_norm()
        )

        self.edge_input_layer: Final
        self.edge_input_norm: Final
        (self.edge_input_layer, self.edge_input_norm) = (
            self._get_edge_input_layer_and_norm(config)
        )
        self.edge_encoding_linear: Final
        self.edge_encoding_norm: Final
        (self.edge_encoding_linear, self.edge_encoding_norm) = (
            self._get_edge_encoding_linear_and_norm()
        )

        self.grass: Final = GRASS(
            num_layers=config["num_layers"],
            dim=self.dim,
            hidden_dim=config["hidden_dim"],
            output_pooling_type=self.pooling_type,
            norm_type=config["norm_type"],
            attention_edge_removal_rate=config["attention_edge_removal_rate"],
            alternate_edge_direction=config["alternate_edge_direction"],
            residual_scale=config["residual_scale"],
            enable_checkpointing=self.enable_checkpointing,
        )

        self.task_head: Final = self._get_task_head(config)

        self.reset_parameters()

    def _get_node_input_layer_and_norm(
        self, config: GRASSModelConfigDict
    ) -> Tuple[nn.Module, nn.Module]:
        return self._get_input_layer_and_norm(
            input_type=config["node_input_type"], input_dim=config["node_input_dim"]
        )

    def _get_edge_input_layer_and_norm(
        self, config: GRASSModelConfigDict
    ) -> Tuple[nn.Module, nn.Module]:
        # account for added self-loop and added edge indicators
        augmented_edge_input_dim = config["edge_input_dim"] + 1
        if config["num_added_edges_per_node"] is not None:
            augmented_edge_input_dim += 1

        return self._get_input_layer_and_norm(
            input_type=config["edge_input_type"],
            input_dim=augmented_edge_input_dim,
        )

    def _get_input_layer_and_norm(
        self, input_type: str, input_dim: int
    ) -> Tuple[nn.Module, nn.Module]:
        input_layer = None
        input_norm = None
        if input_type == "continuous":
            input_layer = KaimingLinear(input_dim=input_dim, output_dim=self.dim)
            input_norm = FixedDistributionBatchNorm(dim=input_dim)
        elif input_type == "index":
            input_layer = nn.Embedding(num_embeddings=input_dim, embedding_dim=self.dim)
        elif input_type == "one_hot":
            input_layer = nn.Linear(
                in_features=input_dim, out_features=self.dim, bias=False
            )
        elif input_type == "atom":
            input_layer = OGBMoleculeEmbedding(
                embedding_type="atom", embedding_dim=self.dim
            )
        elif input_type == "bond":
            input_layer = OGBMoleculeEmbedding(
                embedding_type="bond", embedding_dim=self.dim, augment_dim=2
            )
        else:
            raise ValueError(f"invalid input type: {input_type}")

        return (input_layer, input_norm)

    def _get_node_encoding_linear_and_norm(self) -> Tuple[nn.Module, nn.Module]:
        node_encoding_dim = self.node_encoding_dim
        if node_encoding_dim is None:
            if self.add_degree_encoding:
                node_encoding_dim = 0
            else:
                return None

        if self.add_degree_encoding:
            node_encoding_dim += 2

        return self._get_encoding_linear_and_norm(encoding_dim=node_encoding_dim)

    def _get_edge_encoding_linear_and_norm(self) -> Tuple[nn.Module, nn.Module]:
        if self.edge_encoding_dim is None:
            return (None, None)

        return self._get_encoding_linear_and_norm(encoding_dim=self.edge_encoding_dim)

    def _get_encoding_linear_and_norm(
        self, encoding_dim: int
    ) -> Tuple[nn.Module, nn.Module]:
        encoding_linear = KaimingLinear(input_dim=encoding_dim, output_dim=self.dim)
        encoding_norm = FixedDistributionBatchNorm(dim=encoding_dim)
        return (encoding_linear, encoding_norm)

    def _get_task_head(self, config: GRASSModelConfigDict) -> nn.Module:
        task_head_type = config["task_head_type"]
        if task_head_type == "node":
            assert self.pooling_type is None
            return NodeHead(
                input_dim=self.dim,
                output_dim=config["output_dim"],
                hidden_dim=config["task_head_hidden_dim"],
            )
        elif task_head_type == "edge":
            assert self.pooling_type is None
            return EdgeHead(
                input_dim=self.dim,
                output_dim=config["output_dim"],
                hidden_dim=config["task_head_hidden_dim"],
            )
        elif task_head_type == "pooling":
            assert self.pooling_type is not None
            has_added_edges = config["num_added_edges_per_node"] is not None
            return PoolingHead(
                input_dim=self.dim,
                output_dim=config["output_dim"],
                hidden_dim=config["task_head_hidden_dim"],
                has_added_edges=has_added_edges,
                expected_num_nodes=config["expected_num_nodes"],
                expected_num_existing_edges=config["expected_num_existing_edges"],
                expected_num_added_edges=config["expected_num_added_edges"],
            )
        else:
            raise ValueError(f"invalid task head type: {task_head_type}")

    @torch.no_grad()
    def reset_parameters(self) -> None:
        self._reset_input_layers()
        self._reset_input_norm_running_stats()
        self.grass.reset_parameters()
        self.task_head.reset_parameters()

    def _reset_input_layers(self) -> None:
        self._reset_layer(layer=self.node_input_layer)
        self._reset_layer(layer=self.node_encoding_linear)
        self._reset_layer(layer=self.edge_input_layer)
        self._reset_layer(layer=self.edge_encoding_linear)

    @staticmethod
    def _reset_layer(layer: nn.Module) -> None:
        if layer is None:
            pass
        elif isinstance(layer, KaimingLinear):
            layer.reset_parameters()
        elif isinstance(layer, (nn.Embedding, nn.Linear)):
            layer.reset_parameters()
            bound = math.sqrt(3)
            nn.init.uniform_(layer.weight, a=-bound, b=bound)
        elif isinstance(layer, OGBMoleculeEmbedding):
            layer.reset_parameters()
        else:
            raise RuntimeError(f"unexpected input layer type: {type(layer)}")

    def _reset_input_norm_running_stats(self) -> None:
        self._reset_norm_running_stats(self.node_input_norm)
        self._reset_norm_running_stats(self.node_encoding_norm)
        self._reset_norm_running_stats(self.edge_input_norm)
        self._reset_norm_running_stats(self.edge_encoding_norm)

    @staticmethod
    def _reset_norm_running_stats(norm: nn.Module) -> None:
        if norm is not None:
            norm.reset_running_stats()

    def toggle_track_and_use_batch_norm_stats(self, enable: bool) -> None:
        self.grass.toggle_track_and_use_batch_norm_stats(enable)

    def forward(
        self,
        batch: torch.Tensor,
        num_graphs: int,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        degree: torch.Tensor | None = None,
        node_encoding: torch.Tensor | None = None,
        edge_encoding: torch.Tensor | None = None,
        added_edge_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self._prepare_x(x=x, node_encoding=node_encoding, degree=degree)
        edge_attr = self._prepare_edge_attr(
            edge_attr=edge_attr, edge_encoding=edge_encoding
        )

        (x, edge_attr) = self.grass(
            batch=batch,
            num_graphs=num_graphs,
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            added_edge_mask=added_edge_mask,
        )

        return self.task_head(x=x, edge_attr=edge_attr)

    def _prepare_x(
        self,
        x: torch.Tensor,
        node_encoding: torch.Tensor | None,
        degree: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.node_input_norm is not None:
            x = self.node_input_norm(x)

        x = self.node_input_layer(x)

        if self.node_encoding_dim is not None:
            assert node_encoding is not None
            if self.add_degree_encoding:
                assert degree is not None
                node_encoding = torch.cat((node_encoding, degree), dim=-1)
        elif self.add_degree_encoding:
            assert degree is not None
            node_encoding = degree.to(dtype=x.dtype)
        else:
            return x

        node_encoding = self.node_encoding_norm(node_encoding)
        x = x + self.node_encoding_linear(node_encoding)

        return x

    def _prepare_edge_attr(
        self, edge_attr: torch.Tensor, edge_encoding: torch.Tensor | None
    ) -> torch.Tensor:
        if self.edge_input_norm is not None:
            edge_attr = self.edge_input_norm(edge_attr)

        edge_attr = self.edge_input_layer(edge_attr)

        if self.edge_encoding_dim is not None:
            assert edge_encoding is not None
            edge_encoding = self.edge_encoding_norm(edge_encoding)
            edge_attr = edge_attr + self.edge_encoding_linear(edge_encoding)

        return edge_attr
