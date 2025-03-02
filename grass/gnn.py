from typing import Final, Tuple
from abc import ABC, abstractmethod
import math

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint

from grass.nn import KaimingLinear
from grass.norm import get_norm_module
from grass.utils import scatter_reduce


class GRASSModule(nn.Module, ABC):
    @abstractmethod
    def toggle_track_and_use_batch_norm_stats(self, enable: bool) -> None:
        raise NotImplementedError()

    def __init__(self) -> None:
        nn.Module.__init__(self)


class GRASSLayer(GRASSModule):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        norm_type: str | None = "batch",
        residual_scale: float = 1.0,
        attention_edge_removal_rate: float = 0.0,
        eps: float = 1e-7,
    ) -> None:
        assert dim >= 1
        assert hidden_dim >= 1
        assert residual_scale > 0.0
        assert 0.0 <= attention_edge_removal_rate and attention_edge_removal_rate < 1.0
        assert 0.0 < eps and eps <= 1.0

        super().__init__()
        self.dim: Final = dim
        self.norm_type: Final = norm_type
        self.residual_scale: Final = residual_scale
        self.attention_edge_removal_rate: Final = attention_edge_removal_rate
        self.eps: Final = eps

        self.attn_weight_eps: Final = math.sqrt(eps)

        node_input_linear_output_dim = 4 * hidden_dim
        self.node_input_linear: Final = KaimingLinear(
            input_dim=dim, output_dim=node_input_linear_output_dim, bias=False
        )
        edge_input_linear_output_dim = 3 * hidden_dim
        self.edge_input_linear: Final = KaimingLinear(
            input_dim=dim, output_dim=edge_input_linear_output_dim, bias=False
        )

        self.node_message_bias: Final = nn.Parameter(torch.zeros(hidden_dim))
        self.edge_message_bias: Final = nn.Parameter(torch.zeros(hidden_dim))

        self.node_norm: Final = None
        self.edge_norm: Final = None
        self.output_linear_needs_bias: Final = True
        self.track_and_use_batch_norm_stats = False
        if norm_type is not None:
            self.node_norm = get_norm_module(norm_type=norm_type, dim=dim, eps=eps)
            self.edge_norm = get_norm_module(norm_type=norm_type, dim=dim, eps=eps)
            self.output_linear_needs_bias = not self.node_norm.is_shift_invariant

        self.node_output_linear: Final = KaimingLinear(
            input_dim=hidden_dim, output_dim=dim, bias=self.output_linear_needs_bias
        )
        self.edge_output_linear: Final = KaimingLinear(
            input_dim=hidden_dim, output_dim=dim, bias=self.output_linear_needs_bias
        )

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        self._reset_input_linear()
        self._reset_message_biases()
        self._reset_output_linear()

        if self.norm_type is not None and "batch" in self.norm_type:
            self.toggle_track_and_use_batch_norm_stats(enable=False)

    def _reset_input_linear(self) -> None:
        self.node_input_linear.reset_parameters()
        self.edge_input_linear.reset_parameters()

    def _reset_message_biases(self) -> None:
        nn.init.zeros_(self.node_message_bias)
        nn.init.zeros_(self.edge_message_bias)

    def _reset_output_linear(self) -> None:
        self.node_output_linear.reset_parameters()
        self.edge_output_linear.reset_parameters()

    def toggle_track_and_use_batch_norm_stats(self, enable: bool) -> None:
        self.node_norm.toggle_track_and_use_running_stats(enable)
        self.edge_norm.toggle_track_and_use_running_stats(enable)
        self.track_and_use_batch_norm_stats = enable

    def forward(
        self,
        x: torch.Tensor,
        head_node_index: torch.Tensor,
        tail_node_index: torch.Tensor,
        edge_attr: torch.Tensor,
        node_batch: torch.Tensor,
        edge_batch: torch.Tensor,
        num_nodes: int,
        num_graphs: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # prepare node messages
        (
            node_to_self_message,
            node_to_neighbor_message,
            node_to_edge_message,
        ) = self._get_node_messages(
            x=x,
            head_node_index=head_node_index,
            tail_node_index=tail_node_index,
        )

        # prepare edge messages
        (
            edge_to_self_message,
            edge_to_tail_node_message,
            edge_attn_message,
        ) = self._get_edge_messages(edge_attr=edge_attr)

        # compute attention weights
        edge_attn_weight = self._get_edge_attn_weight(
            edge_attn_message=edge_attn_message,
            tail_node_index=tail_node_index,
            num_nodes=num_nodes,
        )

        # aggregate node messages
        node_received_message = self._get_node_received_message(
            node_to_neighbor_message=node_to_neighbor_message,
            edge_to_tail_node_message=edge_to_tail_node_message,
            edge_attn_weight=edge_attn_weight,
            head_node_index=head_node_index,
            tail_node_index=tail_node_index,
            num_nodes=num_nodes,
        )
        x_out = self._get_x_out(
            x=x,
            node_received_message=node_received_message,
            node_to_self_message=node_to_self_message,
            node_batch=node_batch,
            num_graphs=num_graphs,
        )

        # aggregate edge messages
        edge_attr_out = self._get_edge_attr_out(
            edge_attr=edge_attr,
            node_to_edge_message=node_to_edge_message,
            edge_to_self_message=edge_to_self_message,
            edge_batch=edge_batch,
            num_graphs=num_graphs,
        )

        return (x_out, edge_attr_out)

    def _get_node_messages(
        self,
        x: torch.Tensor,
        head_node_index: torch.Tensor,
        tail_node_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # generate messages
        node_messages = self.node_input_linear(x)
        (
            node_to_self_message,
            node_to_neighbor_message,
            head_node_to_edge_message,
            tail_node_to_edge_message,
        ) = node_messages.tensor_split(sections=4, dim=-1)

        # aggregate messages
        head_node_to_edge_message = head_node_to_edge_message[head_node_index]
        tail_node_to_edge_message = tail_node_to_edge_message[tail_node_index]
        node_to_edge_message = head_node_to_edge_message + tail_node_to_edge_message

        return (node_to_self_message, node_to_neighbor_message, node_to_edge_message)

    def _get_edge_messages(
        self, edge_attr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # generate messages
        edge_messages = self.edge_input_linear(edge_attr)
        (
            edge_to_self_message,
            edge_to_tail_node_message,
            edge_attn_message,
        ) = edge_messages.tensor_split(sections=3, dim=-1)

        return (
            edge_to_self_message,
            edge_to_tail_node_message,
            edge_attn_message,
        )

    def _get_edge_attn_weight(
        self,
        edge_attn_message: torch.Tensor,
        tail_node_index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        detached_edge_attn_message = edge_attn_message.detach()
        with torch.no_grad():
            max_edge_attn_message_per_tail_node = scatter_reduce(
                input=detached_edge_attn_message,
                index=tail_node_index,
                num_bins=num_nodes,
                reduction="amax",
                broadcast_index=True,
                gather=True,
            )

        edge_attn_weight = torch.exp(
            edge_attn_message - max_edge_attn_message_per_tail_node
        )
        if self.attention_edge_removal_rate is not None:
            dropout_training = self.training and not self.track_and_use_batch_norm_stats
            edge_attn_weight = F.dropout(
                input=edge_attn_weight,
                p=self.attention_edge_removal_rate,
                training=dropout_training,
            )

        return edge_attn_weight + self.attn_weight_eps

    def _get_node_received_message(
        self,
        node_to_neighbor_message: torch.Tensor,
        edge_to_tail_node_message: torch.Tensor,
        edge_attn_weight: torch.Tensor,
        head_node_index: torch.Tensor,
        tail_node_index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        head_node_to_tail_node_message = node_to_neighbor_message[head_node_index]
        sent_message = (
            head_node_to_tail_node_message + edge_to_tail_node_message
        ) * edge_attn_weight
        sent_message_and_attn_weight = torch.cat(
            (sent_message, edge_attn_weight), dim=-1
        )

        received_message_and_weight = scatter_reduce(
            input=sent_message_and_attn_weight,
            index=tail_node_index,
            num_bins=num_nodes,
            reduction="sum",
            broadcast_index=True,
        )
        (received_message, received_weight) = received_message_and_weight.tensor_split(
            sections=2, dim=-1
        )

        return received_message / (received_weight + self.eps)

    def _get_x_out(
        self,
        x: torch.Tensor,
        node_received_message: torch.Tensor,
        node_to_self_message: torch.Tensor,
        node_batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        # aggregate messages
        x_residual = (
            node_received_message + node_to_self_message + self.node_message_bias
        )

        # activation
        x_residual = self.node_output_linear(F.mish(x_residual))
        if not self.output_linear_needs_bias:
            x_residual = self._workaround_identity_operation(x_residual)

        # skip connection
        x = self._apply_residual(skip_connection=x, residual_connection=x_residual)

        # norm
        if self.node_norm is not None:
            x = self.node_norm(
                input=x,
                batch=node_batch,
                num_graphs=num_graphs,
                eps=self.eps,
            )

        return x

    def _get_edge_attr_out(
        self,
        edge_attr: torch.Tensor,
        node_to_edge_message: torch.Tensor,
        edge_to_self_message: torch.Tensor,
        edge_batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        # aggregate edge messages
        edge_attr_residual = (
            node_to_edge_message + edge_to_self_message + self.edge_message_bias
        )

        # activation
        edge_attr_residual = self.edge_output_linear(F.mish(edge_attr_residual))
        if not self.output_linear_needs_bias:
            edge_attr_residual = self._workaround_identity_operation(edge_attr_residual)

        # skip connection
        edge_attr = self._apply_residual(
            skip_connection=edge_attr, residual_connection=edge_attr_residual
        )

        # norm
        if self.edge_norm is not None:
            edge_attr = self.edge_norm(
                input=edge_attr,
                batch=edge_batch,
                num_graphs=num_graphs,
                eps=self.eps,
            )

        return edge_attr

    def _apply_residual(
        self, skip_connection: torch.Tensor, residual_connection: torch.Tensor
    ) -> torch.Tensor:
        return skip_connection.add(residual_connection, alpha=self.residual_scale)

    @staticmethod
    def _workaround_identity_operation(input: torch.Tensor) -> torch.Tensor:
        # work around an issue in torch.compile by performing an identity operation
        return input + 0.0


class GRASS(GRASSModule):
    def __init__(
        self,
        num_layers: int,
        dim: int,
        hidden_dim: int,
        output_pooling_type: str | None = None,
        norm_type: str | None = "batch",
        attention_edge_removal_rate: float = 0.0,
        alternate_edge_direction: bool = True,
        residual_scale: float = 1.0,
        enable_checkpointing: bool = False,
        eps: float = 1e-7,
    ) -> None:
        assert num_layers >= 1
        assert dim >= 1
        assert hidden_dim >= 1
        assert 0.0 <= attention_edge_removal_rate and attention_edge_removal_rate < 1.0
        assert residual_scale > 0.0
        assert 0.0 < eps and eps <= 1.0

        super().__init__()
        self.num_layers: Final = num_layers
        self.dim: Final = dim
        self.output_pooling_type: Final = output_pooling_type
        self.alternate_edge_direction: Final = alternate_edge_direction
        self.enable_checkpointing: bool = enable_checkpointing

        self.layers: Final = nn.ModuleList()
        for _ in range(num_layers):
            layer = GRASSLayer(
                dim=dim,
                hidden_dim=hidden_dim,
                norm_type=norm_type,
                residual_scale=residual_scale,
                attention_edge_removal_rate=attention_edge_removal_rate,
                eps=eps,
            )
            self.layers.append(layer)

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        for layer in self.layers:
            layer.reset_parameters()

    def toggle_track_and_use_batch_norm_stats(self, enable: bool) -> None:
        for layer in self.layers:
            layer.toggle_track_and_use_batch_norm_stats(enable)

    def forward(
        self,
        batch: torch.Tensor | None,
        num_graphs: int,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        added_edge_mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_nodes = x.size(0)
        if batch is None:
            batch = x.new_zeros(1, dtype=torch.long).expand(num_nodes)

        num_edges = edge_index.size(-1)
        if edge_attr is None:
            edge_attr = x.new_zeros(1).expand(num_edges, self.dim)

        assert batch.shape == (num_nodes,)
        assert x.shape == (num_nodes, self.dim)
        assert edge_index.shape == (2, num_edges)
        assert edge_attr.shape == (num_edges, self.dim)

        (head_node_index, tail_node_index) = edge_index
        edge_batch = batch[tail_node_index]

        for layer in self.layers:
            args = (
                x,
                head_node_index,
                tail_node_index,
                edge_attr,
                batch,
                edge_batch,
                num_nodes,
                num_graphs,
            )
            if self.enable_checkpointing:
                (x, edge_attr) = torch.utils.checkpoint.checkpoint(
                    layer,
                    *args,
                    use_reentrant=False,
                )
            else:
                (x, edge_attr) = layer(*args)

            if self.alternate_edge_direction:
                # invert edges
                head_node_index, tail_node_index = tail_node_index, head_node_index

        return self._apply_graph_pooling(
            x=x,
            edge_attr=edge_attr,
            node_batch=batch,
            edge_batch=edge_batch,
            added_edge_mask=added_edge_mask,
            num_graphs=num_graphs,
        )

    def _apply_graph_pooling(
        self,
        x: torch.Tensor,
        edge_attr: torch.Tensor,
        node_batch: torch.Tensor,
        edge_batch: torch.Tensor,
        added_edge_mask: torch.Tensor | None,
        num_graphs: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.output_pooling_type is None:
            return (x, edge_attr)

        pooled_node_feature = scatter_reduce(
            input=x,
            index=node_batch,
            num_bins=num_graphs,
            reduction=self.output_pooling_type,
            broadcast_index=True,
        )

        num_edge_bins = num_graphs
        if added_edge_mask is not None:
            edge_batch = edge_batch.add(added_edge_mask, alpha=num_graphs)
            num_edge_bins *= 2

        pooled_edge_feature = scatter_reduce(
            input=edge_attr,
            index=edge_batch,
            num_bins=num_edge_bins,
            reduction=self.output_pooling_type,
            broadcast_index=True,
        ).view(-1, num_graphs, self.dim)

        return (pooled_node_feature, pooled_edge_feature)
