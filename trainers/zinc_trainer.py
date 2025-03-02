import os.path as osp

import torch
import torch_geometric
import torch_geometric.data
import torch_geometric.loader
from torch_geometric.datasets import ZINC
from torch_geometric.transforms import Compose, AddSelfLoops

from grass.model import GRASSModel
from grass.transforms import (
    AddDegree,
    AddRandomEdges,
    AddUndirectedRandomWalkProbabilities,
    GenerateEncoding,
)
from grass.trainer import BatchedGraphRegressionGRASSTrainer
from grass.config import GRASSConfig
from grass.utils import get_device


class ZINCTrainer(BatchedGraphRegressionGRASSTrainer):
    def __init__(self, config: GRASSConfig) -> None:
        device = get_device()
        model = GRASSModel(config.model_config).to(device=device)

        num_random_walk_steps = config.model_config["node_encoding_dim"]
        added_self_loop_attr = torch.tensor(
            config.model_config["edge_input_dim"], dtype=torch.long
        )
        added_random_edge_attr = torch.tensor(
            config.model_config["edge_input_dim"] + 1, dtype=torch.long
        )
        pre_transform = Compose(
            [
                AddDegree(is_directed=False),
                AddUndirectedRandomWalkProbabilities(
                    num_random_walk_steps=num_random_walk_steps,
                    include_reverse_random_walk=True,
                    degree_attr_name="degree",
                ),
                GenerateEncoding(
                    mode="node",
                    encoding_attr_name="random_walk_node_encoding",
                    generator_attr_name="random_walk_probabilities",
                ),
                AddSelfLoops(attr="edge_attr", fill_value=added_self_loop_attr),
            ]
        )
        transform = Compose(
            [
                AddRandomEdges(
                    num_added_edges_per_node=config.model_config[
                        "num_added_edges_per_node"
                    ],
                    added_edge_attr=added_random_edge_attr,
                    added_edge_mask_attr_name="added_edge_mask",
                ),
                GenerateEncoding(
                    mode="edge",
                    encoding_attr_name="random_walk_edge_encoding",
                    generator_attr_name="random_walk_probabilities",
                    remove_generator=True,
                ),
            ]
        )

        data_root = osp.join(
            osp.dirname(osp.realpath(__file__)), "..", "data", "ZINC_RW"
        )
        train_dataset = ZINC(
            root=data_root,
            subset=config.task_specific_config["use_subset"],
            pre_transform=pre_transform,
            transform=transform,
            split="train",
        )
        val_dataset = ZINC(
            root=data_root,
            subset=config.task_specific_config["use_subset"],
            pre_transform=pre_transform,
            transform=transform,
            split="val",
        )
        test_dataset = ZINC(
            root=data_root,
            subset=config.task_specific_config["use_subset"],
            pre_transform=pre_transform,
            transform=transform,
            split="test",
        )

        super().__init__(
            config=config.trainer_config,
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            device=device,
            loss_type="L1",
        )

    def predict(self, data: torch_geometric.data.Data) -> torch.Tensor:
        x = data.x.squeeze(1)
        return 2.0 * self.model(
            batch=data.batch,
            num_graphs=data.num_graphs,
            x=x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            degree=data.degree,
            node_encoding=data.random_walk_node_encoding,
            edge_encoding=data.random_walk_edge_encoding,
            added_edge_mask=data.added_edge_mask,
        ).squeeze(1)
