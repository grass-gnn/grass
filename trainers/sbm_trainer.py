import os.path as osp

import torch
import torch_geometric
import torch_geometric.data
import torch_geometric.loader
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.transforms import Compose, AddSelfLoops

from grass.model import GRASSModel
from grass.transforms import (
    AddZeroEdgeAttr,
    AddDegree,
    AddRandomEdges,
    AddUndirectedRandomWalkProbabilities,
    GenerateEncoding,
)
from grass.trainer import BatchedNodeClassificationGRASSTrainer
from grass.config import GRASSConfig
from grass.utils import get_device


class SBMTrainer(BatchedNodeClassificationGRASSTrainer):
    def __init__(self, config: GRASSConfig) -> None:
        device = get_device()
        model = GRASSModel(config.model_config).to(device=device)

        num_random_walk_steps = config.model_config["node_encoding_dim"]
        added_self_loop_attr = torch.tensor(1, dtype=torch.long)
        added_random_edge_attr = torch.tensor(2, dtype=torch.long)
        pre_transform = Compose(
            [
                AddZeroEdgeAttr(),
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
                ),
                GenerateEncoding(
                    mode="edge",
                    encoding_attr_name="random_walk_edge_encoding",
                    generator_attr_name="random_walk_probabilities",
                    remove_generator=True,
                ),
            ]
        )

        dataset_name = config.task_specific_config["dataset_name"]
        data_root = osp.join(
            osp.dirname(osp.realpath(__file__)),
            "..",
            "data",
            dataset_name + "_RW",
        )
        train_dataset = GNNBenchmarkDataset(
            root=data_root,
            name=dataset_name,
            pre_transform=pre_transform,
            transform=transform,
            split="train",
        )
        val_dataset = GNNBenchmarkDataset(
            root=data_root,
            name=dataset_name,
            pre_transform=pre_transform,
            transform=transform,
            split="val",
        )
        test_dataset = GNNBenchmarkDataset(
            root=data_root,
            name=dataset_name,
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
            loss_type="weighted_cross_entropy",
            metric="average_accuracy",
            num_classes=config.model_config["output_dim"],
        )

    def predict(self, data: torch_geometric.data.Data) -> torch.Tensor:
        return self.model(
            batch=data.batch,
            num_graphs=data.num_graphs,
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            degree=data.degree,
            node_encoding=data.random_walk_node_encoding,
            edge_encoding=data.random_walk_edge_encoding,
        )
