from typing import Final
import os.path as osp

import torch
import torch_geometric
import torch_geometric.data
import torch_geometric.loader
from torch_geometric.datasets import LRGBDataset
from torch_geometric.transforms import Compose, AddSelfLoops

from grass.model import GRASSModel
from grass.transforms import (
    CastFloat,
    AppendZerosToEdgeAttr,
    AddDegree,
    AddRandomEdges,
    AddDecomposedRandomWalk,
    GenerateEncoding,
)
from grass.trainer import BatchedNodeClassificationGRASSTrainer
from grass.config import GRASSConfig
from grass.utils import get_device


class LRGBSuperpixelTrainer(BatchedNodeClassificationGRASSTrainer):
    def __init__(self, config: GRASSConfig) -> None:
        device = get_device()
        model = GRASSModel(config.model_config).to(device=device)

        random_walk_max_num_components = config.model_config["node_encoding_dim"]
        added_self_loop_attr = torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float16)
        added_random_edge_attr = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float16)
        pre_transform = Compose(
            [
                CastFloat(torch.float16),
                AppendZerosToEdgeAttr(num_zeros=2),
                AddDegree(is_directed=False),
                AddDecomposedRandomWalk(
                    mode="undirected",
                    max_num_components=random_walk_max_num_components,
                    degree_attr_name="degree",
                ),
                GenerateEncoding(
                    mode="node",
                    encoding_attr_name="random_walk_node_encoding",
                    generator_attr_name="decomposed_random_walk",
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
                    generator_attr_name="decomposed_random_walk",
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
        train_dataset = LRGBDataset(
            root=data_root,
            name=dataset_name,
            pre_transform=pre_transform,
            transform=transform,
            split="train",
        )
        val_dataset = LRGBDataset(
            root=data_root,
            name=dataset_name,
            pre_transform=pre_transform,
            transform=transform,
            split="val",
        )
        test_dataset = LRGBDataset(
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
            metric="macro_f1",
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
            added_edge_mask=data.added_edge_mask,
        )
