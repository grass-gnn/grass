from typing import Tuple
import os.path as osp

import torch
import torch_geometric
import torch_geometric.data
import torch_geometric.loader
from torch_geometric.datasets import LRGBDataset
from torch_geometric.transforms import BaseTransform, Compose, AddSelfLoops
from ogb.utils.features import get_bond_feature_dims

from grass.model import GRASSModel
from grass.transforms import (
    AddDegree,
    AddRandomEdges,
    AddUndirectedRandomWalkProbabilities,
    AddDecomposedRandomWalk,
    GenerateEncoding,
)
from grass.trainer import (
    BatchedGraphRegressionGRASSTrainer,
    BatchedMultilabelGraphClassificationGRASSTrainer,
)
from grass.config import GRASSConfig
from grass.utils import get_device


class PeptidesStructTrainer(BatchedGraphRegressionGRASSTrainer):
    def __init__(self, config: GRASSConfig) -> None:
        num_random_walk_steps = config.model_config["node_encoding_dim"]
        added_self_loop_attr = torch.tensor(get_bond_feature_dims(), dtype=torch.long)
        added_random_edge_attr = torch.tensor(
            tuple(i + 1 for i in get_bond_feature_dims()), dtype=torch.long
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
        (model, train_dataset, val_dataset, test_dataset, device) = (
            _get_training_resources(
                config=config, pre_transform=pre_transform, transform=transform
            )
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


class PeptidesFuncTrainer(BatchedMultilabelGraphClassificationGRASSTrainer):
    def __init__(self, config: GRASSConfig) -> None:
        random_walk_max_num_components = config.model_config["node_encoding_dim"]
        added_self_loop_attr = torch.tensor(get_bond_feature_dims(), dtype=torch.long)
        added_random_edge_attr = torch.tensor(
            tuple(i + 1 for i in get_bond_feature_dims()), dtype=torch.long
        )
        pre_transform = Compose(
            [
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
        (model, train_dataset, val_dataset, test_dataset, device) = (
            _get_training_resources(
                config=config, pre_transform=pre_transform, transform=transform
            )
        )

        super().__init__(
            config=config.trainer_config,
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            device=device,
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


def _get_training_resources(
    config: GRASSConfig, pre_transform: BaseTransform, transform: BaseTransform
) -> Tuple[
    torch.device,
    GRASSModel,
    torch_geometric.data.Dataset,
    torch_geometric.data.Dataset,
    torch_geometric.data.Dataset,
]:
    device = get_device()
    model = GRASSModel(config.model_config).to(device=device)

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

    return (model, train_dataset, val_dataset, test_dataset, device)
