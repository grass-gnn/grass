import os.path as osp

import torch
import torch_geometric
import torch_geometric.data
import torch_geometric.loader
from torch_geometric.datasets import HeterophilousGraphDataset
from torch_geometric.transforms import Compose, AddSelfLoops

from grass.model import GRASSModel
from grass.transforms import (
    AddZeroEdgeAttr,
    AddDegree,
    AddRandomEdges,
    AddDecomposedRandomWalk,
    GenerateEncoding,
)
from grass.trainer import SingleGraphNodeClassificationGRASSTrainer
from grass.config import GRASSConfig
from grass.utils import get_device


class HeterophilousGraphTrainer(SingleGraphNodeClassificationGRASSTrainer):
    def __init__(self, config: GRASSConfig) -> None:
        device = get_device()
        model = GRASSModel(config.model_config).to(device=device)

        random_walk_max_num_components = config.model_config["node_encoding_dim"] * 4
        num_random_walk_steps = config.model_config["node_encoding_dim"]
        eigensolver_tolerance = 1e-4 * 1e-4
        added_self_loop_attr = torch.tensor(1, dtype=torch.long)
        added_random_edge_attr = torch.tensor(2, dtype=torch.long)
        pre_transform = Compose(
            [
                AddZeroEdgeAttr(),
                AddDegree(is_directed=True),
                AddDecomposedRandomWalk(
                    mode="undirected",
                    max_num_components=random_walk_max_num_components,
                    num_random_walk_steps=num_random_walk_steps,
                    eigensolver_tolerance=eigensolver_tolerance,
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
                    add_directed_edges=False,
                    added_edge_attr=added_random_edge_attr,
                    added_edge_mask_attr_name="added_edge_mask",
                    device=device,
                ),
                GenerateEncoding(
                    mode="edge",
                    encoding_attr_name="random_walk_edge_encoding",
                    generator_attr_name="decomposed_random_walk",
                    device=device,
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
        dataset = HeterophilousGraphDataset(
            root=data_root, name=dataset_name, pre_transform=pre_transform
        )
        train_mask = dataset[0].train_mask[:, 0]
        val_mask = dataset[0].val_mask[:, 0]
        test_mask = dataset[0].test_mask[:, 0]

        super().__init__(
            config=config.trainer_config,
            model=model,
            dataset=dataset,
            transform=transform,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            device=device,
            metric="accuracy",
            loss_type="cross_entropy",
            num_classes=config.model_config["output_dim"],
        )

    def predict(self, data: torch_geometric.data.Data) -> torch.Tensor:
        return self.model(
            batch=None,
            num_graphs=1,
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            degree=data.degree,
            node_encoding=data.random_walk_node_encoding,
            edge_encoding=data.random_walk_edge_encoding,
            added_edge_mask=data.added_edge_mask,
        )
