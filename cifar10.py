from grass.config import GRASSConfig
from trainers.superpixel_trainer import SuperpixelTrainer

_model_config = {
    # data
    "node_input_type": "continuous",
    "node_input_dim": 5,
    "edge_input_type": "continuous",
    "edge_input_dim": 1,
    # dimensions
    "num_layers": 15,
    "dim": 24,
    "output_dim": 10,
    "hidden_dim": 24,
    # rewiring
    "num_added_edges_per_node": 6,
    # encoding
    "node_encoding_dim": 24,
    "edge_encoding_dim": 2 * 24,
    "add_degree_encoding": True,
    # attention
    "alternate_edge_direction": True,
    "attention_edge_removal_rate": 0.1,
    # normalization
    "norm_type": "batch",
    "residual_scale": 0.4,
    # pooling
    "pooling_type": "sum",
    "expected_num_nodes": 117.647,
    "expected_num_existing_edges": 941.177 + 117.647,
    "expected_num_added_edges": 2 * 6 * 117.647,
    # output
    "task_head_type": "pooling",
    "task_head_hidden_dim": 2 * 3 * 24,
    # performance
    "enable_checkpointing": False,
}

_trainer_config = {
    # optimizer
    "optimizer": "Lion",
    "num_epochs": 400,
    "betas": (0.95, 0.98),
    "grad_scaler_initial_scale": 2.0**14,
    # learning rate schedule
    "warmup_steps_ratio": 0.1,
    "initial_learning_rate": 1e-7,
    "peak_learning_rate": 1e-3,
    "final_learning_rate": 1e-7,
    # weight decay
    "weight_decay_factor": 0.3,
    "exclude_biases_from_weight_decay": True,
    # batch size
    "train_batch_size": 200,
    "test_batch_size": 200,
    # label smoothing:
    "label_smoothing_factor": 0.1,
    # evaluation
    "num_batch_norm_sampling_epochs": 10,
    "num_evaluation_samples": 10,
    # performance
    "data_loader_num_workers": 4,
    "use_triton": True,
    "compile": True,
}

_task_specific_config = {
    "dataset_name": "CIFAR10",
}

if __name__ == "__main__":
    config = GRASSConfig(
        model_config=_model_config,
        trainer_config=_trainer_config,
        task_specific_config=_task_specific_config,
    )
    trainer = SuperpixelTrainer(config)
    trainer.train()
    trainer.evaluate()
