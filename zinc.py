from grass.config import GRASSConfig
from trainers.zinc_trainer import ZINCTrainer

_model_config = {
    # data
    "node_input_type": "index",
    "node_input_dim": 28,
    "edge_input_type": "index",
    "edge_input_dim": 4,
    # dimensions
    "num_layers": 49,
    "dim": 32,
    "output_dim": 1,
    "hidden_dim": 32,
    # rewiring
    "num_added_edges_per_node": 6,
    # encoding
    "node_encoding_dim": 32,
    "edge_encoding_dim": 2 * 32,
    "add_degree_encoding": True,
    # attention
    "alternate_edge_direction": True,
    "attention_edge_removal_rate": 0.1,
    # normalization
    "norm_type": "batch",
    "residual_scale": 0.2,
    # pooling
    "pooling_type": "sum",
    "expected_num_nodes": 23.1664,
    "expected_num_existing_edges": 49.8558 + 23.1664,
    "expected_num_added_edges": 2 * 6 * 23.1664,
    # output
    "task_head_type": "pooling",
    "task_head_hidden_dim": 2 * 3 * 32,
    # performance
    "enable_checkpointing": False,
}

_trainer_config = {
    # optimizer
    "optimizer": "Lion",
    "num_epochs": 2000,
    "betas": (0.95, 0.98),
    "grad_scaler_initial_scale": 2.0**13,
    # learning rate schedule
    "warmup_steps_ratio": 0.1,
    "initial_learning_rate": 1e-7,
    "peak_learning_rate": 5e-4,
    "final_learning_rate": 1e-7,
    # weight decay
    "weight_decay_factor": 0.5,
    "exclude_biases_from_weight_decay": True,
    # batch size
    "train_batch_size": 200,
    "test_batch_size": 200,
    # label smoothing:
    "label_smoothing_factor": None,
    # evaluation
    "num_batch_norm_sampling_epochs": 100,
    "num_evaluation_samples": 100,
    # performance
    "data_loader_num_workers": 4,
    "use_triton": True,
    "compile": True,
}

_task_specific_config = {
    "use_subset": True,
}

if __name__ == "__main__":
    config = GRASSConfig(
        model_config=_model_config,
        trainer_config=_trainer_config,
        task_specific_config=_task_specific_config,
    )
    trainer = ZINCTrainer(config)
    trainer.train()
    trainer.evaluate()
