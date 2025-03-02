from grass.config import GRASSConfig
from trainers.peptides_trainer import PeptidesStructTrainer

_model_config = {
    # data
    "node_input_type": "atom",
    "node_input_dim": 9,
    "edge_input_type": "bond",
    "edge_input_dim": 3,
    # dimensions
    "num_layers": 48,
    "dim": 32,
    "output_dim": 11,
    "hidden_dim": 32,
    # rewiring
    "num_added_edges_per_node": 3,
    # encoding
    "node_encoding_dim": 64,
    "edge_encoding_dim": 2 * 64,
    "add_degree_encoding": True,
    # attention
    "alternate_edge_direction": True,
    "attention_edge_removal_rate": 0.1,
    # normalization
    "norm_type": "batch",
    "residual_scale": 0.2,
    # pooling
    "pooling_type": "sum",
    "expected_num_nodes": 151.5390,
    "expected_num_existing_edges": 308.5432 + 151.5390,
    "expected_num_added_edges": 2 * 3 * 151.5390,
    # output
    "task_head_type": "pooling",
    "task_head_hidden_dim": 2 * 3 * 32,
    # performance
    "enable_checkpointing": False,
}

_trainer_config = {
    # optimizer
    "optimizer": "Lion",
    "num_epochs": 500,
    "betas": (0.95, 0.98),
    "grad_scaler_initial_scale": 2.0**14,
    # learning rate schedule
    "warmup_steps_ratio": 0.1,
    "initial_learning_rate": 1e-7,
    "peak_learning_rate": 1e-3,
    "final_learning_rate": 1e-7,
    # weight decay
    "weight_decay_factor": 3.0,
    "exclude_biases_from_weight_decay": True,
    # batch size
    "train_batch_size": 200,
    "test_batch_size": 200,
    # label smoothing:
    "label_smoothing_factor": None,
    # evaluation
    "num_batch_norm_sampling_epochs": 10,
    "num_evaluation_samples": 10,
    # performance
    "data_loader_num_workers": 4,
    "use_triton": True,
    "compile": True,
}

_task_specific_config = {
    "dataset_name": "Peptides-struct",
}

if __name__ == "__main__":
    config = GRASSConfig(
        model_config=_model_config,
        trainer_config=_trainer_config,
        task_specific_config=_task_specific_config,
    )
    trainer = PeptidesStructTrainer(config)
    trainer.train()
    trainer.evaluate()
