from typing import Final, Iterable, Callable, Tuple, List, Dict, TypedDict
from abc import ABC, abstractmethod
from contextlib import nullcontext
import time

import numpy as np

import torch
import torch.nn.functional as F
import torch.utils.data

import torch_geometric
import torch_geometric.data
import torch_geometric.loader
import torch_geometric.transforms

from torchmetrics.functional.classification import (
    multilabel_average_precision,
    multiclass_f1_score,
)

from grass.gnn import GRASSModule
from grass.utils import scatter_reduce


class GRASSTrainerConfigDict(TypedDict):
    # optimizer
    optimizer: str
    num_epochs: int
    betas: Tuple[float, float]
    grad_scaler_initial_scale: float
    # learning rate schedule
    warmup_steps_ratio: float
    initial_learning_rate: float
    peak_learning_rate: float
    final_learning_rate: float
    # weight decay
    weight_decay_factor: float
    exclude_biases_from_weight_decay: bool
    # batch size
    train_batch_size: int
    test_batch_size: int
    # label smoothing:
    label_smoothing_factor: float | None
    # evaluation
    num_batch_norm_sampling_epochs: int | None
    num_evaluation_samples: int
    # performance
    data_loader_num_workers: int
    use_triton: bool
    compile: bool


class GRASSTrainer(ABC):
    @abstractmethod
    def predict(self, data: torch_geometric.data.Data) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def _get_training_loss(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def _process_epoch(
        self, train: bool
    ) -> Tuple[Tuple[float, ...], Tuple[float, ...], Tuple[float, ...], float]:
        raise NotImplementedError()

    @abstractmethod
    @torch.no_grad()
    def _skim_training_set(self) -> None:
        raise NotImplementedError()

    def __init__(
        self,
        config: GRASSTrainerConfigDict,
        model: GRASSModule,
        device: torch.device | None,
        steps_per_epoch: int,
    ) -> None:
        self.device: Final = device
        self.model: Final = model
        if config["compile"]:
            self.model = torch.compile(self.model, fullgraph=True, dynamic=True)

        self.num_epochs: Final = config["num_epochs"]
        assert self.num_epochs >= 1
        self.num_batch_norm_sampling_epochs: Final = config[
            "num_batch_norm_sampling_epochs"
        ]
        assert (
            self.num_batch_norm_sampling_epochs is None
            or self.num_batch_norm_sampling_epochs >= 1
        )
        self.num_evaluation_samples: Final = config["num_evaluation_samples"]
        assert self.num_evaluation_samples >= 1

        parameter_groups = _get_parameter_groups(
            model=self.model,
            exclude_biases_from_weight_decay=config["exclude_biases_from_weight_decay"],
        )
        self.optimizer: Final = _get_optimizer(
            parameter_groups=parameter_groups,
            optimizer_name=config["optimizer"],
            peak_learning_rate=config["peak_learning_rate"],
            betas=config["betas"],
            weight_decay_factor=config["weight_decay_factor"],
            use_triton=config["use_triton"],
        )
        self.scheduler: Final = _get_scheduler(
            optimizer=self.optimizer,
            num_epochs=self.num_epochs,
            steps_per_epoch=steps_per_epoch,
            initial_learning_rate=config["initial_learning_rate"],
            peak_learning_rate=config["peak_learning_rate"],
            final_learning_rate=config["final_learning_rate"],
            warmup_steps_ratio=config["warmup_steps_ratio"],
        )
        self.grad_scaler: Final = torch.cuda.amp.GradScaler(
            init_scale=config["grad_scaler_initial_scale"]
        )

    def train_batch(
        self, data: torch_geometric.data.Data
    ) -> tuple[torch.Tensor, float]:
        self.model.train()
        data = data.to(device=self.device, non_blocking=True)
        self.optimizer.zero_grad(set_to_none=True)
        prediction = self._autocast_predict(data=data)
        loss = self._autocast_get_training_loss(prediction=prediction, target=data.y)

        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        self.scheduler.step()

        return (prediction.detach(), loss.item())

    @torch.no_grad()
    def infer_batch(self, data: torch_geometric.data.Data) -> torch.Tensor:
        self.model.eval()
        data = data.to(device=self.device, non_blocking=True)
        return self._autocast_predict(data=data).detach()

    def _autocast_predict(self, data: torch_geometric.data.Data) -> torch.Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            return self.predict(data=data)

    def _autocast_get_training_loss(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            return self._get_training_loss(prediction=prediction, target=target)

    def train(self, verbose: bool = True) -> None:
        if verbose:
            print(f"Training {self.num_trainable_parameters} trainable parameters...\n")

        for epoch in range(1, self.num_epochs + 1):
            (train_result, val_result, test_result, training_time) = (
                self._process_epoch(train=True)
            )
            if verbose:
                _print_epoch_info(
                    epoch=epoch,
                    train_result=train_result,
                    val_result=val_result,
                    test_result=test_result,
                    training_time=training_time,
                )

    def evaluate(
        self, verbose: bool = True
    ) -> Tuple[Tuple[float, ...], Tuple[float, ...], Tuple[float, ...]]:
        if self.num_batch_norm_sampling_epochs is None:
            return self._evaluate_and_print_results(verbose=verbose)

        if verbose:
            print("\nCollecting batch normalization statistics for final evaluation...")

        self._collect_and_activate_batch_norm_stats()
        result = self._evaluate_and_print_results(verbose=verbose)
        self._deactivate_batch_norm_stats()
        return result

    def _evaluate_and_print_results(
        self, verbose: bool
    ) -> Tuple[Tuple[float, ...], Tuple[float, ...], Tuple[float, ...]]:
        if verbose:
            print("\nFinal evaluation:")

        results = []
        for _ in range(self.num_evaluation_samples):
            (train_result, val_result, test_result, _) = self._process_epoch(
                train=False
            )
            results.append([list(train_result), list(val_result), list(test_result)])
            if verbose:
                _print_results(
                    train_result=train_result,
                    val_result=val_result,
                    test_result=test_result,
                )

        results = np.array(results)
        results_mean = results.mean(axis=0)
        train_result_mean = tuple(results_mean[0])
        val_result_mean = tuple(results_mean[1])
        test_result_mean = tuple(results_mean[2])
        if verbose and self.num_evaluation_samples >= 2:
            print("Mean:")
            _print_results(
                train_result=train_result_mean,
                val_result=val_result_mean,
                test_result=test_result_mean,
                format=".6e",
            )

            results_variance = results.var(axis=0, ddof=1)
            train_result_variance = tuple(results_variance[0])
            val_result_variance = tuple(results_variance[1])
            test_result_variance = tuple(results_variance[2])
            print("Variance:")
            _print_results(
                train_result=train_result_variance,
                val_result=val_result_variance,
                test_result=test_result_variance,
                format=".6e",
            )

        return (train_result_mean, val_result_mean, test_result_mean)

    def _collect_and_activate_batch_norm_stats(self) -> None:
        self.model.train()
        self.model.toggle_track_and_use_batch_norm_stats(enable=False)
        self.model.toggle_track_and_use_batch_norm_stats(enable=True)
        for _ in range(self.num_batch_norm_sampling_epochs):
            self._skim_training_set()

    def _deactivate_batch_norm_stats(self) -> None:
        self.model.toggle_track_and_use_batch_norm_stats(enable=False)

    @property
    def num_trainable_parameters(self) -> int:
        return _get_num_trainable_parameters(self.model)


class BatchedGRASSTrainer(GRASSTrainer):
    @abstractmethod
    def _process_dataset(
        self, loader: torch_geometric.loader.DataLoader, train: bool
    ) -> Tuple[float, float]:
        raise NotImplementedError()

    def __init__(
        self,
        config: GRASSTrainerConfigDict,
        model: GRASSModule,
        train_dataset: torch_geometric.data.Dataset,
        val_dataset: torch_geometric.data.Dataset,
        test_dataset: torch_geometric.data.Dataset,
        device: torch.device | None,
    ) -> None:
        self.train_loader: Final = torch_geometric.loader.DataLoader(
            dataset=train_dataset,
            batch_size=config["train_batch_size"],
            shuffle=True,
            drop_last=True,
            num_workers=config["data_loader_num_workers"],
            pin_memory=True,
            persistent_workers=True,
        )
        self.val_loader: Final = torch_geometric.loader.DataLoader(
            dataset=val_dataset,
            batch_size=config["test_batch_size"],
            num_workers=config["data_loader_num_workers"],
            pin_memory=True,
            persistent_workers=True,
        )
        self.test_loader: Final = torch_geometric.loader.DataLoader(
            dataset=test_dataset,
            batch_size=config["test_batch_size"],
            num_workers=config["data_loader_num_workers"],
            pin_memory=True,
            persistent_workers=True,
        )
        steps_per_epoch = len(self.train_loader)
        super().__init__(
            config=config, model=model, device=device, steps_per_epoch=steps_per_epoch
        )

    def _process_epoch(
        self, train: bool
    ) -> Tuple[Tuple[float, ...], Tuple[float, ...], Tuple[float, ...], float]:
        with nullcontext() if train else torch.no_grad():
            training_start_time = time.time()
            train_result = self._process_dataset(loader=self.train_loader, train=train)
            training_time = time.time() - training_start_time

        with torch.no_grad():
            val_result = self._process_dataset(loader=self.val_loader, train=False)
            test_result = self._process_dataset(loader=self.test_loader, train=False)

        return (train_result, val_result, test_result, training_time)

    def _process_batch(
        self, data: torch_geometric.data.Data, train: bool
    ) -> Tuple[torch.Tensor, float]:
        if train:
            return self.train_batch(data=data)

        prediction = self.infer_batch(data=data)
        loss = self._autocast_get_training_loss(prediction=prediction, target=data.y)
        return (prediction, loss.item())

    @torch.no_grad()
    def _skim_training_set(self) -> None:
        for data in self.train_loader:
            data = data.to(device=self.device, non_blocking=True)
            self._autocast_predict(data=data)


class BatchedGraphRegressionGRASSTrainer(BatchedGRASSTrainer):
    def __init__(
        self,
        config: GRASSTrainerConfigDict,
        model: GRASSModule,
        train_dataset: torch_geometric.data.Dataset,
        val_dataset: torch_geometric.data.Dataset,
        test_dataset: torch_geometric.data.Dataset,
        device: torch.device | None,
        loss_type: str,
    ) -> None:
        super().__init__(
            config=config,
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            device=device,
        )
        self.loss_type: Final = loss_type

    def _get_training_loss(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        if self.loss_type == "L1":
            return F.l1_loss(input=prediction, target=target)
        else:
            raise ValueError(f"invalid loss type: {self.loss_type}")

    def _process_dataset(
        self, loader: torch_geometric.loader.DataLoader, train: bool
    ) -> Tuple[float, ...]:
        total_loss = 0.0
        for data in loader:
            (_, loss) = self._process_batch(data=data, train=train)
            total_loss += loss * data.num_graphs

        epoch_loss = total_loss / len(loader.dataset)
        return (epoch_loss,)


class BatchedMultilabelGraphClassificationGRASSTrainer(BatchedGRASSTrainer):
    def __init__(
        self,
        config: GRASSTrainerConfigDict,
        model: GRASSModule,
        train_dataset: torch_geometric.data.Dataset,
        val_dataset: torch_geometric.data.Dataset,
        test_dataset: torch_geometric.data.Dataset,
        device: torch.device | None,
    ) -> None:
        super().__init__(
            config=config,
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            device=device,
        )
        self.label_smoothing_factor: Final = config["label_smoothing_factor"]
        assert self.label_smoothing_factor is not None

    def _get_training_loss(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return _get_binary_cross_entropy_loss(
            prediction=prediction,
            target=target,
            label_smoothing_factor=self.label_smoothing_factor,
        )

    def _process_dataset(
        self, loader: torch_geometric.loader.DataLoader, train: bool
    ) -> Tuple[float, ...]:
        epoch_num_graphs = 0
        epoch_total_loss = 0
        predictions = []
        targets = []
        for data in loader:
            (prediction, loss) = self._process_batch(data=data, train=train)

            num_graphs = data.num_graphs
            epoch_num_graphs += num_graphs
            epoch_total_loss += loss * num_graphs

            predictions.append(prediction)
            targets.append(data.y.int())

        epoch_average_loss = epoch_total_loss / epoch_num_graphs

        prediction = torch.cat(predictions, dim=0)
        target = torch.cat(targets, dim=0)
        epoch_average_precision = _get_multilabel_average_precision(
            prediction=prediction, target=target
        )

        return (epoch_average_loss, epoch_average_precision)


class BatchedNodeClassificationGRASSTrainer(BatchedGRASSTrainer):
    def __init__(
        self,
        config: GRASSTrainerConfigDict,
        model: GRASSModule,
        train_dataset: torch_geometric.data.Dataset,
        val_dataset: torch_geometric.data.Dataset,
        test_dataset: torch_geometric.data.Dataset,
        device: torch.device | None,
        metric: str,
        loss_type: str,
        num_classes: int | None,
    ) -> None:
        super().__init__(
            config=config,
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            device=device,
        )
        self.loss_type: Final = loss_type
        self.num_classes: Final = num_classes
        self.get_metric_value: Final = _get_metric(metric)
        self.label_smoothing_factor: Final = config["label_smoothing_factor"]
        assert self.label_smoothing_factor is not None

    def _get_training_loss(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        if self.loss_type == "cross_entropy":
            return _get_cross_entropy_loss(
                prediction=prediction,
                target=target,
                label_smoothing_factor=self.label_smoothing_factor,
            )
        elif self.loss_type == "weighted_cross_entropy":
            assert self.num_classes is not None
            return _get_weighted_cross_entropy_loss(
                prediction=prediction,
                target=target,
                num_classes=self.num_classes,
                label_smoothing_factor=self.label_smoothing_factor,
            )
        else:
            raise ValueError(f"invalid loss type: {self.loss_type}")

    def _process_dataset(
        self, loader: torch_geometric.loader.DataLoader, train: bool
    ) -> Tuple[float, ...]:
        epoch_num_nodes = 0
        epoch_total_loss = 0
        predictions = []
        targets = []
        for data in loader:
            (prediction, loss) = self._process_batch(data=data, train=train)

            num_nodes = data.num_nodes
            epoch_num_nodes += num_nodes
            epoch_total_loss += loss * num_nodes

            predictions.append(prediction)
            targets.append(data.y)

        epoch_average_loss = epoch_total_loss / epoch_num_nodes

        prediction = torch.cat(predictions, dim=0)
        target = torch.cat(targets, dim=0)
        epoch_metric_value = self.get_metric_value(prediction=prediction, target=target)

        return (epoch_average_loss, epoch_metric_value)


class SingleGraphGRASSTrainer(GRASSTrainer):
    @abstractmethod
    def _get_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def _get_result(
        self, prediction: torch.Tensor, target: torch.Tensor, loss: float | None = None
    ) -> Tuple[float, ...]:
        raise NotImplementedError()

    def __init__(
        self,
        config: GRASSTrainerConfigDict,
        model: GRASSModule,
        dataset: torch_geometric.data.Dataset,
        transform: torch_geometric.transforms.BaseTransform | None,
        train_mask: torch.Tensor | None,
        val_mask: torch.Tensor | None,
        test_mask: torch.Tensor | None,
        device: torch.device | None,
    ) -> None:
        if len(dataset) > 1:
            raise ValueError("the dataset contains more than one graph")

        super().__init__(config=config, model=model, device=device, steps_per_epoch=1)
        self.transform: Final = transform
        self.data: Final = dataset[0].to(device=self.device)

        self.train_mask: Final = train_mask
        if self.train_mask is None:
            self.train_mask = self.data.train_mask
        else:
            self.train_mask = self.train_mask.to(device=self.device)

        self.val_mask: Final = val_mask
        if self.val_mask is None:
            self.val_mask = self.data.val_mask
        else:
            self.val_mask = self.val_mask.to(device=self.device)

        self.test_mask: Final = test_mask
        if self.test_mask is None:
            self.test_mask = self.data.test_mask
        else:
            self.test_mask = self.test_mask.to(device=self.device)

    def _get_data(self) -> torch_geometric.data.Data:
        data = self.data.clone()
        if self.transform is not None:
            data = self.transform(data)

        return data

    def _get_autocast_loss(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            return self._get_loss(prediction=prediction, target=target)

    def _get_training_loss(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        train_prediction = prediction[self.train_mask]
        train_target = target[self.train_mask]
        return self._get_loss(prediction=train_prediction, target=train_target)

    def _process_epoch(
        self, train: bool
    ) -> Tuple[Tuple[float, ...], Tuple[float, ...], Tuple[float, ...], float]:
        data = self._get_data()
        training_start_time = time.time()
        if train:
            (prediction, training_loss) = self.train_batch(data=data)
        else:
            prediction = self.infer_batch(data=data)
            training_loss = None

        training_time = time.time() - training_start_time
        train_result = self._get_masked_result(
            prediction=prediction,
            target=data.y,
            mask=self.train_mask,
            loss=training_loss,
        )
        val_result = self._get_masked_result(
            prediction=prediction,
            target=data.y,
            mask=self.val_mask,
        )
        test_result = self._get_masked_result(
            prediction=prediction,
            target=data.y,
            mask=self.test_mask,
        )
        return (train_result, val_result, test_result, training_time)

    def _get_masked_result(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        loss: float | None = None,
    ) -> Tuple[float, ...]:
        prediction = prediction[mask]
        target = target[mask]
        return self._get_result(prediction=prediction, target=target, loss=loss)

    @torch.no_grad()
    def _skim_training_set(self) -> None:
        data = self._get_data()
        self._autocast_predict(data=data)


class SingleGraphNodeClassificationGRASSTrainer(SingleGraphGRASSTrainer):
    def __init__(
        self,
        config: GRASSTrainerConfigDict,
        model: GRASSModule,
        dataset: torch_geometric.data.Dataset,
        transform: torch_geometric.transforms.BaseTransform | None,
        train_mask: torch.Tensor,
        val_mask: torch.Tensor,
        test_mask: torch.Tensor,
        device: torch.device | None,
        metric: str,
        loss_type: str,
        num_classes: int,
    ) -> None:
        super().__init__(
            config=config,
            model=model,
            dataset=dataset,
            transform=transform,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            device=device,
        )
        self.loss_type: Final = loss_type
        self.num_classes: Final = num_classes
        self.get_metric_value: Final = _get_metric(metric)
        self.label_smoothing_factor: Final = config["label_smoothing_factor"]
        assert self.label_smoothing_factor is not None

    def _get_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "cross_entropy":
            return _get_cross_entropy_loss(
                prediction=prediction,
                target=target,
                label_smoothing_factor=self.label_smoothing_factor,
            )
        elif self.loss_type == "weighted_cross_entropy":
            assert self.num_classes is not None
            return _get_weighted_cross_entropy_loss(
                prediction=prediction,
                target=target,
                num_classes=self.num_classes,
                label_smoothing_factor=self.label_smoothing_factor,
            )
        else:
            raise ValueError(f"invalid loss type: {self.loss_type}")

    def _get_result(
        self, prediction: torch.Tensor, target: torch.Tensor, loss: float | None = None
    ) -> Tuple[float, float]:
        if loss is None:
            loss = self._get_autocast_loss(prediction=prediction, target=target).item()

        metric_value = self.get_metric_value(prediction=prediction, target=target)
        return (loss, metric_value)


class SingleGraphNodeRegressionGRASSTrainer(SingleGraphGRASSTrainer):
    def __init__(
        self,
        config: GRASSTrainerConfigDict,
        model: GRASSModule,
        dataset: torch_geometric.data.Dataset,
        transform: torch_geometric.transforms.BaseTransform | None,
        train_mask: torch.Tensor,
        val_mask: torch.Tensor,
        test_mask: torch.Tensor,
        device: torch.device | None,
        loss_type: str,
    ) -> None:
        super().__init__(
            config=config,
            model=model,
            dataset=dataset,
            transform=transform,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            device=device,
        )
        self.loss_type: Final = loss_type

    def _get_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "L1":
            return F.l1_loss(input=prediction, target=target)
        else:
            raise ValueError(f"invalid loss type: {self.loss_type}")

    def _get_result(
        self, prediction: torch.Tensor, target: torch.Tensor, loss: float | None = None
    ) -> Tuple[float]:
        if loss is None:
            loss = self._get_autocast_loss(prediction=prediction, target=target).item()

        return (loss,)


def _get_cross_entropy_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    label_smoothing_factor: float = 0.0,
) -> torch.Tensor:
    target = target.squeeze(-1)
    return F.cross_entropy(
        input=prediction, target=target, label_smoothing=label_smoothing_factor
    )


def _get_weighted_cross_entropy_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    label_smoothing_factor: float = 0.0,
) -> torch.Tensor:
    target = target.squeeze(-1)
    num_nodes = target.size(0)
    label_frequency = torch.bincount(target, minlength=num_classes)
    weight = (num_nodes - label_frequency).float() / max(num_nodes, 1)
    return F.cross_entropy(
        input=prediction,
        target=target,
        weight=weight,
        label_smoothing=label_smoothing_factor,
    )


def _get_binary_cross_entropy_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    label_smoothing_factor: float = 0.0,
) -> torch.Tensor:
    target = target * (1.0 - label_smoothing_factor) + 0.5 * label_smoothing_factor
    return F.binary_cross_entropy_with_logits(input=prediction, target=target)


@torch.no_grad()
def _get_accuracy(prediction: torch.Tensor, target: torch.Tensor) -> float:
    target = target.squeeze(-1)
    is_accurate = prediction.argmax(dim=-1).eq(target).float()
    accuracy = is_accurate.mean()
    return accuracy.item()


@torch.no_grad()
def _get_average_accuracy(prediction: torch.Tensor, target: torch.Tensor) -> float:
    num_classes = prediction.size(-1)
    target = target.squeeze(-1)
    is_accurate = prediction.argmax(dim=-1).eq(target).float()
    accuracy_per_class = scatter_reduce(
        input=is_accurate, index=target, num_bins=num_classes, reduction="mean"
    )
    accuracy = accuracy_per_class.nanmean()
    return accuracy.item()


@torch.no_grad()
def _get_macro_f1_score(prediction: torch.Tensor, target: torch.Tensor) -> float:
    num_classes = prediction.size(-1)
    macro_f1_score = multiclass_f1_score(
        preds=prediction, target=target, num_classes=num_classes, average="macro"
    )
    return macro_f1_score.item()


@torch.no_grad()
def _get_multilabel_average_precision(
    prediction: torch.Tensor, target: torch.Tensor
) -> float:
    num_labels = prediction.size(-1)
    prediction_probability = F.sigmoid(prediction)
    average_precision = multilabel_average_precision(
        preds=prediction_probability, target=target, num_labels=num_labels
    )
    return average_precision.item()


def _get_metric(metric: str) -> Callable[[torch.Tensor, torch.Tensor], float]:
    if metric == "accuracy":
        return _get_accuracy
    elif metric == "average_accuracy":
        return _get_average_accuracy
    elif metric == "macro_f1":
        return _get_macro_f1_score
    else:
        raise ValueError(f"invalid metric: {metric}")


def _get_num_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def _get_parameter_groups(
    model: torch.nn.Module, exclude_biases_from_weight_decay: bool
) -> List[Dict]:
    def should_disable_weight_decay(parameter_name: str) -> bool:
        if exclude_biases_from_weight_decay:
            return "bias" in parameter_name

        return False

    weight_decay_enabled_parameters = (
        parameter
        for name, parameter in model.named_parameters()
        if not should_disable_weight_decay(name)
    )
    weight_decay_disabled_parameters = (
        parameter
        for name, parameter in model.named_parameters()
        if should_disable_weight_decay(name)
    )
    return [
        {"params": weight_decay_enabled_parameters},
        {"params": weight_decay_disabled_parameters, "weight_decay": 0.0},
    ]


def _get_optimizer(
    parameter_groups: Iterable[Dict],
    optimizer_name: str,
    peak_learning_rate: float,
    betas: Tuple[float, float],
    weight_decay_factor: float,
    use_triton: bool = False,
) -> torch.optim.Optimizer:
    if optimizer_name == "AdamW":
        return torch.optim.AdamW(
            params=parameter_groups,
            lr=peak_learning_rate,
            betas=betas,
            eps=1e-7,
            weight_decay=weight_decay_factor,
            fused=True,
        )
    elif optimizer_name == "Lion":
        from lion_pytorch import Lion

        return Lion(
            params=parameter_groups,
            lr=peak_learning_rate,
            betas=betas,
            weight_decay=weight_decay_factor,
            use_triton=use_triton,
        )
    else:
        raise ValueError(f"invalid optimizer: {optimizer_name}")


def _get_scheduler(
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    steps_per_epoch: int,
    initial_learning_rate: float,
    peak_learning_rate: float,
    final_learning_rate: float,
    warmup_steps_ratio: float,
) -> torch.optim.lr_scheduler.LRScheduler:
    initial_div_factor = peak_learning_rate / initial_learning_rate
    final_div_factor = initial_learning_rate / final_learning_rate
    assert initial_div_factor > 0.0
    assert final_div_factor > 0.0
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=peak_learning_rate,
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=warmup_steps_ratio,
        cycle_momentum=False,
        div_factor=initial_div_factor,
        final_div_factor=final_div_factor,
    )


def _print_epoch_info(
    epoch: int,
    train_result: Tuple[float, ...],
    val_result: Tuple[float, ...],
    test_result: Tuple[float, ...],
    training_time: float,
) -> None:
    print(
        f"Epoch: {epoch:04d}; "
        f"Train: {_format_tuple(train_result, '.6f')}; "
        f"Val: {_format_tuple(val_result, '.6f')}; "
        f"Test: {_format_tuple(test_result, '.6f')}; "
        f"Time: {training_time:.2f}"
    )


def _print_results(
    train_result: Tuple[float, ...],
    val_result: Tuple[float, ...],
    test_result: Tuple[float, ...],
    format: str = ".6f",
) -> None:
    print(
        f"Train: {_format_tuple(train_result, format)}, "
        f"Val: {_format_tuple(val_result, format)}, "
        f"Test: {_format_tuple(test_result, format)}"
    )


def _format_tuple(input: Tuple[float, ...], format_spec: str):
    return ", ".join(format(value, format_spec) for value in input)
