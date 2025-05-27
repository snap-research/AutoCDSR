import logging
from typing import Any, Dict, Optional, Tuple

import torch
import transformers
from lightning import LightningModule
from torchmetrics import MeanMetric, MinMetric
from torchmetrics.aggregation import BaseAggregator

from src.components.eval_metrics import Evaluator
from src.data.components.interfaces import (
    SequentialModelInputData,
    SequentialModuleLabelData,
)


class HFTransformerBaseModule(LightningModule):
    def __init__(
        self,
        huggingface_model: transformers.PreTrainedModel,
        postprocessor: torch.nn.Module,
        aggregator: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        loss_function: torch.nn.Module,
        evaluator: Evaluator,
        weight_tying: bool,
        compile: bool,
        trainining_loop_function: callable = None,
        feature_to_model_input_map: Dict[str, str] = {},
        decoder: torch.nn.Module = None,
        multi_task_plugin: callable = None,
    ) -> None:

        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        # we remove the nn.Modules as they are already checkpointed to avoid doing it twice

        self.save_hyperparameters(
            logger=False,
            ignore=[
                "huggingface_model",
                "postprocessor",
                "aggregator",
                "decoder",
                "loss_function",
            ],
        )

        self.encoder = huggingface_model
        self.embedding_post_processor = postprocessor
        self.decoder = decoder
        self.aggregator = aggregator
        # loss function
        self.criterion = loss_function
        self.evaluator = evaluator

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # For tracking lowest validation loss
        self.val_loss_lowest = MinMetric()

        # we create one mean metric for each metric in the evaluator so we can aggregate them
        self.metrics = {
            metric_name: MeanMetric() for metric_name in evaluator.metrics.keys()
        }
        # we set the metrics to be class attributes as required by lightning. To log metrics,
        # lightning requires it to be class attributes
        for metric_name in self.metrics:
            setattr(self, metric_name, self.metrics[metric_name])

        # if training_loop function is passed, we need to disable automatic optimization
        self.trainining_loop_function = trainining_loop_function
        if self.trainining_loop_function is not None:
            self.automatic_optimization = False
        self.feature_to_model_input_map = feature_to_model_input_map
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)

        self.multi_task_plugin = multi_task_plugin

    def forward(
        self,
        attention_mask: torch.Tensor,
        **kwargs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        raise NotImplementedError(
            "Inherit from this class and implement the forward method."
        )

    def model_step(
        self,
        model_input: SequentialModelInputData,
        label_data: Optional[SequentialModuleLabelData] = None,
    ):
        raise NotImplementedError(
            "Inherit from this class and implement the forward method."
        )

    def get_embedding_table(self):
        if self.hparams.weight_tying:  # type: ignore
            return self.encoder.get_input_embeddings().weight
        else:
            return self.decoder.weight

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.evaluator.reset()
        self.train_loss.reset()
        self.test_loss.reset()
        self.val_loss_lowest.reset()
        for metric in self.metrics:
            self.metrics[metric].reset()

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data of data (tuple). Because of lightning, the tuple is wrapped in another tuple,
        and the actual batch is at position 0. The batch is a tuple of data where first object is a SequentialModelInputData object
        and second is a SequentialModuleLabelData object.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        # Lightning wraps it in a tuple for training, we get the batch from position 0.
        # this behavior only happens for training_step.
        batch = batch[0]
        # Batch is a tuple of model inputs and labels.
        model_input: SequentialModelInputData = batch[0]
        label_data: SequentialModuleLabelData = batch[1]
        # Batch will be a tuple of model inputs and labels. We use the index here to access them.
        model_output, loss = self.model_step(
            model_input=model_input, label_data=label_data
        )

        # update and log metrics. Will only be logged at the interval specified in the logger config
        self.train_loss(loss)
        self.log(
            "train/loss", self.train_loss, on_step=True, on_epoch=False, prog_bar=True
        )

        if self.multi_task_plugin is not None:
            loss = self.multi_task_plugin(self, loss)

        # If a training loop function is passed, we call it with the module and the loss.
        # otherwise we use the automatic optimization provided by lightning
        if self.trainining_loop_function is not None:
            self.trainining_loop_function(self, loss)

        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        train_loss = self.train_loss.compute()
        self.log("train/loss_epoch", train_loss, on_step=False, on_epoch=True)

    def on_validation_epoch_start(self) -> None:
        """Lightning hook that is called when a validation epoch starts."""
        self.val_loss.reset()
        self.evaluator.reset()
        for metric in self.metrics:
            self.metrics[metric].reset()

    def eval_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        loss_to_aggregate: BaseAggregator,
    ):
        """Perform a single evaluation step on a batch of data from the validation or test set.
        The method will update the metrics and the loss that is passed.
        """
        # Batch is a tuple of model inputs and labels.
        model_input: SequentialModelInputData = batch[0]
        label_data: SequentialModuleLabelData = batch[1]

        model_output_before_aggregation, loss = self.model_step(
            model_input=model_input, label_data=label_data
        )

        model_output_after_aggregation = self.aggregator(
            model_output_before_aggregation, model_input.mask
        )
        results = self.evaluator(
            query_embeddings=model_output_after_aggregation,
            key_embeddings=self.get_embedding_table().to(
                model_output_after_aggregation.device
            ),
            labels=list(label_data.labels.values())[0].to(
                model_output_after_aggregation.device
            ),
        )
        loss_to_aggregate(loss)
        for metric in self.metrics:
            # extract float from metric
            self.metrics[metric].update(
                value=results[metric],
                weight=results[self.evaluator._DEFAULT_TOTAL_SAMPLE_NAME],
            )

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data of data (tuple) where first object is a SequentialModelInputData object
        and second is a SequentialModuleLabelData object.
        """
        self.eval_step(batch, self.val_loss)

    def log_metrics(self, prefix: str) -> Dict[str, Any]:
        for metric in self.metrics:
            metric_value = self.metrics[metric].compute().to(self.device)
            self.log(
                f"{prefix}/{metric}",
                metric_value,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            self.metrics[metric].reset()

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        validation_loss = self.val_loss.compute().to(self.device)
        self.val_loss_lowest(validation_loss)

        self.log(
            "val/lowest_loss",
            self.val_loss_lowest.compute().to(self.device),
            sync_dist=True,
            prog_bar=True,
        )
        self.log_metrics("val")

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data of data (tuple) where first object is a SequentialModelInputData object
        and second is a SequentialModuleLabelData object.
        """
        self.eval_step(batch, self.test_loss)

    def on_test_epoch_end(self) -> None:
        test_loss = self.test_loss.compute().to(self.device)
        self.log("test/loss", test_loss, sync_dist=True, prog_bar=True)
        self.log_metrics("test")

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        # if self.hparams.compile and stage == "fit":
        #     self.net = torch.compile(self.net)
        pass

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
