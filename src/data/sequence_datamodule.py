"""Wrapper around a LightningDataModule."""

import logging
from functools import partial
from typing import Any, Dict, List, Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.data.components.interfaces import DataloaderConfig
from src.data.utils import assign_files_to_workers
from src.utils.file_utils import list_files


class SequenceDataModule(LightningDataModule):
    """A LightningDataModule that encapsulates data splitting, preprocessing,
    parallelization and batching.
    """

    def __init__(
        self,
        train_dataloader_config: DataloaderConfig,
        val_dataloader_config: DataloaderConfig,
        test_dataloader_config: DataloaderConfig,
    ) -> None:
        """Construct a SequenceDataModule using the provided config files.

        The attributes `map_train_files_per_device`,
        `map_val_files_per_device`, and `map_test_files_per_device` are
        initialized as None, and are later modified by `setup()` to contain
        mappings from device indices to lists of data files assigned to that
        device.

        :param train_dataloader_config: Training dataloader configuration passed
            by Hydra.
        :param val_dataloader_config: Validation dataloader configuration passed
            by Hydra.
        :param test_dataloader_config: Test dataloader configuration passed
            by Hydra.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.list_of_training_files = list_files(
            folder_path=self.hparams.train_dataloader_config.data_folder,
            suffix=f"*{self.hparams.train_dataloader_config.dataset_config.data_iterator.get_file_suffix()}",
        )
        self.list_of_val_files = list_files(
            folder_path=self.hparams.val_dataloader_config.data_folder,
            suffix=f"*{self.hparams.val_dataloader_config.dataset_config.data_iterator.get_file_suffix()}",
        )
        self.list_of_test_files = list_files(
            folder_path=self.hparams.test_dataloader_config.data_folder,
            suffix=f"*{self.hparams.test_dataloader_config.dataset_config.data_iterator.get_file_suffix()}",
        )
        self.map_train_files_per_device = None
        self.map_val_files_per_device = None
        self.map_test_files_per_device = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Assign data files to GPUs.

        Note that `self.trainer.world_size` is the total number of GPUs, and is
        equal to (# nodes) x (# GPUs per node).
        This method is called by Lightning before `trainer.fit()`,
        `trainer.validate()`, `trainer.test()`, and `trainer.predict()`, so be
        careful not to execute things like random split twice! Also, it is
        called after `self.prepare_data()` and there is a barrier in between
        which ensures that all the processes proceed to `self.setup()` once the
        data is prepared and available for use.

        :param stage: Unused parameter. Lightning implementation of setup() uses
            stage to determine which dataset splits (train, val, test) to set
            up, but we choose to set up all splits on each call to setup() here.

        :raise AttributeError: If `self.trainer` is not initialized.
        """
        if not hasattr(self, "trainer"):
            raise AttributeError(
                f"self.trainer must be initialized before call to setup()."
            )

        # assign files to `self.trainer.world_size` number of GPU workers
        self.map_train_files_per_device, _ = assign_files_to_workers(
            list_of_files=self.list_of_training_files,
            total_workers=self.trainer.world_size,
            assign_by_size=self.hparams.train_dataloader_config.assign_files_by_size,
        )
        self.map_val_files_per_device, _ = assign_files_to_workers(
            list_of_files=self.list_of_val_files,
            total_workers=self.trainer.world_size,
            assign_by_size=self.hparams.val_dataloader_config.assign_files_by_size,
        )
        self.map_test_files_per_device, _ = assign_files_to_workers(
            list_of_files=self.list_of_test_files,
            total_workers=self.trainer.world_size,
            assign_by_size=self.hparams.test_dataloader_config.assign_files_by_size,
        )

    def get_dataloader(
        self,
        curr_config: DataloaderConfig,
        map_files_per_device: Dict[int, List[str]],
    ) -> DataLoader[Any]:
        """Construct a DataLoader on a single GPU using config `curr_config`.

        The single GPU is managed by Lightning and corresponds to
        `self.trainer.global_rank`.

        :param curr_config: Config that determines dataloader properties.
        :param map_files_idx_per_device: Map from GPUs to file indices.

        :return: DataLoader running on one GPU that processes the files
            assigned to that GPU, according to `map_files_per_device`.

        :raise AttributeError: If `self.trainer` is not initialized.
        """
        if not hasattr(self, "trainer"):
            raise AttributeError(
                f"self.trainer must be initialized before call to get_dataloader()."
            )

        # We initialize the dataset with the parameters passed on the config.
        dataset = curr_config.dataset_class(
            dataset_config=curr_config.dataset_config,
            data_folder=curr_config.data_folder,
            should_shuffle_rows=curr_config.should_shuffle_rows,
            batch_size=curr_config.batch_size_per_device,
        )  # type: ignore

        # get the files corresponding to the current GPU
        device_file_list = map_files_per_device[self.trainer.global_rank]
        dataset.set_list_of_files(list_of_files=device_file_list)
        # set the number of total GPUs and GPU index
        dataset.set_distributed_params(
            total_workers=self.trainer.world_size,
            global_worker_id=self.trainer.global_rank,
        )

        # Any additional parameters for the masking function should be added to
        # the config and passed there. This is required because we can't pickle
        # lambda functions but the collate fn needs to receive just the batch.
        collate_fn_partial = partial(
            curr_config.collate_fn,
            labels=curr_config.labels,
            sequence_length=curr_config.sequence_length,
            masking_token=curr_config.masking_token,
            padding_token=curr_config.padding_token,
            oov_token=curr_config.oov_token,
        )
        if curr_config.num_workers == 0:
            persistent_workers = False
            logging.warning(
                "num_workers is set to 0, persistent_workers will be set to"
                " False as persistent workers require num_workers > 0"
            )
        else:
            persistent_workers = curr_config.persistent_workers

        return (
            DataLoader(
                dataset=dataset,
                batch_size=curr_config.batch_size_per_device
                if curr_config.dataset_config.iterate_per_row
                else None,
                num_workers=curr_config.num_workers,  # num workers per GPU
                pin_memory=curr_config.pin_memory,
                persistent_workers=persistent_workers,
                drop_last=curr_config.drop_last
                if curr_config.dataset_config.iterate_per_row
                else False,
                collate_fn=collate_fn_partial,
            ),
        )  # type: ignore

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.

        :raise AttributeError: If `self.map_train_files_per_device` is not
            initialized.
        """
        if not hasattr(self, "map_train_files_per_device"):
            raise AttributeError(
                f"self.map_train_files_per_device must be initialized."
            )
        return self.get_dataloader(
            self.hparams.train_dataloader_config,
            self.map_train_files_per_device,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.

        :raise AttributeError: If `self.map_val_files_per_device` is not
            initialized.
        """
        if not hasattr(self, "map_val_files_per_device"):
            raise AttributeError(f"self.map_val_files_per_device must be initialized.")
        return self.get_dataloader(
            self.hparams.val_dataloader_config,
            self.map_val_files_per_device,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.

        :raise AttributeError: If `self.map_test_files_per_device` is not
            initialized.
        """
        if not hasattr(self, "map_test_files_per_device"):
            raise AttributeError(f"self.map_test_files_per_device must be initialized.")
        return self.get_dataloader(
            self.hparams.test_dataloader_config,
            self.map_test_files_per_device,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`,
        `trainer.validate()`, `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the
        datamodule state.

        :return: A dictionary containing the datamodule state that you want to
            save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule
        state given datamodule `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass
