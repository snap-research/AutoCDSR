import logging
from typing import List, Optional

import torch
from torch.utils.data import IterableDataset, get_worker_info

from src.data.components.interfaces import DatasetConfig
from src.data.components.pre_processing import filter_features_to_consider


class MultiSequenceIterable(IterableDataset):
    """This dataset is used to read files from gcs or local storage and extract sequential data from it.
    It is a streaming iterable dataset that can automatically determines the worker id so no duplicate sample is generated across all worker.

    Parameters
    ----------
    gcs_or_local_path: str
        address for gcs or local storage
    user_id_field: str
       the column name for user_id
    sequence_field: str
        the column name for sequence data
    shuffle: bool = False
        whether to shuffle the data
    is_for_training: bool = True
        whether or not the dataset is for training purpose
        training dataset will be infinite iterable dataset
    for_train: bool = False
        whether or not the multi sequence dataset is for training
    """

    def __init__(
        self,
        dataset_config: DatasetConfig,
        data_folder: str,
        should_shuffle_rows: bool = False,
        batch_size: int = 1,
        for_train: bool = True,
    ):
        self.dataset_config = dataset_config
        self.should_shuffle_rows = should_shuffle_rows
        self.data_iterator = dataset_config.data_iterator
        self.data_folder = data_folder
        self.list_of_file_paths = []
        self.batch_size = batch_size
        self.for_train = for_train

    def set_list_of_files(self, list_of_files: List[str]):
        self.list_of_file_paths = list_of_files

    def set_distributed_params(self, total_workers: int, global_worker_id: int):
        self.total_workers = total_workers
        self.global_worker_id = global_worker_id

    def get_worker_id_and_num_workers(self):
        worker_info = get_worker_info()

        if worker_info is None:
            # Single-worker setup (no multiprocessing)
            worker_id = 0
            num_workers = 1
        else:
            # Multi-worker setup
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        return worker_id, num_workers

    def get_list_of_worker_files(self):
        # Get information about worker and then separate only files that belong to this worker
        worker_id, num_workers = self.get_worker_id_and_num_workers()
        worker_files = self.list_of_file_paths[worker_id::num_workers]
        logging.debug(
            f"GPU Worker: {self.global_worker_id}/{self.total_workers} CPU Worker {worker_id} has {len(worker_files)} files"
        )
        return worker_files

    def __iter__(self):
        # We update each worker's data iterator with the files just for that worker.
        while True:
            self.data_iterator.update_list_of_file_paths(
                self.get_list_of_worker_files()
            )

            self.data_iterator = (
                self.data_iterator.shuffle()
                if self.should_shuffle_rows
                else self.data_iterator
            )
            # We provide the flexibility to iterate per row, if per row preprocessing is needed, or per batch.
            dataset_to_iterate = (
                self.data_iterator.iterrows()
                if self.dataset_config.iterate_per_row
                else self.data_iterator.iter_batches(self.batch_size)
            )

            for row_or_batch in dataset_to_iterate:
                row_or_batch = filter_features_to_consider(
                    row_or_batch, self.dataset_config
                )
                for (
                    preprocessing_function
                ) in self.dataset_config.preprocessing_functions:
                    row_or_batch = preprocessing_function(
                        row_or_batch, dataset_config=self.dataset_config
                    )
                    if row_or_batch is None:
                        break

                if row_or_batch:
                    yield row_or_batch

            if not self.for_train:
                break
