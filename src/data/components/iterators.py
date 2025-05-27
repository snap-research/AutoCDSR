import logging
import os
import random
from abc import ABC, abstractmethod
from typing import Dict, List

import dask.dataframe as dd
import fastavro
import fsspec

# We suppress the tensorflow warnings. Needs to happend before the tf import
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf


class RawDataIterator(ABC):
    """the abstract class for raw data iterator (e.g., parquet, avro, etc.)

    Parameters
    ----------
    list_of_file_paths : List[str]
        the list of file paths to read from
    read_from_gcs : bool
        whether to read from gcs or local storage
    """

    def __init__(
        self,
        **kwargs,
    ):
        self.list_of_file_paths = None

    def update_list_of_file_paths(self, list_of_file_paths: List[str]):
        self.list_of_file_paths = list_of_file_paths

    @abstractmethod
    def get_file_suffix(self) -> str:
        raise NotImplementedError("Must be implemented in child classes")

    @abstractmethod
    def iterrows(self):
        raise NotImplementedError("Must be implemented in child classes")

    @abstractmethod
    def shuffle(self):
        raise NotImplementedError("Must be implemented in child classes")

    @abstractmethod
    def iter_batches(self, batch_size: int):
        raise NotImplementedError("Must be implemented in child classes")


class ParquetDataIterator(RawDataIterator):
    """Data iterator class for parquet files

    Parameters
    ----------
    list_of_file_paths : List[str]
        the list of file paths to read from
    """

    def iterrows(self):
        assert self.list_of_file_paths is not None, "list_of_file_paths is not set"
        self.df = dd.read_parquet(  # type: ignore
            self.list_of_file_paths,
            engine="pyarrow",
        )
        for row in self.df.iterrows():
            yield row[1]

    def shuffle(self) -> RawDataIterator:
        self.df = self.df.sample(frac=1)
        return self

    def get_file_suffix(self) -> str:
        return "parquet"


class AvroDataIterator(RawDataIterator):
    """Data iterator class for avro files

    Parameters
    ----------
    list_of_file_paths : List[str]
        the list of file paths to read from
    """

    def iterrows(self):
        assert self.list_of_file_paths is not None, "list_of_file_paths is not set"
        for file_path in self.list_of_file_paths:
            with fsspec.open(file_path, "rb") as f:
                reader = fastavro.reader(f)
                for records in reader:
                    yield records

    def shuffle(self) -> RawDataIterator:
        random.shuffle(self.list_of_file_paths)  # type: ignore
        return self

    def get_file_suffix(self) -> str:
        return "avro"


class TFRecordIterator(RawDataIterator):
    """Data iterator class for tfrecord files

    Parameters
    ----------
    list_of_file_paths : List[str]
        the list of file paths to read from
    read_from_gcs : bool
        whether to read from gcs or local storage
    """

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.feature_description = None

    def initialize_feature_description(self, raw_dataset: tf.data.TFRecordDataset):
        """
        If the feature description is not set, infer the feature description from the first record in the dataset.
        """
        if self.feature_description is None:
            sample_record: tf.Tensor = next(iter(raw_dataset))  # type: ignore

            self.feature_description = self.infer_feature_type(
                self.parse_tfrecord(sample_record).features.feature  # type: ignore
            )

    def iterrows(self):
        assert self.list_of_file_paths is not None, "list_of_file_paths is not set"
        for file_path in self.list_of_file_paths:
            raw_dataset = tf.data.TFRecordDataset([file_path], compression_type="GZIP")
            self.initialize_feature_description(raw_dataset)

            for raw_record in raw_dataset:
                example = tf.io.parse_single_example(
                    raw_record, self.feature_description
                )
                yield example

    def iter_batches(self, batch_size: int) -> Dict[str, tf.Tensor]:  # type: ignore
        assert self.list_of_file_paths is not None, "list_of_file_paths is not set"
        for file_path in self.list_of_file_paths:
            raw_dataset = tf.data.TFRecordDataset([file_path], compression_type="GZIP")
            self.initialize_feature_description(raw_dataset)
            # to avoid the issues with tf record warnings, we drop the last instances
            for batch in raw_dataset.batch(batch_size, drop_remainder=True).prefetch(
                buffer_size=tf.data.AUTOTUNE
            ):
                example = tf.io.parse_example(batch, self.feature_description)
                yield example

    def iter_batches(self, batch_size: int) -> Dict[str, tf.Tensor]:  # type: ignore
        assert self.list_of_file_paths is not None, "list_of_file_paths is not set"
        for file_path in self.list_of_file_paths:
            raw_dataset = tf.data.TFRecordDataset([file_path], compression_type="GZIP")
            self.initialize_feature_description(raw_dataset)
            # to avoid the issues with tf record warnings, we drop the last instances
            for batch in raw_dataset.batch(batch_size, drop_remainder=True).prefetch(
                buffer_size=tf.data.AUTOTUNE
            ):
                example = tf.io.parse_example(batch, self.feature_description)
                yield example

    # dynamic inferring the feature description of tfrecord files
    def infer_feature_type(self, example_proto: tf.Tensor) -> dict:
        feature_description = {}
        for key, value in example_proto.items():  # type: ignore
            if isinstance(value, tf.train.Feature):
                if value.HasField("bytes_list"):
                    feature_description[key] = tf.io.VarLenFeature(tf.string)
                elif value.HasField("float_list"):
                    feature_description[key] = tf.io.VarLenFeature(tf.float32)
                elif value.HasField("int64_list"):
                    feature_description[key] = tf.io.VarLenFeature(tf.int64)
                else:
                    raise ValueError("Unknown feature type")
        return feature_description

    # parsing the tfrecord files from bytes
    def parse_tfrecord(self, record: tf.Tensor) -> tf.Tensor:
        example = tf.train.Example()
        example.ParseFromString(record.numpy())  # type: ignore
        return example

    def shuffle(self) -> RawDataIterator:
        random.shuffle(self.list_of_file_paths)  # type: ignore
        return self

    def get_file_suffix(self) -> str:
        return "tfrecord.gz"
