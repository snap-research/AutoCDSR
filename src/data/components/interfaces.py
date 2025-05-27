from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from torch.utils.data import IterableDataset

from src.data.components.iterators import RawDataIterator


@dataclass
class DatasetConfig:
    """The dataset configuration class used to store the dataset configuration.

    Parameters:
    ----------
    user_id_field: str
        The user id field name.
    data_iterator: RawDataIterator
        The raw data iterator.
    preprocessing_functions: list[callable]
        The list of preprocessing functions. Should be in the order they must be applied.
    num_placeholder_tokens_map: Optional[dict]
        The number of placeholder tokens map.
    field_type_map: Optional[dict]
        The field type map.
    min_sequence_length: int
        The minimum sequence length. Only works if iterating per row.
    feature_map: Optional[dict]
        maps the feature names to the desired feature names.
    iterate_per_row: bool
        Whether to iterate per row or per batches.
    features_to_consider: list[str]
        List of features to consider. If not specified, consider all features.
    """

    user_id_field: str
    data_iterator: RawDataIterator
    preprocessing_functions: list[callable]  # type: ignore
    iterate_per_row: bool = False
    num_placeholder_tokens_map: Optional[dict] = field(default_factory=dict)
    field_type_map: Optional[dict] = field(default_factory=dict)
    min_sequence_length: int = 10
    feature_map: Optional[dict] = None
    features_to_consider: list[str] = field(default_factory=list)


@dataclass
class DataloaderConfig:
    """The dataloader configuration class used to store the dataloader configuration.

    Each instance of this class is run on one device.

    Parameters:
    ----------
    dataset_class: IterableDataset
        The dataset class.
    data_folder: str
        Path to the folder containingthe dataset files.
    dataset_config: DatasetConfig
        The dataset configuration.
    labels: Dict[str, callable]
        A dictionary mapping from feature names to
    batch_size_per_device: list[callable]
        The batch size per dataloader, also per device (GPU).
    num_workers: int
        The number of workers per dataloader, also per device (GPU).
    assign_files_by_size: Optional[dict]
        Whether to assign files to workers by file size to balance computation
        across workers.
    oov_token: Optional[int]
        The token used to represent OOV items.
    masking_token: int
        The token used to represent masked items.
    collate_fn: callable
        Collate function used to construct batches.
    sequence_length: int = 200
        The length of sequences the dataloader should return. If raw sequences
        are shorter, the dataloader will pad them to reach sequence_length.
    padding_token: int = 0
        The token used for padding sequences.
    drop_last: bool = True
        Whether to drop the last batch if it is smaller than
        batch_size_per_device.
    pin_memory: bool = True
        Whether to allocate memory on CPU to ensure data is always available for
        fast transfer to GPU.
    should_shuffle_rows: bool = False
        Whether to shuffle rows between epochs.
    persistent_workers: bool = False
        Whether to maintain worker processes across epochs.
    """

    dataset_class: IterableDataset
    data_folder: str
    dataset_config: DatasetConfig
    labels: Dict[str, callable]  # type: ignore
    batch_size_per_device: int
    num_workers: int
    assign_files_by_size: bool
    oov_token: Optional[int]
    masking_token: int
    collate_fn: callable  # type: ignore
    sequence_length: int = 200
    padding_token: int = 0
    drop_last: bool = True
    pin_memory: bool = True
    should_shuffle_rows: bool = False
    persistent_workers: bool = False


@dataclass
class LabelFunctionOutput:
    """class to unify the output of label functions, making it easier to merge those into SequentialModelInputData and SequentialModuleLabelData"""

    sequence: torch.Tensor
    labels: torch.Tensor
    label_location: torch.Tensor = None


@dataclass
class SequentialModuleLabelData:
    """The label data class used to wrap the label data for training/testing.

    Parameters
    ----------
    labels: Dict[str, torch.Tensor]
        Dictionary of label_name to label tensor.
        Label tensor is the shape of mask size # long tensor
    label_location: Dict[str, torch.Tensor]
        Dictionary of label_name to label location tensor.
        Label location tensor is the shape of mask_size, 2 as it contains coordinates # long tensor
    """

    labels: Dict[str, torch.Tensor] = field(default_factory=dict)
    label_location: Dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class SequentialModelInputData:
    """The model input data class used to wrap the model input data for training/testing

    Parameters
    ----------
    transformed_sequences: Dict[str, torch.Tensor]
        Dictionary of sequence_name to sequence tensor.
        Sequence tensor is (batch_size_per_device x sequence length)
    mask: torch.Tensor
        The mask for the sequence data.
        (batch_size_per_device x sequence length)
    """

    transformed_sequences: Dict[str, torch.Tensor] = field(default_factory=dict)
    mask: torch.Tensor = (
        None  # Single mask if needed as all sequences are padded the same way.
    )
