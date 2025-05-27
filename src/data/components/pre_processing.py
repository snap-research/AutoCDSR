from typing import Dict, List, Optional

import numpy as np
import tensorflow as tf
import torch

from src.data.components.interfaces import DatasetConfig

# support functions


def is_feature_in_features_to_apply(features_to_apply: List[str], k: str) -> bool:
    if len(features_to_apply) > 0 and k not in features_to_apply:
        return False
    return True


# All pre_processing functions must have the following signature:
# def my_pre_processing_function(batch_or_row : Dict[str, Any],  dataset_config: DatasetConfig, features_to_apply: Optional[List[str]]=[], **kwargs) -> Any:
# If the function only works for batch or for a row, make it explicit in the documentation and/or function name


def filter_features_to_consider(
    batch_or_row: Dict[str, tf.Tensor],
    dataset_config: DatasetConfig,
    features_to_apply: Optional[List[str]] = [],
    **kwargs
) -> Dict[str, tf.Tensor]:
    batch_or_row = map_feature_names(batch_or_row, dataset_config)
    if len(dataset_config.features_to_consider):
        # Given a batch or row, filter the features to consider.
        return {
            k: v
            for k, v in batch_or_row.items()
            if k in dataset_config.features_to_consider
        }
    # if not specified, we consider all features
    return batch_or_row


def convert_to_dense_numpy_array(
    batch_or_row: Dict[str, tf.Tensor],
    dataset_config: DatasetConfig,
    features_to_apply: Optional[List[str]] = [],
    **kwargs
) -> Dict[str, np.ndarray]:
    # Transform a tfrecord example to a dictionary of numpy arrays, converting sparse tensors to dense numpy arrays.

    for k in batch_or_row:
        if is_feature_in_features_to_apply(features_to_apply, k):
            batch_or_row[k] = tf.sparse.to_dense(batch_or_row[k]).numpy()
    return batch_or_row


def map_feature_names(
    batch_or_row: Dict[str, np.ndarray],
    dataset_config: DatasetConfig,
    features_to_apply: Optional[List[str]] = [],
    **kwargs
) -> Dict[str, np.ndarray]:
    # Given a batch or row, map the feature names to the desired feature names.
    if dataset_config.feature_map:
        batch_or_row = {
            v: batch_or_row[k]
            for k, v in dataset_config.feature_map.items()
            if k in batch_or_row
        }
    return batch_or_row


def convert_fields_to_tensors(
    batch_or_row: Dict[str, np.ndarray],
    dataset_config: DatasetConfig,
    features_to_apply: Optional[List[str]] = [],
    **kwargs
) -> Dict[str, torch.Tensor]:
    # Given a batch or row, convert all fields to torch tensors. Uses the field type map to determine the dtype, defaulting to torch.long
    # if no dtype is specified.
    for k, v in batch_or_row.items():
        if is_feature_in_features_to_apply(features_to_apply, k):
            batch_or_row[k] = torch.tensor(v, dtype=dataset_config.field_type_map.get(k, torch.long))  # type: ignore
    return batch_or_row


def add_placeholder_tokens(
    batch_or_row: Dict[str, torch.Tensor],
    dataset_config: DatasetConfig,
    features_to_apply: Optional[List[str]] = [],
    **kwargs
) -> Dict[str, torch.Tensor]:
    # Given a batch or row of torch Tensors, add the number of placeholder tokens based on map of placeholder tokens.
    # If not specified, defaults to 0. If the value is 0, the field is not modified as it is
    for k, v in batch_or_row.items():
        if is_feature_in_features_to_apply(features_to_apply, k):
            num_placeholder_tokens = dataset_config.num_placeholder_tokens_map.get(k, 0)
            if num_placeholder_tokens > 0:
                # We do not modify the 0 index as it is used for padding
                non_zero_mask = (v > 0).long()
                summing_mask = (
                    non_zero_mask * dataset_config.num_placeholder_tokens_map.get(k, 0)
                )
                batch_or_row[k] = v + summing_mask
    return batch_or_row


## Row only


def remove_oov_tokens(row: Dict[str, torch.Tensor], dataset_config: DatasetConfig, features_to_apply: Optional[List[str]] = [], **kwargs) -> Dict[str, torch.Tensor]:  # type: ignore
    # Given a row, remove the oov tokens from the row.
    oov_token = kwargs.get("oov_token", -1)
    # We only need to access the first element of the values, so we use next(iter()) to speed it up
    mask = torch.ones_like(next(iter(row.values())), dtype=torch.bool)
    for k, v in row.items():
        if is_feature_in_features_to_apply(features_to_apply, k):
            mask = mask & (v != oov_token)
    for k, v in row.items():
        if is_feature_in_features_to_apply(features_to_apply, k):
            row[k] = v[mask]
    return row


def filter_sequence_length_row(row: Dict[str, torch.Tensor], dataset_config: DatasetConfig, features_to_apply: Optional[List[str]] = [], **kwargs) -> Dict[str, np.ndarray]:  # type: ignore
    # Only works for a row right now. This filters out rows that have fields with sequence length smaller than the min threshold.
    for _, tensor in row.items():
        if len(tensor) < dataset_config.min_sequence_length:
            return None
    return row


def filter_only_oov(row: Dict[str, torch.Tensor], dataset_config: DatasetConfig, features_to_apply: Optional[List[str]] = [], **kwargs) -> Dict[str, np.ndarray]:  # type: ignore
    # Only works for a row right now. This filters out rows that have fields with only the oov token.
    oov_token = kwargs.get("oov_token", -1)
    for _, tensor in row.items():
        if (tensor == oov_token).all():
            return None
    return row
