from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.nn.utils.rnn import pad_sequence

from src.data.components.interfaces import (
    LabelFunctionOutput,
    SequentialModelInputData,
    SequentialModuleLabelData,
)
from src.data.utils import combine_list_of_tensor_dicts, pad_or_trim_sequence


def collate_fn_train(
    # batch can be a list or a dict
    batch: Union[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]],
    labels: Dict[str, callable],  # type: ignore
    sequence_length: int = 200,
    masking_token: int = 1,
    padding_token: int = 0,
    oov_token: Optional[
        int
    ] = None,  # If oov_token is passed, we remove it from the sequence
) -> Tuple[SequentialModelInputData, SequentialModuleLabelData]:
    """The collate function passed to dataloader. It can do training masking and padding for the input sequence.

    Parameters
    ----------
    batch : Union[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]
        The batch of data to be collated. Can be a list of dictionaries, in the case we were
        loading the data per row, or a dictionary of tensors, in the case we were loading the data per batch.
    labels : List[Dict[str, callable]]
        The list of functions to apply to generate the labels. If
    """
    if isinstance(batch, list):
        batch = combine_list_of_tensor_dicts(batch)  # type: ignore
    model_input_data = SequentialModelInputData()
    model_label_data = SequentialModuleLabelData()

    for field_name, field_sequence in batch.items():  # type: ignore
        current_sequence = field_sequence  # type: ignore
        if oov_token:
            current_sequence = [
                sequence[sequence != oov_token] for sequence in field_sequence
            ]
        # 1. in-batch padding s.t. all sequences have the same length and in the format of pt tensor
        current_sequence = pad_sequence(
            current_sequence, batch_first=True, padding_value=padding_token
        )

        # 2. padding or trimming the sequence to the desired length for training
        current_sequence = pad_or_trim_sequence(
            padded_sequence=current_sequence,
            sequence_length=sequence_length,
            padding_token=padding_token,
        )
        # Currently supports a single masking per sequence
        if model_input_data.mask is None:
            model_input_data.mask = (current_sequence != padding_token).long()

        # creating labels if the field is in the labels list
        if field_name in labels:
            label_function = labels[field_name].transform
            label_function_output: LabelFunctionOutput = label_function.transform_label(
                sequence=current_sequence,
                padding_token=padding_token,
                masking_token=masking_token,
            )
            model_label_data.labels[field_name] = label_function_output.labels
            model_label_data.label_location[
                field_name
            ] = label_function_output.label_location
            model_input_data.transformed_sequences[
                field_name
            ] = label_function_output.sequence
        else:
            model_input_data.transformed_sequences[field_name] = current_sequence
    return model_input_data, model_label_data  # type: ignore
