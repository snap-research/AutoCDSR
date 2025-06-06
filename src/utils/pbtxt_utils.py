from collections import deque
from typing import Any, Dict, List, Union

from src.utils.file_utils import open_local_or_remote


def extract_vocabulary_cardinality_from_pbtxt(pbtxt_file_path: str) -> Dict[str, int]:
    """
    This function assumes that categorical-id features follow the follow format, default to TFT:
    feature {
      name: "feature_name"
      type: "INT"
      int_domain {
        min: 0
        max: max_idx
        is_categorical: true
      }
    ... other fields
    }

    It extracts the vocabulary cardinality from the pbtxt file and returns a dictionary
    with the feature name as key and the cardinality as value.
    """
    features = parse_pbtxt_file(pbtxt_file_path, return_dict=True)
    vocab = {}
    for feature_name, feature in features.items():
        if "int_domain" in feature:
            # We add one since the vocabulary starts at 0
            vocab[feature_name] = int(feature["int_domain"]["max"]) + 1
    return vocab


def get_vocabulary_for_feature_from_pbtxt(
    pbtxt_file_path: str, feature_name: str, placeholder_token_buffer: int = 0
) -> int:
    return (
        extract_vocabulary_cardinality_from_pbtxt(pbtxt_file_path)[feature_name]
        + placeholder_token_buffer
    )


def parse_pbtxt_file(
    file_path: str, return_dict=False
) -> Union[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Function to parse pbtxt file without needing tensorflow. It assumes the file was generated by the transform_fn from TFT.
    It returns a list of dictionaries or a dictionary of dictionaries with the name of the feature as key.
    """
    with open_local_or_remote(file_path, "r") as f:
        pbtxt_lines = f.readlines()
    features = []
    stack = deque()
    stack.append({})
    for line in pbtxt_lines:
        if line.startswith("feature"):
            if not "{" in line:
                raise ValueError(
                    "Invalid pbtxt file. We expect files generated by TFT transform_fn where feature is followed by {"
                )
            current_dict = stack.pop()
        elif "{" in line:
            value_name = line.split()[0].strip()
            current_dict[value_name] = {}
            stack.append(current_dict)
            current_dict = current_dict[value_name]
        elif "}" in line:
            if len(stack):
                current_dict = stack.pop()
            else:
                features.append(current_dict)
                stack.append({})
        elif ":" in line:
            parameter, value = line.split(": ")
            current_dict[parameter.strip()] = value.strip().replace('"', "")
    if return_dict:
        features = {f["name"]: f for f in features}
    return features
