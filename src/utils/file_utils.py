import json
import logging
from typing import BinaryIO, List

from fsspec.core import url_to_fs


def get_file_size(file_path: str) -> int:
    fs, _ = url_to_fs(file_path)
    return fs.size(file_path)


def copy_to_remote(local_path: str, remote_path: str, recursive: bool = True) -> None:
    logging.info(f"Copying {local_path} to {remote_path}")
    fs, _ = url_to_fs(remote_path)
    fs.put(local_path, remote_path, recursive=recursive)
    logging.info(f"Finished copying {local_path} to {remote_path}")


def open_local_or_remote(file_path: str, mode: str = "r") -> BinaryIO:
    fs, _ = url_to_fs(file_path)
    return fs.open(file_path, mode)


def load_json(file_path: str) -> dict:
    with open_local_or_remote(file_path, "r") as f:
        feature_map = json.load(f)
    return feature_map


def list_files(
    folder_path: str,
    suffix: str = "*",
    # if should_update_prefix is True, adds the prefix based on the filesystem,
    # otherwise returns the path generated by glob
    should_update_prefix: bool = True,
) -> List[str]:

    # We remove trailing slashes to avoid double slashes in the path
    folder_path = folder_path.removesuffix("/")

    fs, _ = url_to_fs(folder_path)
    return (
        # add the prefix for gcs.
        [f"{fs.protocol[0]}://{x}" for x in fs.glob(f"{folder_path}/{suffix}")]
        if should_update_prefix
        else fs.glob(f"{folder_path}/{suffix}")
    )
