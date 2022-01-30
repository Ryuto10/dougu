import gzip
import json
from pathlib import Path
from typing import Any, Dict, Generator

import yaml
from logzero import logger
from tqdm import tqdm


def read_line(file_path: str, progress: bool = True) -> Generator[str, None, None]:
    """Read lines from a file

    Args:
        file_path: Path to a file (.txt, .txt.gzip, .txt.gz)
        progress:  If true, displays the progress of loading the file

    Yields:
        line: Read line

    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"'{file_path}' is not found")

    is_gzip = (
        True if file_path.endswith(".gzip") or file_path.endswith(".gz") else False
    )
    fi = (
        gzip.open(file_path, mode="rt", encoding="utf-8")
        if is_gzip
        else open(file_path)
    )
    loader = tqdm(fi) if progress else fi

    for line in loader:
        yield line.rstrip("\n")
    fi.close()


def read_jsonl(file_path: str) -> Generator[str, None, None]:
    """Read lines from json format file.

    Args:
        file_path: Path to a file (.jsonl, .jsonl.gzip, .jsonl.gz)

    Yields:
        line: Read line

    """
    for line in read_line(file_path):
        if not line:
            continue
        yield json.loads(line)


def count_file_length(file_path: str) -> int:
    """Count the number of lines in the file

    Args:
        file_path: Path to a file

    """
    idx = 0
    for idx, _ in enumerate(read_line(file_path), 1):
        pass
    return idx


def read_yaml(file_path: str) -> Dict[str, Any]:
    """Read a yaml format file

    Args:
        file_path: Path to a yaml file

    Returns:
        yaml_dict: Loaded yaml file

    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"yaml file is not found: {file_path}")

    logger.debug(f"load: {Path(file_path).absolute()}")
    yaml_dict = yaml.safe_load(open(file_path))
    logger.debug("done")

    return yaml_dict
