import gzip

import json
from logzero import logger


def read_file(file_path: str):
    """Read file (text file or gzip file)"""
    if file_path.endswith(".gzip") or file_path.endswith(".gz"):
        with gzip.open(file_path, "rt", "utf_8") as fi:
            for line in fi:
                yield line.rstrip("\n")
    else:
        with open(file_path) as fi:
            for line in fi:
                yield line.rstrip("\n")


def read_jsonl(file_path: str):
    """Read file whose format per line is JSON"""
    with open(file_path) as fi:
        for line in fi:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def count_file_length(file: str) -> int:
    """Count the number of lines in the file"""
    for idx, _ in enumerate(read_file(file), 1):
        pass
    return idx

