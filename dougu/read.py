import gzip

import json


def read_file(file_path: str):
    if file_path.endswith(".gzip") or file_path.endswith(".gz"):
        with gzip.open(file_path, "rt", "utf_8") as fi:
            for line in fi:
                if not line.strip():
                    continue
                yield line.rstrip("\n")

    else:
        with open(file_path) as fi:
            for line in fi:
                if not line.strip():
                    continue
                yield line.rstrip("\n")


def read_jsonl(file_path: str):
    with open(file_path) as fi:
        for line in fi:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def count_file_length(file):
    with open(file) as fi:
        for idx, _ in enumerate(fi, 1):
            pass
    return idx
