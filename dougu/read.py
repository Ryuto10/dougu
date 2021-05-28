import gzip
import json
from typing import Generator


def read_line(file_path: str) -> Generator[str]:
    """ファイルの行を読み込みます。gzipで圧縮したファイルも読み込みます。

    Args:
        file_path (str): 読み込むfileまでのpath (.gz, .gzip も対応)

    Yields:
        Generator[str]: 1行ごとに吐き出すジェネレータ関数
    """
    fi = gzip.open(file_path, "rt", "utf-8") if is_gzip(file_path) else open(file_path)

    for line in fi:
        yield line.rstrip("\n")

    fi.close()


def is_gzip(file_path: str) -> bool:
    """拡張子でfileがgzipファイルかどうか判定します
    Args:
        file_path (str): Path to input file

    Returns:
        bool: Whether the file is a gzip file or not
    """

    if file_path.endswith(".gzip") or file_path.endswith(".gz"):
        return True
    else:
        return False


def read_jsonl(file_path: str) -> Generator[str]:
    """jsonl形式のファイルを読み込みます。gzipで圧縮したファイルも読み込みます。

    Args:
        file_path (str): 読み込むfileまでのpath (.gz, .gzip も対応)

    Yields:
        Generator[str]: 1行ごとに吐き出すジェネレータ関数
    """
    for line in read_line(file_path):
        if not line:
            continue
        yield json.loads(line)


def count_file_length(file_path: str) -> int:
    """Count the number of lines in the file"""
    for idx, _ in enumerate(read_line(file_path), 1):
        pass
    return idx
