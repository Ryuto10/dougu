import unittest
from pathlib import Path

from dougu.read import count_file_length, read_jsonl, read_line, read_yaml


class TestReadLine(unittest.TestCase):
    def test_read_line_without_newline(self) -> None:
        sample_file = str(Path(__file__).resolve().parent / "samples" / "sample.txt")
        actual = [line for line in read_line(sample_file)]
        expected = ["test", "sample"]
        self.assertEqual(actual, expected)


class TestReadJsonl(unittest.TestCase):
    def test_read_json_line_as_dict(self) -> None:
        sample_file = str(Path(__file__).resolve().parent / "samples" / "sample.jsonl")
        actual = [line for line in read_jsonl(sample_file)]
        expected = [{"name": "test", "value": 1}, {"name": "sample", "value": 2}]
        self.assertEqual(actual, expected)


class TestCountFileLength(unittest.TestCase):
    def test_count_lines(self) -> None:
        sample_file = str(Path(__file__).resolve().parent / "samples" / "sample.txt")
        actual = count_file_length(sample_file)
        expected = 2
        self.assertEqual(actual, expected)


class TestReadYaml(unittest.TestCase):
    def test_read_yaml(self) -> None:
        sample_file = str(Path(__file__).resolve().parent / "samples" / "sample.yaml")
        actual = read_yaml(sample_file)
        expected = {"sample": {"value": 1, "items": ["a", 1.23, "c"]}}
        self.assertEqual(actual, expected)
