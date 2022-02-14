import subprocess
import unittest

from dougu.timer import get_current_time, timer


class TestTimer:
    def test_timer(self) -> None:
        with timer(name="test timer"):
            for _ in range(1000):
                pass


class TestGetTime(unittest.TestCase):
    def test_get_time(self) -> None:
        command = "TZ=JST-9 date +'%Y/%m/%d %H:%M:%S'"
        expected = (
            subprocess.check_output(command, shell=True).decode("utf-8").rstrip("\n")
        )
        actual = get_current_time(timezone="Asia/Tokyo", readable=True)
        self.assertEqual(actual, expected)

        command = "TZ=JST-9 date +'%m%d%H%M'"
        expected = (
            subprocess.check_output(command, shell=True).decode("utf-8").rstrip("\n")
        )
        actual = get_current_time(timezone="Asia/Tokyo", readable=False)
        self.assertEqual(actual, expected)
