import time
from contextlib import contextmanager
from datetime import datetime
from typing import Generator

from dateutil import tz
from logzero import logger


def get_current_time(timezone: str = "Asia/Tokyo", readable: bool = True) -> str:
    """
    Args:
        timezone: Timezone (e.g. "UTC", "Asia/Tokyo")
        readable: If true, returns the readable time

    Returns:
        current_time: Current time

    """
    now = datetime.now(tz.gettz(timezone))
    time_format = "%Y/%m/%d %H:%M:%S" if readable else "%m%d%H%M"
    current_time = now.strftime(time_format)

    return current_time


@contextmanager
def timer(name: str) -> Generator[None, None, None]:
    """Measure the execution time
    Args:
        name: A name representative of the measurement target

    Usage:
        ```
        with timer(name="train the model"):
            # Process for which you want to measure the execution time
            loss = trainer.train()
            loss.backward()
            ...
        ```
    """
    start_time = time.time()
    logger.info(f"{name}: start")
    yield
    end_time = time.time()

    logger.info(f"{name}: elapsed time is {end_time - start_time:.3f}")
