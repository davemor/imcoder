import logging
import time
from typing import Any


def log_timings(func) -> Any:
    def wrapper(*args, **kwargs):
        logging.info(f"Running {func.__name__}")
        start, proc_start = time.time(), time.process_time()
        func(*args, **kwargs)
        delta, proc_delta = time.time() - start, time.process_time() - proc_start
        logging.info(f"{func.__name__} complete.")
        logging.info(f"{delta:.2f}s total, {proc_delta:.2f}s on the CPU.")

    return wrapper
