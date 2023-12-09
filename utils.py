import time
from typing import List, Optional, Union

from logger_config import logger as LOGGER


def select_day(args, day):
    if args.day == 0:
        return True
    elif args.day == day:
        return True
    else:
        return False


def load_file(filename):
    with open(filename) as f:
        lines = [line.rstrip("\n") for line in f]
    return lines


class Timer:
    def __init__(self, name, logger=None, log_level=0):
        self.name = name
        self.logger = LOGGER
        self.log_level = True

    def _log(self, message):
        if self.logger:
            self.logger.info(message)
        else:
            print(message)

    def __enter__(self):
        self.start_time = time.perf_counter()
        if "part" in str.lower(self.name):
            self._log(f"⭐ {self.name}: Timer started.")
        else:
            self._log(f"{self.name}: Timer started.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.perf_counter() - self.start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        # Formatting time to HH:MM:SS.mmm (milliseconds)
        elapsed_time_str = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}.{int((seconds - int(seconds)) * 1000):03}"

        if exc_type is not None:
            self._log(f"⚠️ {self.name}: Timer stopped with an exception: {exc_val}")
        else:
            self._log(
                f"⏱️ Elapsed time for {self.name}: {elapsed_time_str} (HH:MM:SS.mmm)"
            )
        return False  # Do not suppress exceptions

    def get_elapsed_time(self):
        return time.perf_counter() - self.start_time
