import csv
import os
from typing import List


class CSVLogger:
    """
    CSV logger for optimisation experiments.
    """

    def __init__(self, path: str, headers: List[str]):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(self.path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def log(self, values: List):
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(values)
