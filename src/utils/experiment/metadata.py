import json
import os
from typing import Dict, Any


class ExperimentMetadata:
    """
    Experiment metadata container.
    """

    def __init__(self, algorithm: str, parameters: Dict[str, Any]):
        self.algorithm = algorithm
        self.parameters = parameters

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        payload = {
            "algorithm": self.algorithm,
            "parameters": self.parameters,
        }

        with open(path, "w") as f:
            json.dump(payload, f, indent=4)
