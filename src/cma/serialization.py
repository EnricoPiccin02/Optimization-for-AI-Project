import numpy as np
import os

from src.utils.paths import CMA_CHAMPION_PATH


def save_champion(weights, path=CMA_CHAMPION_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, weights)


def load_champion(path=CMA_CHAMPION_PATH):
    return np.load(path)
