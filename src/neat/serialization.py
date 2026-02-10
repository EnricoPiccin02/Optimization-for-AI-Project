import os
import pickle

from src.utils.paths import NEAT_CHAMPION_PATH


def save_champion(genome, path=NEAT_CHAMPION_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(genome, f)


def load_champion(path=NEAT_CHAMPION_PATH):
    with open(path, "rb") as f:
        genome = pickle.load(f)
    return genome
