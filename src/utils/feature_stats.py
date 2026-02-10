import os
import numpy as np
import random
import chess

from src.engine.features import FeatureExtractor
from src.utils.paths import FEATURE_MEAN_PATH, FEATURE_STD_PATH


def compute_feature_stats(num_positions=1000):
    """
    Freeze feature statistics to define a fixed input distribution
    for all optimisation algorithms (CMA-ES, NEAT).
    """
    extractor = FeatureExtractor()
    features = []

    board = chess.Board()

    for _ in range(num_positions):
        if board.is_game_over():
            board.reset()

        move = random.choice(list(board.legal_moves))
        board.push(move)
        features.append(extractor.extract(board))

    X = np.vstack(features)
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8  # numerical stability

    return mean, std


def load_or_compute_feature_stats(num_positions=1000):
    """
    Freeze feature statistics.
    If statistics exist, load them.
    Otherwise, compute once and persist them.
    """
    if os.path.exists(FEATURE_MEAN_PATH) and os.path.exists(FEATURE_STD_PATH):
        mean = np.load(FEATURE_MEAN_PATH)
        std = np.load(FEATURE_STD_PATH)
        return mean, std

    os.makedirs(os.path.dirname(FEATURE_MEAN_PATH), exist_ok=True)

    mean, std = compute_feature_stats(num_positions)

    np.save(FEATURE_MEAN_PATH, mean)
    np.save(FEATURE_STD_PATH, std)

    return mean, std
