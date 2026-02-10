import numpy as np
from src.engine.features import FeatureExtractor


class LinearEvaluator:
    def __init__(self, weights: np.ndarray):
        self.weights = weights
        self.extractor = FeatureExtractor()

    def evaluate(self, board) -> float:
        features = self.extractor.extract(board)
        return float(np.dot(self.weights, features))
