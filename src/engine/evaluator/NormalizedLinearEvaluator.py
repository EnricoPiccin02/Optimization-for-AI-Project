import numpy as np
from src.engine.features import FeatureExtractor


class NormalizedLinearEvaluator:
    def __init__(self, weights, mean, std):
        self.weights = weights
        self.mean = mean
        self.std = std
        self.extractor = FeatureExtractor()

    def evaluate(self, board):
        raw = self.extractor.extract(board)
        norm = (raw - self.mean) / self.std
        return float(np.dot(self.weights, norm))
