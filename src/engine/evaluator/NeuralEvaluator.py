from src.neat.network import NeuralNetwork
from src.engine.features import FeatureExtractor


class NeuralEvaluator:
    def __init__(self, genome, mean, std):
        self.net = NeuralNetwork(genome)
        self.mean = mean
        self.std = std
        self.extractor = FeatureExtractor()

    def evaluate(self, board):
        raw = self.extractor.extract(board)
        norm = (raw - self.mean) / self.std
        return float(self.net.forward(norm))
