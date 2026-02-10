import numpy as np
from src.engine.evaluator.NormalizedLinearEvaluator import NormalizedLinearEvaluator
from src.engine.player import EnginePlayer


def make_baseline_opponent(mean, std, depth=2):
    """
    Returns a frozen baseline opponent with unit weights.
    """
    weights = np.ones(len(mean), dtype=np.float32)
    evaluator = NormalizedLinearEvaluator(weights, mean, std)
    return EnginePlayer(evaluator, depth=depth)
