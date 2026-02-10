from src.engine.player import EnginePlayer
from src.engine.evaluator.NormalizedLinearEvaluator import NormalizedLinearEvaluator
from src.evolution.game_fitness import evaluate_player


def fitness(weights, mean, std, opponent, num_games):
    evaluator = NormalizedLinearEvaluator(weights, mean, std)
    player = EnginePlayer(evaluator, depth=2)
    return evaluate_player(player, opponent, num_games)
