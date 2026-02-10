from src.engine.player import EnginePlayer
from src.engine.evaluator.NeuralEvaluator import NeuralEvaluator
from src.evolution.game_fitness import evaluate_player

def evaluate_genome(genome, mean, std, baseline, games):
    evaluator = NeuralEvaluator(genome, mean, std)
    player = EnginePlayer(evaluator, depth=2)
    return evaluate_player(player, baseline, games)
