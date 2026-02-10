from src.engine.player import EnginePlayer
from src.engine.evaluator.LinearEvaluator import LinearEvaluator
from src.engine.evaluator.NeuralEvaluator import NeuralEvaluator
from src.neat.serialization import load_champion as load_neat_champion
from src.cma.serialization import load_champion as load_cma_champion
from src.utils.feature_stats import load_or_compute_feature_stats
from src.utils.visual_game import VisualGame

DEPTH = 2


def main():
    mean, std = load_or_compute_feature_stats()

    # CMA-ES champion
    weights = load_cma_champion()
    cma_eval = LinearEvaluator(weights)
    cma_player = EnginePlayer(cma_eval, depth=DEPTH)

    # NEAT champion
    genome = load_neat_champion()
    neat_eval = NeuralEvaluator(genome, mean, std)
    neat_player = EnginePlayer(neat_eval, depth=DEPTH)

    print("Starting CMA-ES vs NEAT match...\n")
    game = VisualGame(
        white=cma_player,
        black=neat_player,
        delay=0.5,
        title="CMA-ES (White) vs NEAT (Black)",
    )
    game.play()


if __name__ == "__main__":
    main()
