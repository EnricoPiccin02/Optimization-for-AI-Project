from src.engine.baseline import make_baseline_opponent
from src.neat.fitness import evaluate_genome
from src.neat.serialization import load_champion
from src.utils.feature_stats import load_or_compute_feature_stats


def evaluate_neat_champion(num_games=100):
    """
    High-confidence evaluation of the NEAT champion
    against a frozen baseline opponent.
    """
    mean, std = load_or_compute_feature_stats()
    baseline = make_baseline_opponent(mean, std)

    champion = load_champion()

    score = evaluate_genome(
        genome=champion,
        mean=mean,
        std=std,
        baseline=baseline,
        games=num_games,
    )

    print("=" * 60)
    print("NEAT Champion evaluation")
    print(f"Games played     : {num_games}")
    print(f"Win rate         : {score:.3f}")
    print("Baseline expected: 0.500")
    print("=" * 60)

    return score
