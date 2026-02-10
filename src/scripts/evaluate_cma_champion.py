from src.cma.fitness import fitness
from src.engine.baseline import make_baseline_opponent
from src.cma.serialization import load_champion


def evaluate_cma_champion(mean, std, num_games=100):
    """
    High-confidence evaluation of the CMA-ES champion
    against a frozen baseline opponent.
    """
    weights = load_champion()

    baseline = make_baseline_opponent(mean, std)

    score = fitness(
        weights=weights,
        mean=mean,
        std=std,
        opponent=baseline,
        num_games=num_games,
    )

    print("=" * 60)
    print("CMA-ES Champion evaluation")
    print(f"Games played     : {num_games}")
    print(f"Win rate         : {score:.3f}")
    print("Baseline expected: 0.500")
    print("=" * 60)

    return score
