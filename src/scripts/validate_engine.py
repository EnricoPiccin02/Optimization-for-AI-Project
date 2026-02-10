import numpy as np
import random
from collections import Counter

from src.engine.evaluator.LinearEvaluator import LinearEvaluator
from src.engine.player import EnginePlayer
from src.utils.game_loop import play_game
from src.utils.results import result_to_score


N_GAMES = 50
DEPTH = 2
N_FEATURES = 11
RANDOM_SEED = 42


def validate():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    weights = np.ones(N_FEATURES, dtype=np.float32)
    evaluator = LinearEvaluator(weights)

    scores = []
    results = []

    for i in range(N_GAMES):
        if i % 2 == 0:
            white = EnginePlayer(evaluator, depth=DEPTH)
            black = EnginePlayer(evaluator, depth=DEPTH)
            result = play_game(white, black)
        else:
            white = EnginePlayer(evaluator, depth=DEPTH)
            black = EnginePlayer(evaluator, depth=DEPTH)
            result = play_game(black, white)
            # Invert result because perspective is swapped
            if result == "1-0":
                result = "0-1"
            elif result == "0-1":
                result = "1-0"

        score = result_to_score(result)
        scores.append(score)
        results.append(result)

        print(f"Game {i + 1:02d}: {result}")

    result_counts = Counter(results)

    print("\nValidation summary")
    print("------------------")
    print(f"Games played: {N_GAMES}")
    print(f"Results: {dict(result_counts)}")
    print(f"Average score: {np.mean(scores):.3f}")
    print(f"Score std dev: {np.std(scores):.3f}")

    assert abs(np.mean(scores) - 0.5) < 0.1, (
        "Self-play symmetry violated: average score too far from 0.5"
    )


if __name__ == "__main__":
    validate()
