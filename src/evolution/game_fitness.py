import numpy as np
from src.utils.game_loop import play_game
from src.utils.results import result_to_score


def evaluate_player(player, opponent, num_games):
    """
    Monte Carlo estimate of expected score of `player`
    against a fixed `opponent`.

    Notes
    -----
    - This objective is stochastic.
    - Variance decreases as num_games increases.
    - Optimisation is therefore noisy.
    """
    scores = []

    for i in range(num_games):
        if i % 2 == 0:
            result = play_game(player, opponent)
            scores.append(result_to_score(result))
        else:
            result = play_game(opponent, player)
            scores.append(1.0 - result_to_score(result))

    return float(np.mean(scores))
