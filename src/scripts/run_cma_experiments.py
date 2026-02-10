import numpy as np

from src.cma.run_cma import run_cma
from src.scripts.evaluate_cma_champion import evaluate_cma_champion
from src.cma.cma_plots import plot_cma_convergence
from src.utils.feature_stats import load_or_compute_feature_stats
from src.utils.paths import CMA_LOG_PATH, CMA_PLOT_PATH
from src.cma.config import (
    MAX_GENERATIONS,
    NUM_GAMES,
    REEVAL_INTERVAL,
    REEVAL_GAMES,
)


def run_cma_experiments():
    mean, std = load_or_compute_feature_stats()

    dim = len(mean)
    popsize = 4 + int(3 * np.log(dim))

    # Compute CMA-ES budget
    base_games = popsize * MAX_GENERATIONS * NUM_GAMES

    num_reevals = MAX_GENERATIONS // REEVAL_INTERVAL
    reeval_games = num_reevals * REEVAL_GAMES

    cma_games = base_games + reeval_games

    print(f"[CMA-ES] Total evaluation games: {cma_games}")

    print("\nRunning CMA-ES optimization...")
    run_cma(mean, std)

    print("\nEvaluating champion...")
    evaluate_cma_champion(mean, std, num_games=200)

    print("\nPlotting convergence...")
    plot_cma_convergence(CMA_LOG_PATH, CMA_PLOT_PATH)


if __name__ == "__main__":
    run_cma_experiments()
