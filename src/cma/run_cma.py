import numpy as np
import cma

from src.cma.fitness import fitness
from src.engine.baseline import make_baseline_opponent
from src.cma.cma_logger import CMALogger
from src.utils.experiment.metadata import ExperimentMetadata
from src.cma.serialization import save_champion
from src.utils.paths import (
    CMA_LOG_PATH,
    CMA_METADATA_PATH,
)

from src.cma.config import (
    MAX_GENERATIONS,
    NUM_GAMES,
    SIGMA,
    SEED,
    REEVAL_INTERVAL,
    REEVAL_GAMES,
)


def run_cma(mean, std):
    dim = len(mean)

    opponent = make_baseline_opponent(mean, std)

    init = np.zeros(dim)
    popsize = 4 + int(3 * np.log(dim))

    # Save experiment metadata once
    metadata = ExperimentMetadata(
        algorithm="CMA-ES",
        parameters={
            "sigma": SIGMA,
            "popsize": popsize,
            "max_generations": MAX_GENERATIONS,
            "seed": SEED,
            "num_games_per_fitness": NUM_GAMES,
            "reeval_interval": REEVAL_INTERVAL,
            "reeval_games": REEVAL_GAMES,
        },
    )

    metadata.save(CMA_METADATA_PATH)

    es = cma.CMAEvolutionStrategy(
        init,
        SIGMA,
        {
            "popsize": popsize,
            "seed": SEED,
            "verb_log": 0,
            "verb_disp": 1,
        },
    )

    logger = CMALogger(CMA_LOG_PATH)

    for generation in range(MAX_GENERATIONS):
        solutions = es.ask()
        values = []

        for w in solutions:
            score = fitness(w, mean, std, opponent, num_games=NUM_GAMES)
            values.append(-score)

        es.tell(solutions, values)

        best_fitness = -min(values)
        mean_fitness = -np.mean(values)

        logger.log_generation(generation, best_fitness, mean_fitness, es.sigma)

        if generation > 0 and generation % REEVAL_INTERVAL == 0:
            champ = es.result.xbest
            champ_score = fitness(champ, mean, std, opponent, num_games=REEVAL_GAMES)
            print(
                f"[Re-eval] Generation {generation}: champion score = {champ_score:.3f}"
            )

    save_champion(es.result.xbest)
    return es.result.xbest
