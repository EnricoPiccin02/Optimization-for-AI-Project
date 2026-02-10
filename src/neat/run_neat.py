import random
import numpy as np

from src.neat.population import Population
from src.neat.neat_logger import NeatLogger
from src.utils.experiment.metadata import ExperimentMetadata
from src.neat.serialization import save_champion
from src.utils.paths import (
    NEAT_LOG_PATH,
    NEAT_METADATA_PATH,
)
from src.neat.config import NEAT_CONFIG


def run_neat(mean, std, baseline):
    num_inputs = len(mean)

    config = dict(NEAT_CONFIG)
    config["num_inputs"] = num_inputs

    random.seed(config["seed"])
    np.random.seed(config["seed"])

    metadata = ExperimentMetadata(
        algorithm="NEAT",
        parameters=config,
    )
    metadata.save(NEAT_METADATA_PATH)

    population = Population(
        config=config,
        mean=mean,
        std=std,
        baseline=baseline,
    )

    logger = NeatLogger(NEAT_LOG_PATH)

    for generation in range(config["max_generations"]):
        population.evaluate()

        best = population.best_genome.fitness
        mean_fit = sum(g.fitness for g in population.genomes) / len(population.genomes)

        logger.log_generation(generation, best, mean_fit)

        print(f"[NEAT] Gen {generation:03d} | Best: {best:.3f} | Mean: {mean_fit:.3f}")

        population.reproduce()

    save_champion(population.best_genome)
    return population.best_genome