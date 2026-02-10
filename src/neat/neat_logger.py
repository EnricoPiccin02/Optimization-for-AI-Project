from src.utils.logging.csv_logger import CSVLogger


class NeatLogger:
    """
    NEAT-specific logger defining the logging schema.
    """

    HEADERS = ["generation", "best_fitness", "mean_fitness"]

    def __init__(self, path: str):
        self.logger = CSVLogger(path, self.HEADERS)

    def log_generation(self, generation, best_fitness, mean_fitness):
        self.logger.log(
            [
                generation,
                best_fitness,
                mean_fitness,
            ]
        )
