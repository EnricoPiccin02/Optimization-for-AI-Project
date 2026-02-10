from src.utils.logging.csv_logger import CSVLogger


class CMALogger:
    """
    CMA-ES specific logger defining the logging schema.
    """

    HEADERS = ["generation", "best_fitness", "mean_fitness", "sigma"]

    def __init__(self, path: str):
        self.logger = CSVLogger(path, self.HEADERS)

    def log_generation(
        self,
        generation: int,
        best_fitness: float,
        mean_fitness: float,
        sigma: float,
    ):
        self.logger.log(
            [
                generation,
                best_fitness,
                mean_fitness,
                sigma,
            ]
        )
