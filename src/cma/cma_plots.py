from src.utils.plots.convergence import plot_convergence


def plot_cma_convergence(log_path: str, output_path: str | None = None):
    plot_convergence(
        log_path=log_path,
        x="generation",
        ys=["best_fitness", "mean_fitness"],
        title="CMA-ES Convergence",
        ylabel="Fitness",
        output_path=output_path,
    )
