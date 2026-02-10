from src.neat.run_neat import run_neat
from src.scripts.evaluate_neat_champion import evaluate_neat_champion
from src.utils.feature_stats import load_or_compute_feature_stats
from src.engine.baseline import make_baseline_opponent
from src.neat.neat_plots import plot_neat_convergence
from src.neat.config import NEAT_CONFIG
from src.utils.paths import NEAT_LOG_PATH, NEAT_PLOT_PATH


def run_neat_experiments():
    mean, std = load_or_compute_feature_stats()
    baseline = make_baseline_opponent(mean, std)

    # Compute NEAT budget
    neat_games = (
        NEAT_CONFIG["population_size"]
        * NEAT_CONFIG["max_generations"]
        * NEAT_CONFIG["games_per_eval"]
    )

    print(f"[NEAT] Total evaluation games: {neat_games}")

    print("\nRunning NEAT optimization...")
    run_neat(mean, std, baseline)

    print("\nEvaluating champion...")
    evaluate_neat_champion(num_games=200)

    print("\nPlotting convergence...")
    plot_neat_convergence(NEAT_LOG_PATH, NEAT_PLOT_PATH)


if __name__ == "__main__":
    run_neat_experiments()
