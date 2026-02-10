RESULTS_DIR = "results"

# Paths for feature normalization statistics
FEATURE_MEAN_PATH = f"{RESULTS_DIR}/feature_mean.npy"
FEATURE_STD_PATH = f"{RESULTS_DIR}/feature_std.npy"

# Paths for CMA-ES results
CMA_DIR = f"{RESULTS_DIR}/cma"
CMA_LOG_PATH = f"{CMA_DIR}/cma_log.csv"
CMA_PLOT_PATH = f"{CMA_DIR}/convergence.png"
CMA_CHAMPION_PATH = f"{CMA_DIR}/champion_weights.npy"
CMA_METADATA_PATH = f"{CMA_DIR}/metadata.json"

# Paths for NEAT results
NEAT_DIR = f"{RESULTS_DIR}/neat"
NEAT_LOG_PATH = f"{NEAT_DIR}/neat_log.csv"
NEAT_PLOT_PATH = f"{NEAT_DIR}/convergence.png"
NEAT_CHAMPION_PATH = f"{NEAT_DIR}/champion_genome.pkl"
NEAT_METADATA_PATH = f"{NEAT_DIR}/metadata.json"
