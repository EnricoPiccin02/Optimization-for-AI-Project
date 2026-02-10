import pandas as pd
import matplotlib.pyplot as plt
from typing import List


def plot_convergence(
    log_path: str,
    x: str,
    ys: List[str],
    title: str,
    ylabel: str,
    output_path: str | None = None,
):
    df = pd.read_csv(log_path)

    plt.figure()
    for y in ys:
        plt.plot(df[x], df[y], label=y)

    plt.xlabel(x.capitalize())
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()

    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
