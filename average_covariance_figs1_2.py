from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.graph import Graph
from utils.plotting import PLOTS, RESULTS, finish_plot
from utils.signals import diagonality, gma_sources


def run(n_nodes=5000, counts=(1, 10, 100, 1000, 10000),
        results_dir=RESULTS, plots_dir=PLOTS, show=True):
    graph = Graph("barabasi", n_nodes)
    w = graph.adjacency

    theta = np.tile([0.05, 0.0015, 1e-5], (4, 1))
    sources = gma_sources(w, theta)
    c0 = sources @ sources.T / 1e4
    c1 = sources @ w @ sources.T / 1e6

    scores0, scores1 = np.zeros(len(counts)), np.zeros(len(counts))
    for index, repetitions in enumerate(counts):
        covariance0, covariance1 = np.zeros((4, 4)), np.zeros((4, 4))
        for _ in range(repetitions):
            sample = gma_sources(w, theta)
            covariance0 += sample @ sample.T
            covariance1 += sample @ w @ sample.T

        scores0[index] = diagonality(covariance0 / repetitions)
        scores1[index] = diagonality(covariance1 / repetitions)
        print(f"Averaged {repetitions} realizations")

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    output = results_dir / "covariance_figs1_2.npz"
    np.savez(output, sources=sources, C0=c0, C1=c1, counts=counts,
             d_score0=scores0, d_score1=scores1)
    print("Figure 1 sample matrices:\n", c0, "\n", c1)

    figure, ax = plt.subplots()
    ax.loglog(counts, scores0, "-s", color="black", linewidth=2,
              markerfacecolor="white", label=r"$C_0^M$")
    ax.loglog(counts, scores1, "--o", color="black", linewidth=2,
              markerfacecolor="white", label=r"$C_1^M$")
    ax.set(xlabel="M", ylabel=r"diagonality ($\mathcal{D}$)")
    ax.legend()
    ax.grid(True)
    finish_plot(figure, "averaged_covariance_fig2", plots_dir, show)
    return output


if __name__ == "__main__":
    run()
