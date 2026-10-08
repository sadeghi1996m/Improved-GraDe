from pathlib import Path

import numpy as np

from utils.experiments import moment_trials
from utils.graph import Graph
from utils.plotting import PLOTS, RESULTS, finish_plot, plot_moments


def run(n_nodes=5000, decays=(1, 3.3, 10, 33, 100, 333, 1000, 3333, 10000),
        repetitions=1000, results_dir=RESULTS, plots_dir=PLOTS, show=True):
    graph = Graph("geometric", n_nodes, radius=0.1)
    u, _, _ = graph.svd()

    moments = np.zeros((4, 4, len(decays), repetitions))
    ratios = np.zeros_like(moments)
    measure = np.zeros((len(decays), repetitions))

    for index, decay in enumerate(decays):
        values = np.exp(-np.arange(n_nodes) / decay)
        values = values / values.sum() * n_nodes
        adjacency = (u * values) @ u.T

        moments[:, :, index], ratios[:, :, index], measure[index] = moment_trials(
            adjacency, values, repetitions
        )
        print(f"Decay={decay}")

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    output = results_dir / "decay_fig7.npz"
    np.savez(output, decays=decays, moments=moments, ratio=ratios, D_measure=measure)

    figure, ax = plot_moments(decays, moments, measure, r"$\rho$ (decay parameter)")
    ax.set(xlim=(1, 1e4), ylim=(1e-4, 1e2))
    ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100])
    finish_plot(figure, "decay_diagonality_fig7", plots_dir, show)
    return output


if __name__ == "__main__":
    run()
