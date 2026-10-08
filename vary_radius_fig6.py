from pathlib import Path

import numpy as np

from utils.experiments import moment_trials
from utils.graph import Graph
from utils.plotting import PLOTS, RESULTS, finish_plot, plot_moments


def run(n_nodes=5000, radii=(0.01, 0.035, 0.07, 0.1, 0.3, 0.5, 0.7),
        repetitions=1000, results_dir=RESULTS, plots_dir=PLOTS, show=True):
    moments = np.zeros((4, 4, len(radii), repetitions))
    ratios = np.zeros_like(moments)
    measure = np.zeros((len(radii), repetitions))

    for index, radius in enumerate(radii):
        graph = Graph("geometric", n_nodes, radius=radius)
        _, values, _ = graph.svd()
        moments[:, :, index], ratios[:, :, index], measure[index] = moment_trials(
            graph.adjacency, values, repetitions
        )
        print(f"Radius={radius}")

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    output = results_dir / "radius_fig6.npz"
    np.savez(output, radii=radii, moments=moments, ratio=ratios, D_measure=measure)

    figure, ax = plot_moments(radii, moments, measure, "neighborhood radius")
    ax.set_xlim(0.01, 0.71)
    ax.set_ylim(top=6)
    finish_plot(figure, "radius_diagonality_fig6", plots_dir, show)
    return output


if __name__ == "__main__":
    run()
