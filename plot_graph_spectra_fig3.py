import matplotlib.pyplot as plt
import numpy as np

from utils.graph import Graph, MODELS
from utils.plotting import PLOTS, finish_plot


def run(n_nodes=5000, plots_dir=PLOTS, show=True):
    limits = {"barabasi": 200, "erdos": 60, "sensor": 6, "geometric": 170, "stochastic_block": 110}

    for model, settings in MODELS.items():
        graph = Graph(model, n_nodes)
        _, values, _ = graph.svd()

        figure, ax = plt.subplots()
        ax.plot(np.arange(1, n_nodes + 1), values, "-o", color="black",
                markersize=5, linewidth=0.5)
        ax.set(xlabel="number", ylabel="magnitude of eigenvalues", title=settings["label"],
               xlim=(-100, n_nodes), ylim=(0, limits[model]))
        ax.grid(True)
        finish_plot(figure, f"spectrum_{model}_fig3", plots_dir, show)


if __name__ == "__main__":
    run()
