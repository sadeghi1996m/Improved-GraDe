from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from compare_methods_fig4 import RESULT_FILES
from utils.graph import MODELS
from utils.plotting import PLOTS, RESULTS, finish_plot


def plot_graphs(results_dir=RESULTS, plots_dir=PLOTS, show=True):
    SDR_GraDe, SDR_ImGraDe = [], []
    for model in MODELS:
        with np.load(Path(results_dir) / RESULT_FILES[model]) as data:
            SDR_GraDe.append(data["SDR_GraDe"].mean())
            SDR_ImGraDe.append(data["SDR_ImGraDe"].mean())

    positions = np.arange(len(MODELS))
    figure, ax = plt.subplots()
    ax.bar(positions - 0.2, SDR_ImGraDe, width=0.4, label="ImGraDe")
    ax.bar(positions + 0.2, SDR_GraDe, width=0.4, label="GraDe")
    ax.set_xticks(positions, [settings["label"] for settings in MODELS.values()], rotation=20)
    ax.set(ylabel="SDR (dB)", ylim=(0, 30))
    ax.grid(True, axis="y")
    ax.legend()
    finish_plot(figure, "method_comparison_fig4", plots_dir, show)


def plot_nodes(results_dir=RESULTS, plots_dir=PLOTS, show=True):
    with np.load(Path(results_dir) / "G1_SNR_N2.npz") as data:
        SDR_GraDe = data["SDR_GraDe"].mean(axis=(1, 2))
        SDR_ImGraDe = data["SDR_ImGraDe"].mean(axis=(1, 2))
        node_counts = data["node_counts"]

    positions = np.arange(len(node_counts))
    figure, ax = plt.subplots()
    ax.plot(positions, SDR_ImGraDe, "-s", color="black", linewidth=2,
            markerfacecolor="white", label="ImGraDe")
    ax.plot(positions, SDR_GraDe, "--o", color="black", linewidth=2,
            markerfacecolor="white", label="GraDe")
    ax.set_xticks(positions, node_counts)
    ax.set(xlabel="number of nodes (N)", ylabel="SDR (dB)", ylim=(0, 25))
    ax.grid(True)
    ax.legend()
    finish_plot(figure, "methods_vs_node_count_fig5", plots_dir, show)


if __name__ == "__main__":
    plot_graphs()
    plot_nodes()
