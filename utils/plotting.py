from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PLOTS = ROOT / "plots"


def finish_plot(figure, filename, output_dir=PLOTS, show=True):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figure.tight_layout()
    figure.savefig(output_dir / f"{filename}.png", dpi=300)
    figure.savefig(output_dir / f"{filename}.pdf")

    if show:
        plt.show()
    else:
        plt.close(figure)


def plot_moments(values, moments, measure, xlabel):
    figure, ax = plt.subplots()
    ddof = 1 if measure.shape[-1] > 1 else 0
    ax.errorbar(values, measure.mean(axis=-1), measure.std(axis=-1, ddof=ddof),
                fmt="-o", color="black", markersize=4, markerfacecolor="white",
                linewidth=1, label=r"$\mathcal{D}$")

    averaged = moments.mean(axis=-1)
    curves = [np.sqrt(averaged[1, 0]) / averaged[0, 0],
              np.sqrt(averaged[1, 0]) / averaged[1, 1],
              np.sqrt(averaged[3, 1]) / averaged[1, 1],
              np.sqrt(averaged[3, 1]) / averaged[3, 3]]
    labels = [r"$\sigma_{12}/m_{11}$", r"$\sigma_{12}/m_{22}$",
              r"$\sigma_{24}/m_{22}$", r"$\sigma_{24}/m_{44}$"]

    for curve, style, color, label in zip(curves, ["--", ":", "-.", "-"],
                                         ["black", "black", "black", "0.4"], labels):
        ax.plot(values, curve, linestyle=style, color=color, linewidth=1, label=label)

    ax.set(xscale="log", yscale="log", xlabel=xlabel, ylabel="averaged value")
    ax.grid(True)
    ax.legend()
    return figure, ax
