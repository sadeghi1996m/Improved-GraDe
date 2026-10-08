from pathlib import Path

import numpy as np

from utils.experiments import compare_once
from utils.graph import Graph
from utils.plotting import RESULTS


def run(node_counts=(500, 1000, 2500, 5000), repetitions=40,
        results_dir=RESULTS):
    SDR_GraDe = np.zeros((len(node_counts), 4, repetitions))
    SDR_ImGraDe = np.zeros_like(SDR_GraDe)

    for index, n_nodes in enumerate(node_counts):
        for repetition in range(repetitions):
            graph = Graph("barabasi", n_nodes)
            SDR_GraDe[index, :, repetition], SDR_ImGraDe[index, :, repetition] = compare_once(graph, 1)
            print(f"N={n_nodes}: {repetition + 1}/{repetitions}")

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    output = results_dir / "G1_SNR_N2.npz"
    np.savez(output, SDR_GraDe=SDR_GraDe, SDR_ImGraDe=SDR_ImGraDe,
             node_counts=node_counts)
    return output


if __name__ == "__main__":
    run()
