from pathlib import Path

import numpy as np

from utils.experiments import compare_once
from utils.graph import Graph, MODELS
from utils.plotting import RESULTS


RESULT_FILES = {
    "barabasi": "G1_SDR.npz",
    "erdos": "G2_SDR.npz",
    "sensor": "G3_SDR.npz",
    "geometric": "G4_SDR.npz",
    "stochastic_block": "G5_SDR.npz",
}


def run(graph_type="barabasi", n_nodes=5000, repetitions=100, removed=None,
        results_dir=RESULTS):
    if graph_type == "all":
        return [run(model, n_nodes, repetitions, removed, results_dir) for model in MODELS]

    removed = MODELS[graph_type]["remove"] if removed is None else removed
    SDR_GraDe = np.zeros((4, repetitions))
    SDR_ImGraDe = np.zeros_like(SDR_GraDe)

    for repetition in range(repetitions):
        graph = Graph(graph_type, n_nodes)
        SDR_GraDe[:, repetition], SDR_ImGraDe[:, repetition] = compare_once(graph, removed)
        print(f"{graph_type}: {repetition + 1}/{repetitions}")

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    output = results_dir / RESULT_FILES[graph_type]
    np.savez(output, SDR_GraDe=SDR_GraDe, SDR_ImGraDe=SDR_ImGraDe,
             n_nodes=n_nodes, removed=removed)
    return output


if __name__ == "__main__":
    # run(graph_type="barabasi")
    # run(graph_type="erdos")
    # run(graph_type="sensor")
    # run(graph_type="geometric")
     run(graph_type="stochastic_block")
