import numpy as np

from .separation import calculate_sdr, graph_jd
from .signals import gma_sources, normalized_theta, source_moments


def compare_once(graph, removed):
    if not 0 <= removed < graph.n_nodes:
        raise ValueError("Removed components must be fewer than the graph nodes.")

    rng = np.random.default_rng()
    u, d, _ = graph.svd()
    sources = gma_sources(graph.adjacency, normalized_theta(d, 4))
    observations = rng.standard_normal((4, 4)) @ sources

    baseline = graph_jd(observations, graph.adjacency, center=True)

    spectral = u.T @ observations.T
    spectral[:removed] = 0
    filtered = (u @ spectral).T
    proposed = graph_jd(filtered, graph.adjacency)

    SDR_GraDe = calculate_sdr(np.linalg.solve(baseline, observations), sources)
    SDR_ImGraDe = calculate_sdr(np.linalg.solve(proposed, observations), sources)
    return SDR_GraDe, SDR_ImGraDe


def moment_trials(adjacency, singular_values, repetitions):
    moments = np.zeros((4, 4, repetitions))
    ratios = np.zeros_like(moments)
    measure = np.zeros(repetitions)
    theta = normalized_theta(singular_values, 4, random_second=False)

    for repetition in range(repetitions):
        sources = gma_sources(adjacency, theta)
        moments[:, :, repetition], ratios[:, :, repetition], measure[repetition] = source_moments(
            sources, adjacency
        )

    return moments, ratios, measure
