import numpy as np


def gma_sources(adjacency, theta):
    """Generate independent Gaussian sources with the specified GMA coefficients."""
    rng = np.random.default_rng()
    noise = rng.standard_normal((len(theta), len(adjacency)))
    sources = noise.copy()
    shifted = noise

    for k in range(theta.shape[1]):
        shifted = shifted @ adjacency
        sources += theta[:, k, None] * shifted

    return sources


def normalized_theta(singular_values, n_sources, random_second=True):
    rng = np.random.default_rng()
    n = len(singular_values)
    l1 = np.sqrt(np.sum(singular_values**2) / n)
    l2 = np.sqrt(np.sum(singular_values**4) / n)

    theta = np.empty((n_sources, 2))
    theta[:, 0] = (0.2 + np.arange(n_sources) / n_sources * 0.8) / l1
    theta[:, 1] = 0.5 * (rng.random(n_sources) if random_second else 1) / l2
    return theta


def diagonality(matrix):
    scale = 1 / np.sqrt(np.diag(matrix))
    normalized = scale[:, None] * matrix * scale[None, :]
    return np.linalg.norm(normalized - np.eye(len(matrix)), "fro") ** 2


def source_moments(sources, adjacency):
    covariance = sources @ adjacency @ sources.T
    diagonal = np.diag(covariance)
    moments = np.diag(diagonal).copy()

    lower = np.tril_indices(len(sources), -1)
    moments[lower] = covariance[lower] ** 2

    ratios = np.zeros_like(moments)
    ratios[lower] = moments[lower] / np.outer(diagonal, diagonal)[lower]
    return moments, ratios, ratios.sum()
