from itertools import permutations

import numpy as np
# from fast_bss_eval import bss_eval_sources


def graph_jd(signals, adjacency, order=1, center=False):
    """Apply the whitening and joint rotations used by Graph_JD and Graph_JD_0."""
    p, n = signals.shape
    if center:
        signals = signals - signals.mean(axis=1, keepdims=True)

    u, d, vh = np.linalg.svd(signals @ signals.T, full_matrices=False)
    tolerance = p * np.finfo(float).eps
    whitening = u @ np.sqrt(np.linalg.pinv(np.diag(d), rcond=tolerance)) @ vh
    signals = whitening @ signals

    matrices = []
    shifted = signals
    for _ in range(order):
        shifted = shifted @ adjacency
        covariance = shifted @ signals.T / n
        matrices.append(covariance / np.linalg.norm(covariance, "fro"))

    matrices = np.hstack(matrices).astype(complex)
    rotation = np.eye(p, dtype=complex)
    b = np.array([[1, 0, 0], [0, 1, 1], [0, -1j, 1j]])
    threshold = 1 / (np.sqrt(n) * 100)
    changed = True

    while changed:
        changed = False
        for row in range(p - 1):
            for column in range(row + 1, p):
                ip = np.arange(row, p * order, p)
                iq = np.arange(column, p * order, p)
                g = np.vstack((matrices[row, ip] - matrices[column, iq],
                               matrices[row, iq], matrices[column, ip]))

                _, vectors = np.linalg.eigh(np.real(b @ g @ g.conj().T @ b.conj().T))
                angles = vectors[:, -1]
                if angles[0] < 0:
                    angles = -angles
                c = np.sqrt(0.5 + angles[0] / 2)
                s = 0.5 * (angles[1] - 1j * angles[2]) / c

                if abs(s) > threshold:
                    changed = True
                    pair = [row, column]
                    givens = np.array([[c, -s.conjugate()], [s, c]])
                    rotation[:, pair] = rotation[:, pair] @ givens
                    matrices[pair, :] = givens.conj().T @ matrices[pair, :]
                    left, right = matrices[:, ip].copy(), matrices[:, iq].copy()
                    matrices[:, ip] = c * left + s * right
                    matrices[:, iq] = -s.conjugate() * left + c * right

    mixing = np.linalg.pinv(whitening, rcond=tolerance) @ rotation
    return np.real_if_close(mixing)


def calculate_sdr(estimated, sources):
    """Calculate SDR."""

    # If you use bss_eval,
    # return bss_eval_sources(sources, estimated)[0]

    p = len(sources)
    energy = np.sum(sources**2, axis=1)

    # Correlation ordering: uncomment this block and comment the SDR ordering block.
    # estimated_energy = np.sum(estimated**2, axis=1)
    # with np.errstate(divide="ignore", invalid="ignore"):
    #     similarity = np.abs(sources @ estimated.T) / np.sqrt(energy[:, None] * estimated_energy)
    # similarity[np.isnan(similarity)] = 0
    # assignment = max(permutations(range(p - 1, -1, -1)),
    #                  key=lambda perm: similarity[np.arange(p), perm].sum())

    # SDR ordering: select the permutation with the highest mean SDR.
    with np.errstate(divide="ignore", invalid="ignore"):
        pair_products = sources @ estimated.T
        pair_alpha = energy[:, None] / pair_products
        pair_residual = sources[:, None, :] - pair_alpha[:, :, None] * estimated[None, :, :]
        pair_sdr = 10 * np.log10(energy[:, None] / np.sum(pair_residual**2, axis=2))
    pair_sdr[pair_products == 0] = -np.inf

    assignments = list(permutations(range(p - 1, -1, -1)))
    with np.errstate(invalid="ignore"):
        scores = np.array([pair_sdr[np.arange(p), perm].mean() for perm in assignments])
    scores[np.isnan(scores)] = -np.inf
    assignment = assignments[np.argmax(scores)]

    ordered = estimated[list(assignment)]

    products = np.sum(sources * ordered, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = energy / products
        residual = sources - alpha[:, None] * ordered
        sdr = 10 * np.log10(energy / np.sum(residual**2, axis=1))

    sdr[products == 0] = -np.inf
    return sdr
