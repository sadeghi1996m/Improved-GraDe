import numpy as np
from pygsp import graphs
from scipy import sparse
from scipy.spatial import cKDTree


MODELS = {
    "barabasi": {"label": "Barabasi-Albert", "remove": 1},
    "erdos": {"label": "Erdos-Renyi", "remove": 1},
    "sensor": {"label": "Sensor network", "remove": 400},
    "geometric": {"label": "Geometric", "remove": 250},
    "stochastic_block": {"label": "Stochastic block", "remove": 3},
}


def scalefree(n, m0, m):
    """Create a Barabasi-Albert graph.
    Python adaptation of the original scalefree.m implementation.
    Source: ETH Zurich course material, "Modeling and Simulating Social Systems with MATLAB".
    Original authors: Stefan Brugger and Cristoph Schwirzer (2011)."""

    rng = np.random.default_rng()
    if not 1 <= m < m0 <= n:
        raise ValueError("Barabasi parameters must satisfy 1 <= m < m0 <= n.")

    initial = np.zeros((m0, m0))
    for node in range(m0):
        neighbors = rng.choice(m0 - 1, m, replace=False)
        neighbors += neighbors >= node
        initial[node, neighbors] = 1
        initial[neighbors, node] = 1

    rows, columns = np.triu(initial, 1).nonzero()
    edges = np.column_stack((rows, columns)).ravel().tolist()
    for node in range(m0, n):
        neighbors = set()
        while len(neighbors) < m:
            neighbors.add(edges[rng.integers(len(edges))])
        for neighbor in neighbors:
            edges.extend((node, neighbor))

    edges = np.asarray(edges).reshape(-1, 2)
    rows = np.concatenate((edges[:, 0], edges[:, 1]))
    columns = np.concatenate((edges[:, 1], edges[:, 0]))
    adjacency = sparse.csr_matrix((np.ones(len(rows)), (rows, columns)), shape=(n, n))
    return graphs.Graph(adjacency)


def geometric_graph(n, radius):
    """Connect uniform unit-square points whose distance is below the radius."""
    rng = np.random.default_rng()
    coords = rng.random((n, 2))

    pairs = cKDTree(coords).query_pairs(radius, output_type="ndarray")
    distances_squared = np.sum((coords[pairs[:, 0]] - coords[pairs[:, 1]]) ** 2, axis=1)
    pairs = pairs[distances_squared < radius**2]

    rows = np.concatenate((pairs[:, 0], pairs[:, 1]))
    columns = np.concatenate((pairs[:, 1], pairs[:, 0]))
    adjacency = sparse.csr_matrix((np.ones(len(rows)), (rows, columns)), shape=(n, n))
    return graphs.Graph(adjacency, coords=coords)


class Graph:
    """Create a graph by type and cache its adjacency SVD on demand."""

    def __init__(self, graph_type, n_nodes=5000, **parameters):
        self.graph_type = graph_type

        if graph_type == "barabasi":
            self.gsp_graph = scalefree(n_nodes, parameters.get("m0", 100), parameters.get("m", 50))

        elif graph_type == "erdos":
            self.gsp_graph = graphs.ErdosRenyi(n_nodes, p=parameters.get("p", 0.01))

        elif graph_type == "sensor":
            k = parameters.get("k", 6)
            for _ in range(10):
                sensor = graphs.Sensor(n_nodes, k=k)
                tree = cKDTree(sensor.coords)
                distances, indices = tree.query(
                    sensor.coords,
                    k=k + 1,
                )

                average_distance = distances.mean()
                sigma = average_distance ** 2

                self.gsp_graph = graphs.NNGraph(
                    sensor.coords, k=k, sigma=sigma, center=False, rescale=False,
                    plotting=sensor.plotting,
                )

                if self.gsp_graph.is_connected():
                    break

        elif graph_type == "geometric":
            self.gsp_graph = geometric_graph(n_nodes, parameters.get("radius", 0.1))

        elif graph_type == "stochastic_block":
            self.gsp_graph = graphs.StochasticBlockModel(
                n_nodes, k=parameters.get("blocks", 3), p=parameters.get("p", 0.05),
                q=parameters.get("q", 0.005),
            )

        elif graph_type == "custom":
            self.gsp_graph = graphs.Graph(parameters["adjacency"])
        else:
            raise ValueError(f"Unknown graph type: {graph_type}")
        
        self.adjacency = self.gsp_graph.W.toarray().astype(float)
        self.n_nodes = self.gsp_graph.N
        self._svd = None

    def svd(self, recompute=False):
        """Compute and cache the adjacency SVD, returning U, singular values, and Vh."""
        if self._svd is None or recompute:
            self._svd = np.linalg.svd(self.adjacency, full_matrices=False)

        return self._svd
