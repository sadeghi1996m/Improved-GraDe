# Improved GraDe for Blind Separation of Graph Signals

This repository provides a Python implementation of **improved GraDe
(ImGraDe)**, a method for blind source separation of Gaussian graph signals
introduced in [*An Improved GraDe Method for Blind Separation of Graph
Signals*](https://doi.org/10.1109/TSP.2023.3331264) by Mohammad Sadeghi and
Massoud Babaie-Zadeh.

The method uses the spectral representation of graph signals to reduce the
influence of outlier eigenvalues of the graph adjacency matrix, which
increases the diagonality of graph autocovariance matrices and improves
separation quality. The original **GraDe** algorithm is also implemented.
The experiments use Gaussian graph moving-average signals to compare the
two methods on Barabási–Albert, Erdős–Rényi, sensor network, geometric,
and stochastic block graphs.

## Citation

Reference publication:

Mohammad Sadeghi and Massoud Babaie-Zadeh, “An Improved GraDe Method for Blind
Separation of Graph Signals,” *IEEE Transactions on Signal Processing*,
vol. 71, pp. 4382–4391, 2023.
[DOI: 10.1109/TSP.2023.3331264](https://doi.org/10.1109/TSP.2023.3331264).


The original implementation used to generate the results reported in the
paper was written in MATLAB. This repository provides a Python version of
that implementation.
