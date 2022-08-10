# -*- coding: utf-8 -*-
"""Algorithms for spectral clustering using fully connected matrices"""

# Author: Gael Varoquaux gael.varoquaux@normalesup.org
#         Brian Cheung
#         Wei LI <kuantkid@gmail.com>
#         Adapted by Lluis Garrido to a fully connected graph, November 2020
# License: BSD 3 clause

import warnings

import numpy as np
import scipy as sp
from sklearn.cluster import k_means


def spectral_clustering_fully_connected(affinity, n_clusters=8, n_init=10):
    """Apply clustering to a projection to the normalized laplacian. This
    code is an adaptation of the original code to a fully connected 
    affinity matrix.

    In practice Spectral Clustering is very useful when the structure of
    the individual clusters is highly non-convex or more generally when
    a measure of the center and spread of the cluster is not a suitable
    description of the complete cluster. For instance when clusters are
    nested circles on the 2D plan.

    Labels are assigned using the k-means algorithm.

    Read more in the :ref:`User Guide <spectral_clustering>`.

    Parameters
    -----------
    affinity : fully connected graph, shape: (n_samples, n_samples)
        The affinity matrix describing the relationship of the samples to
        embed. **Must be symmetric**.

        Possible examples:
          - adjacency matrix of a graph,
          - heat kernel of the pairwise distance matrix of the samples,
          - symmetric k-nearest neighbours connectivity matrix of the samples.

    n_clusters : integer, optional, default: 8
        Number of clusters to extract.

    n_init : int, optional, default: 10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.

    Returns
    -------
    labels : array of integers, shape: n_samples
        The labels of the clusters.

    References
    ----------

    - Normalized cuts and image segmentation, 2000
      Jianbo Shi, Jitendra Malik
      http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.160.2324

    - A Tutorial on Spectral Clustering, 2007
      Ulrike von Luxburg
      http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.165.9323

    - Multiclass spectral clustering, 2003
      Stella X. Yu, Jianbo Shi
      http://www1.icsi.berkeley.edu/~stellayu/publication/doc/2003kwayICCV.pdf

    Notes
    ------
    The graph should contain only one connect component, elsewhere
    the results make little sense.
    """

    if (n_clusters < 1):
        raise ValueError("Number of clusters should be a positive value")

    if (n_init < 1):
        raise ValueError("Number of clusters should be a positive value")

    matrix_check = (affinity.T + affinity)/2 - affinity
    min_value = np.min(matrix_check)
    max_value = np.max(matrix_check)

    if not ((min_value == 0.0) and (max_value == 0.0)):
        warnings.warn("Input affinity matrix seems not to be symmetric,"
                      "spectral_embedding may not work as expected.")

    # Let's compute the Laplacian, L = D - W, where D is the degree matrix.
    D = np.diag(np.sum(affinity, 0))
    L = D - affinity

    # Compute eigenvalues and eigenvectors of the normalized graph laplacian
    eigenvalues, eigenvectors = sp.linalg.eigh(L,D)

    # Compute number of connected components
    connected_components = np.sum(np.abs(eigenvalues) < 1e-5)

    if (connected_components > 1):
        warnings.warn("Affinity matrix seems to have more than one connected component."
                      "The spectral clustering may not work as expected")

    # These are the vector with which the clustering is going to be performed
    maps = eigenvectors[:,0:n_clusters]


#_, labels, _ = k_means(maps, n_clusters, random_state=None, n_init=n_init)
    _, labels, _ = k_means(maps, n_clusters, random_state=0, n_init=n_init)
    return labels
