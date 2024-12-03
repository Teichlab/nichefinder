from typing import Union, Dict, Any, Tuple, List

import numpy as np
import scanpy as sc
import scipy as sp

from sklearn.decomposition import NMF
import matplotlib.pyplot as plt


def aggregate_neighbors(
    spatial: sc.AnnData, labels: Union[str, np.matrix, sp.sparse.csr_matrix]
) -> sp.sparse.csr_matrix:
    """
    Aggregate spatial neighbors based on cell type labels.
    The labels are provided as a matrix of cell type probabilities.

    Args:
        spatial (sc.AnnData): Spatial dataset containing spatial neighbors.
        labels (Union[str,np.matrix,sp.sparse.csr_matrix]): Cell type labels.

    Returns:
        sp.sparse.csr_matrix: Aggregated spatial neighbors.

    Raises:
        ValueError: If the cell type labels are not provided in the correct format.
    """
    if isinstance(labels, str):
        labels = spatial.obsm[labels]
    elif isinstance(labels, np.matrix):
        labels = sp.sparse.csr_matrix(labels)
    elif not isinstance(labels, sp.sparse.csr_matrix):
        raise ValueError(
            "labels must be provided as a string, a matrix or a sparse matrix"
        )

    aggregated = labels.T @ spatial.obsp["connectivities"] @ labels

    return aggregated


def find_niches(
    neighbors: sp.sparse.csr_matrix, max_clusters: int = 10, plot: bool = False
) -> Dict[str, Any]:
    """
    Find niches in spatial neighbors using non-negative matrix factorization (NMF).
    The optimal number of clusters (up to `max_clusters`) is automatically determined using the elbow point method.

    Args:
        neighbors (sp.sparse.csr_matrix): Aggregated spatial neighbors.
        max_clusters (int): Maximum number of clusters for NMF.
        plot (bool): Plot the reconstruction error curve.

    Returns:
        Dict[str, Any]: Dictionary containing the optimal number of clusters, membership matrix and features matrix.
    """
    # Calculate reconstruction errors for different cluster numbers
    errors, models = _calculate_reconstruction_errors(neighbors, max_clusters)

    # Find the elbow point (optimal number of clusters)
    optimal_clusters = _find_elbow_point(errors)

    # Plot the reconstruction error curve if required
    if plot:
        plt.plot(
            range(1, max_clusters + 1),
            errors,
            marker="o",
            label="Reconstruction Error",
        )
        plt.axvline(
            optimal_clusters,
            color="r",
            linestyle="--",
            label=f"Optimal Clusters: {optimal_clusters}",
        )
        plt.xlabel("Number of Clusters")
        plt.ylabel("Reconstruction Error")
        plt.title("NMF Reconstruction Error vs. Number of Clusters")
        plt.legend()
        plt.show()

    # Return the optimal number of clusters and the corresponding model
    W, H = models[optimal_clusters - 1]

    return {"n": optimal_clusters, "membership": W, "features": H}


def plot_niches(
    membership: np.matrix,
    labels: List[str],
    threshold: float = 0.5,
    only_shared: bool = False,
) -> None:
    """
    Plot the niches as a bipartite graph. One type of node represents the niches/clusters,
    while the other type represents the cell types that are shared between multiple niches
    after applying a threshold to the soft-memberships (thresholding is applied after scaling
    the soft-memberships to [0, 1]).

    Args:
        membership (np.matrix): Membership matrix.
        labels (List[str]): Cell type labels.
        threshold (float): Threshold for soft-memberships.
        only_shared (bool): Plot only shared cell types between multiple niches.
    """
    assert len(labels) == membership.shape[0]

    try:
        import networkx as nx
    except ImportError:
        raise ImportError(
            "networkx is required for this function, "
            "install it via `pip install networkx`"
        )

    # Scale and apply threshold to soft-memberships
    membership = (membership - membership.min(axis=0)) / (
        membership.max(axis=0) - membership.min(axis=0)
    )
    membership = (membership > threshold).astype(int)

    # Create a bipartite graph
    G = nx.Graph()
    G.add_nodes_from(labels, bipartite=0)
    G.add_nodes_from(range(membership.shape[1]), bipartite=1)
    for i, ct in enumerate(labels):
        for j in range(membership.shape[1]):
            if membership[i, j] == 1:
                G.add_edge(ct, j)

    # remove cell types that are not shared between multiple niches:
    degree_thr = 2 if only_shared else 1
    degrees = G.degree()
    nodes_to_remove = [
        node for node, degree in degrees if node in labels and degree < degree_thr
    ]
    G.remove_nodes_from(nodes_to_remove)

    # Plot the bipartite graph using a kamada-kawai layout
    pos = nx.kamada_kawai_layout(G)
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=300,
        node_color="skyblue",
        edge_color="gray",
    )
    # nx.draw_networkx_labels(
    #     G, pos, labels={i: labels[i] for i in range(len(labels))}, font_size=10
    # )
    plt.title("Niche Analysis")
    plt.show()


def _calculate_reconstruction_errors(
    adjacency_matrix: sp.sparse.csr_matrix, max_components: int
) -> Tuple[List[float], List[Tuple[np.matrix, np.matrix]]]:
    """
    Calculate the reconstruction errors for different numbers of components in NMF.

    Args:
        adjacency_matrix (sp.sparse.csr_matrix): Aggregated spatial neighbors.
        max_components (int): Maximum number of components for NMF.

    Returns:
        Tuple[List[float], List[Tuple[np.matrix, np.matrix]]]: List of reconstruction errors and models.
    """
    errors = []
    models = []
    for k in range(1, max_components + 1):
        model = NMF(n_components=k, init="random", random_state=0)
        W = model.fit_transform(adjacency_matrix)
        H = model.components_

        # Calculate the Frobenius norm of the difference (reconstruction error)
        reconstruction_error = np.linalg.norm(adjacency_matrix - W @ H, ord="fro")
        errors.append(reconstruction_error)
        models.append((W, H))  # Store models for later use
    return errors, models


def _find_elbow_point(errors: List[float]) -> int:
    """
    Find the elbow point in the reconstruction error curve using the second derivative method.

    Args:
        errors (List[float]): List of reconstruction errors for different cluster numbers.

    Returns:
        int: Elbow point.
    """
    # Calculate first and second differences
    first_differences = np.diff(errors)
    second_differences = np.diff(first_differences)

    # Find the elbow point
    elbow_index = (
        np.argmin(second_differences) + 1
    )  # +1 to adjust for second derivative index shift
    return elbow_index + 1  # +1 to adjust for 1-based cluster count
