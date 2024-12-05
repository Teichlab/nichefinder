from typing import Union, Dict, Any, Tuple, List, Optional
from collections import namedtuple
import logging

import numpy as np
import scanpy as sc
import scipy as sp
from scipy.cluster.hierarchy import linkage, leaves_list, optimal_leaf_ordering
import pandas as pd

from sklearn.decomposition import NMF
import matplotlib.pyplot as plt


NicheResult = namedtuple("NicheResult", ["n", "membership", "features"])


def aggregate_neighbors(
    spatial: sc.AnnData, label: str, scale: bool = True
) -> pd.DataFrame:
    """
    Aggregate spatial neighbors based on cell type labels.
    The labels are provided as a matrix of cell type probabilities.

    Args:
        spatial (sc.AnnData): Spatial dataset containing spatial neighbors.
        label (str): Label with key in `spatial.uns['label_transfer']` containing label probabilities.
        scale (bool): Scale aggregated spatial neighbors by the expected number of connections between cell types.

    Returns:
        sp.sparse.csr_matrix: Aggregated spatial neighbors.
    """
    label_prob = spatial.obsm[spatial.uns["label_transfer"][label]["obsm_key"]]
    labels = spatial.uns["label_transfer"][label]["labels"]

    aggregated = label_prob.T @ spatial.obsp["connectivities"] @ label_prob

    if scale:
        aggregated = _scale_by_expectation(np.array(aggregated), np.array(label_prob))

    return pd.DataFrame(aggregated, index=labels, columns=labels)


def _scale_by_expectation(
    aggregated: np.array,
    label_prob: np.array,
) -> np.array:
    """
    Scale aggregated spatial neighbors by the expected number of connections between cell types.

    Args:
        aggregated (np.array): Aggregated spatial neighbors.
        label_prob (np.array): Label probabilities.

    Returns:
        np.matrix: Normalized aggregated spatial neighbors.
    """
    agg_rand = aggregated.sum(axis=0) * np.nanmean(label_prob, axis=0).reshape(-1, 1)
    agg_norm = aggregated / agg_rand
    return agg_norm


def find_niches(
    neighbors: Union[pd.DataFrame, sp.sparse.csr_matrix],
    max_clusters: int = 10,
    n_clusters: Optional[int] = None,
    plot: bool = False,
    labels: Optional[List[str]] = None,
    return_dataframes: bool = True,
    log: logging.Logger = logging.getLogger(__name__),
    **kwargs,
) -> NicheResult:
    """
    Find niches in spatial neighbors using non-negative matrix factorization (NMF).
    The optimal number of clusters (up to `max_clusters`) is automatically determined using the elbow point method.

    Args:
        neighbors (Union[pd.DataFrame, sp.sparse.csr_matrix]): Aggregated spatial neighbors.
        max_clusters (int): Maximum number of clusters for NMF.
        n_clusters (Optional[int]): Number of clusters for NMF. If None, the optimal number is determined automatically.
        plot (bool): Plot the reconstruction error curve.
        labels (Optional[List[str]]): Cell type labels. Provide if `neighbors` is not a DataFrame.
        return_dataframes (bool): Return the results as DataFrames instead of matrices.
        log (logging.Logger): Logger object for logging messages.
        **kwargs: Additional keyword arguments passed to `sklearn.decomposition.NMF`.

    Returns:
        NicheResult: namedtuple containing the optimal number of clusters (n),
                     membership matrix (W) and features matrix (H).
    """
    if isinstance(neighbors, pd.DataFrame):
        nmat = sp.sparse.csr_matrix(neighbors.to_numpy())
    else:
        nmat = neighbors

    if n_clusters is None or plot:
        # Calculate reconstruction errors for different cluster numbers
        log.info(f"calculate reconstruction errors for {max_clusters} clusters")
        errors, models = _calculate_reconstruction_errors(nmat, max_clusters, **kwargs)

        # Find the elbow point (optimal number of clusters)
        if n_clusters is None:
            n_clusters = _find_elbow_point(errors)
            log.info(f"selected optimal number of clusters: {n_clusters}")

        # Plot the reconstruction error curve if required
        if plot:
            log.info(f"ploting reconstruction error")
            plt.plot(
                range(1, max_clusters + 1),
                errors,
                marker="o",
                label="Reconstruction Error",
            )
            plt.axvline(
                n_clusters,
                color="r",
                linestyle="--",
                label=f"Selected Clusters: {n_clusters}",
            )
            plt.xlabel("Number of Clusters")
            plt.ylabel("Reconstruction Error")
            plt.title("NMF Reconstruction Error vs. Number of Clusters")
            plt.legend()
            plt.show()

        # Return the optimal number of clusters and the corresponding model
        W, H = models[n_clusters - 1]
    else:
        # Calculate NMF with the specified number of clusters
        log.info(f"running NMF with {n_clusters} clusters")
        H, W, _ = _nmf(nmat, n_clusters, **kwargs)

    if return_dataframes:
        if labels is None:
            if isinstance(neighbors, pd.DataFrame):
                labels = neighbors.index
            else:
                raise ValueError(
                    "Labels must be provided for `return_dataframes==True`, if `neighbors` is not a DataFrame."
                )
        W = pd.DataFrame(W, index=labels, columns=range(n_clusters))
        H = pd.DataFrame(H, index=range(n_clusters), columns=labels)

    return NicheResult(n=n_clusters, membership=W, features=H)


def plot_niches(
    membership: Union[np.matrix, pd.DataFrame],
    labels: Optional[List[str]] = None,
    threshold: float = 0.5,
    only_shared: bool = False,
    scale: Optional[int] = 0,
) -> None:
    """
    Plot the niches as a bipartite graph. One type of node represents the niches/clusters,
    while the other type represents the cell types that are shared between multiple niches
    after applying a threshold to the soft-memberships (thresholding is applied after scaling
    the soft-memberships to [0, 1]).

    Args:
        membership (Union[np.matrix, pd.DataFrame]): Membership matrix.
        labels (Optional[List[str]]): Cell type labels. Required if `membership` is not a DataFrame.
        threshold (float): Threshold for soft-memberships.
        only_shared (bool): Plot only shared cell types between multiple niches.
        scale (Optional[int]): Standard-scale the soft-memberships by the specified axis (0 or 1) if not None.
    """
    if isinstance(membership, pd.DataFrame):
        labels = membership.index.tolist()
        niches = membership.columns.tolist()
        membership = membership.to_numpy()
    else:
        assert labels is not None
        assert len(labels) == membership.shape[0]
        niches = range(membership.shape[1])

    try:
        import networkx as nx
    except ImportError:
        raise ImportError(
            "networkx is required for this function, "
            "install it via `pip install networkx`"
        )

    # Scale and apply threshold to soft-memberships
    if scale is not None:
        shp = (-1, 1) if scale == 1 else (1, -1)
        membership = (membership - membership.min(axis=scale).reshape(*shp)) / (
            membership.max(axis=scale).reshape(*shp)
            - membership.min(axis=scale).reshape(*shp)
        )
    membership = (membership > threshold).astype(int)

    # Create a bipartite graph
    G = nx.Graph()
    G.add_nodes_from(labels, bipartite=0)
    G.add_nodes_from(niches, bipartite=1)
    for i, ct in enumerate(labels):
        for j in niches:
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


def plot_aggregated_neighbors(
    neighbors: Union[pd.DataFrame, sp.sparse.csr_matrix],
    membership: Union[np.matrix, pd.DataFrame],
    labels: Optional[List[str]] = None,
    niches: Optional[List[str]] = None,
    **kwargs,
) -> None:
    """
    Plot the aggregated spatial neighbors with cell type labels and niches.
    Aggregated neighbors are displayed as a seaborn clustermap, with niches as annotations.

    Args:
        neighbors (Union[pd.DataFrame, sp.sparse.csr_matrix]): Aggregated spatial neighbors.
        membership (Union[np.matrix, pd.DataFrame]): Membership matrix.
        labels (Optional[List[str]]): Cell type labels. Required if `neighbors` is not a DataFrame.
        niches (Optional[List[str]]): Niche labels. Required if `membership` is not a DataFrame.
        **kwargs: Additional keyword arguments passed to `seaborn.clustermap`.
    """
    try:
        import seaborn as sns
    except ImportError:
        raise ImportError(
            "seaborn is required for this function, "
            "install it via `pip install seaborn`"
        )

    # Check if row/col names are provided
    if not isinstance(neighbors, pd.DataFrame):
        assert labels is not None
        assert len(labels) == neighbors.shape[0]
        neighbors = pd.DataFrame(neighbors.toarray(), index=labels, columns=labels)
    if not isinstance(membership, pd.DataFrame):
        assert niches is not None
        assert len(niches) == membership.shape[1]
        membership = pd.DataFrame(membership, index=labels, columns=niches)

    # Order the niches (columns of membership matrix) based on the hierarchical clustering
    linkage_matrix = linkage(membership.to_numpy().T, method="average")
    linkage_matrix = optimal_leaf_ordering(linkage_matrix, membership.to_numpy().T)
    ordered_indices = leaves_list(linkage_matrix)
    ordered_membership = membership.iloc[:, ordered_indices]

    # Convert membership values to color values using a continuous palette
    cmap = sns.color_palette("viridis", as_cmap=True)
    row_colors = (
        ordered_membership.stack()
        .map(lambda x: cmap(x / membership.values.max()))
        .unstack()
    )

    # Plot the clustermap with niches as annotations
    sns.clustermap(
        neighbors,
        row_colors=row_colors,
        **kwargs,
    )


def _nmf(adjacency_matrix: sp.sparse.csr_matrix, k: int, **kwargs):
    # Set default keyword arguments
    use_kwargs = {"max_iter": 1000, "init": "random", "random_state": 0}
    use_kwargs.update(kwargs)

    # Run NMF
    model = NMF(n_components=k, **use_kwargs)
    W = model.fit_transform(adjacency_matrix)
    H = model.components_

    # Calculate the Frobenius norm of the difference (reconstruction error)
    reconstruction_error = np.linalg.norm(adjacency_matrix - W @ H, ord="fro")
    return H, W, reconstruction_error


def _calculate_reconstruction_errors(
    adjacency_matrix: sp.sparse.csr_matrix, max_components: int, **kwargs
) -> Tuple[List[float], List[Tuple[np.matrix, np.matrix]]]:
    """
    Calculate the reconstruction errors for different numbers of components in NMF.

    Args:
        adjacency_matrix (sp.sparse.csr_matrix): Aggregated spatial neighbors.
        max_components (int): Maximum number of components for NMF.
        **kwargs: Additional keyword arguments passed to `sklearn.decomposition.NMF`.

    Returns:
        Tuple[List[float], List[Tuple[np.matrix, np.matrix]]]: List of reconstruction errors and models.
    """
    errors = []
    models = []
    for k in range(1, max_components + 1):
        H, W, reconstruction_error = _nmf(adjacency_matrix, k, **kwargs)
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
        np.argmax(second_differences) + 1
    )  # +1 to adjust for second derivative index shift
    return elbow_index + 1  # +1 to adjust for 1-based cluster count
