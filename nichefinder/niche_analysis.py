from __future__ import annotations

from typing import Union, Dict, Any, Tuple, List, Optional, Literal
from collections import namedtuple
import logging

import numpy as np
import scanpy as sc
import scipy as sp
import scipy.sparse
from scipy.cluster.hierarchy import linkage, leaves_list, optimal_leaf_ordering
import pandas as pd

from sklearn.decomposition import NMF
import matplotlib.pyplot as plt


NicheResult = namedtuple("NicheResult", ["n", "W", "H"])

Mat = Union[np.ndarray, sp.sparse.spmatrix]


def _row_norm(C: Mat, eps: float = 1e-12) -> Mat:
    """Row-normalise a matrix (dense or sparse)."""
    if sp.sparse.issparse(C):
        C = C.tocsr()
        rs = np.asarray(C.sum(axis=1)).ravel()
        inv = np.zeros_like(rs, dtype=float)
        nz = rs > 0
        inv[nz] = 1.0 / (rs[nz] + eps)
        return sp.sparse.diags(inv) @ C
    C = np.asarray(C, dtype=float)
    rs = C.sum(axis=1, keepdims=True)
    return np.divide(C, rs + eps, out=np.zeros_like(C), where=rs > 0)


def _col_norm(C: Mat, eps: float = 1e-12) -> Mat:
    """Column-normalise a matrix (dense or sparse)."""
    if sp.sparse.issparse(C):
        C = C.tocsc()
        cs = np.asarray(C.sum(axis=0)).ravel()
        inv = np.zeros_like(cs, dtype=float)
        nz = cs > 0
        inv[nz] = 1.0 / (cs[nz] + eps)
        return C @ sp.sparse.diags(inv)
    C = np.asarray(C, dtype=float)
    cs = C.sum(axis=0, keepdims=True)
    return np.divide(C, cs + eps, out=np.zeros_like(C), where=cs > 0)


def _dense(X) -> np.ndarray:
    """Convert sparse or array-like to a dense float numpy array."""
    return np.asarray(X.todense() if sp.sparse.issparse(X) else X, dtype=float)


def aggregate_neighbors(
    spatial: sc.AnnData,
    label: str,
    scale: Optional[Union[Literal["expected", "expected_no_diag"], float]] = "expected",
) -> pd.DataFrame:
    """
    Aggregate spatial neighbors based on cell type labels.
    The labels are provided as a matrix of cell type probabilities.

    Args:
        spatial (sc.AnnData): Spatial dataset containing spatial neighbors.
        label (str): Label with key in `spatial.uns['label_transfer']` containing label probabilities.
        scale (Optional[Union[Literal['expected', 'expected_no_diag'], float]]): Controls how aggregated
            spatial neighbors are scaled by the expected number of connections between cell types.
            ``'expected'`` uses the full random-rewiring baseline (``exclude_diagonal=0.0``).
            ``'expected_no_diag'`` fully removes self-connections from the background
            (``exclude_diagonal=1.0``).
            A float in ``[0, 1]`` sets the diagonal exclusion fraction continuously.
            ``None`` skips scaling. Default is ``'expected'``.

    Returns:
        pd.DataFrame: Aggregated spatial neighbors as a cell-type × cell-type DataFrame.
    """
    label_prob = spatial.obsm[spatial.uns["label_transfer"][label]["obsm_key"]]
    labels = spatial.uns["label_transfer"][label]["labels"]

    aggregated = label_prob.T @ spatial.obsp["connectivities"] @ label_prob
    aggregated = np.array(aggregated)

    if scale == "expected":
        aggregated = _scale_by_expectation(aggregated, exclude_diagonal=0.0)
    elif scale == "expected_no_diag":
        aggregated = _scale_by_expectation(aggregated, exclude_diagonal=1.0)
    elif isinstance(scale, (int, float)) and scale is not True and scale is not False:
        aggregated = _scale_by_expectation(aggregated, exclude_diagonal=float(scale))

    return pd.DataFrame(aggregated, index=labels, columns=labels)


def _scale_by_expectation(
    aggregated: np.ndarray,
    exclude_diagonal: float = 0.0,
) -> np.ndarray:
    """
    Scale aggregated spatial neighbors by the expected number of connections between cell types.

    Args:
        aggregated (np.ndarray): Aggregated spatial neighbors.
        exclude_diagonal (float): Value in ``[0, 1]`` controlling how much of the diagonal is
            excluded when estimating the random-rewiring background.
            ``0.0`` keeps the full diagonal in the background (no exclusion).
            ``1.0`` fully removes self-connections from the background, which increases
            enrichment scores for cell types in homogeneous spatial regions.
            Intermediate values interpolate continuously between the two extremes.
            ``True``/``False`` are accepted and coerce to ``1.0``/``0.0``.

    Returns:
        np.ndarray: Normalised aggregated spatial neighbors.

    Raises:
        ValueError: If ``exclude_diagonal`` is outside ``[0, 1]``.
    """
    exclude_diagonal = float(exclude_diagonal)
    if not (0.0 <= exclude_diagonal <= 1.0):
        raise ValueError("exclude_diagonal must be a value between 0 and 1.")

    agg = aggregated.copy()

    # Proportionally subtract diagonal from background-estimation copy
    if exclude_diagonal > 0.0:
        diag = np.diag(agg)
        agg -= np.diag(diag * exclude_diagonal)

    # Estimate label probabilities from (partially) diagonal-excluded matrix
    label_prob = agg.sum(axis=0)
    label_prob /= label_prob.sum()

    # Calculate expected connections based on overall label frequencies
    agg_rand = aggregated.sum(axis=0) * label_prob.reshape(-1, 1)

    # Interpolate diagonal of agg_rand between random expectation and observed
    agg_rand[np.diag_indices_from(agg_rand)] = np.diag(
        exclude_diagonal * aggregated + (1.0 - exclude_diagonal) * agg_rand
    )

    return aggregated / agg_rand


def find_niches(
    neighbors: Union[pd.DataFrame, sp.sparse.csr_matrix],
    max_clusters: int = 10,
    n_clusters: Optional[int] = None,
    plot: bool = False,
    labels: Optional[List[str]] = None,
    return_dataframes: bool = True,
    symmetric: bool = False,
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
        symmetric (bool): Run symmetric NMF (SymNMF) which factorises the neighbour
            matrix as ``A ≈ H Hᵀ`` instead of the standard ``A ≈ W H``.  This
            guarantees that the returned membership ``W = H`` and features
            ``H = Hᵀ`` are transposes of each other. Accepted ``kwargs``
            for the symmetric solver are ``max_iter`` (default ``1000``),
            ``random_state`` (default ``0``), and ``tol`` (default ``1e-4``).
            Default ``False``.
        log (logging.Logger): Logger object for logging messages.
        **kwargs: Additional keyword arguments passed to `sklearn.decomposition.NMF`
            (standard NMF) or accepted as solver hyper-parameters (symmetric NMF).

    Returns:
        NicheResult: namedtuple containing the optimal number of clusters (n),
                     membership matrix (W) and features matrix (H).  For symmetric
                     NMF, ``W`` is the factor ``H`` of the ``A ≈ H Hᵀ``
                     decomposition and ``result.features`` is its transpose, so
                     ``W @ result.features ≈ A``.
    """
    if isinstance(neighbors, pd.DataFrame):
        nmat = sp.sparse.csr_matrix(neighbors.to_numpy())
    else:
        nmat = neighbors

    if n_clusters is None or plot:
        # Calculate reconstruction errors for different cluster numbers
        log.info(f"calculate reconstruction errors for {max_clusters} clusters")
        errors, models = _calculate_reconstruction_errors(
            nmat, max_clusters, symmetric=symmetric, **kwargs
        )

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
        if symmetric:
            log.info(f"running symmetric NMF with {n_clusters} clusters")
            W, _ = _symnmf(nmat, n_clusters, **kwargs)
            H = W.T
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

    return NicheResult(n=n_clusters, W=W, H=H)


def plot_niches(
    membership: Union[np.ndarray, pd.DataFrame],
    labels: Optional[List[str]] = None,
    threshold: float = 0.5,
    only_shared: bool = False,
    scale: Optional[int] = 0,
    edge_width_scale: Optional[float] = None,
) -> None:
    """
    Plot the niches as a bipartite graph. One type of node represents the niches/clusters,
    while the other type represents the cell types that are shared between multiple niches
    after applying a threshold to the soft-memberships (thresholding is applied after scaling
    the soft-memberships to [0, 1]).

    Args:
        membership (Union[np.ndarray, pd.DataFrame]): Membership matrix (cell types × niches).
        labels (Optional[List[str]]): Cell type labels. Required if `membership` is not a DataFrame.
        threshold (float): Threshold for soft-memberships.
        only_shared (bool): Plot only shared cell types between multiple niches.
        scale (Optional[int]): Standard-scale the soft-memberships by the specified axis (0 or 1) if not None.
        edge_width_scale (Optional[float]): If set, edge widths are scaled by
            ``edge_width_scale * membership_weight``.
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

    # Scale soft-memberships
    if scale is not None:
        shp = (-1, 1) if scale == 1 else (1, -1)
        membership = (membership - membership.min(axis=scale).reshape(*shp)) / (
            membership.max(axis=scale).reshape(*shp)
            - membership.min(axis=scale).reshape(*shp)
        )

    # Preserve continuous weights before thresholding
    weights = membership.copy()

    # Apply threshold for edge existence
    membership = (membership > threshold).astype(int)

    # Create a bipartite graph
    G = nx.Graph()
    G.add_nodes_from(labels, bipartite=0)
    G.add_nodes_from(niches, bipartite=1)
    # Map niche labels to their integer indices for correct access
    niche_to_idx = {n: idx for idx, n in enumerate(niches)}
    for i, ct in enumerate(labels):
        for j in niches:
            j_idx = niche_to_idx[j]
            if membership[i, j_idx] == 1:
                G.add_edge(ct, j, weight=weights[i, j_idx])

    # Remove cell types that are not shared between multiple niches
    degree_thr = 2 if only_shared else 1
    nodes_to_remove = [
        node for node, degree in G.degree()
        if node in labels and degree < degree_thr
    ]
    G.remove_nodes_from(nodes_to_remove)

    # Plot the bipartite graph using a kamada-kawai layout
    pos = nx.kamada_kawai_layout(G)

    edge_widths = None
    if edge_width_scale is not None:
        edge_widths = [edge_width_scale * G[u][v]["weight"] for u, v in G.edges()]

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=300,
        node_color="skyblue",
        edge_color="gray",
        width=edge_widths,
    )
    plt.title("Niche Analysis")


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
    """
    Run standard NMF on the adjacency matrix and return the factors and reconstruction error.
    
    Args:
        adjacency_matrix (sp.sparse.csr_matrix): Aggregated spatial neighbors.
        k (int): Number of components for NMF.
        **kwargs: Additional keyword arguments passed to `sklearn.decomposition.NMF`.

    Returns:
        Tuple[np.ndarray, np.ndarray, float]: (H, W, reconstruction_error) where H is the
        (k, n_types) feature matrix, W is the (n_types, k) membership matrix, and
        reconstruction_error is the Frobenius norm of the difference between the
        original adjacency matrix and the reconstructed matrix W @ H.
    """
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


def _symnmf(
    adjacency_matrix,
    k: int,
    max_iter: int = 1000,
    tol: float = 1e-4,
    random_state: int = 0,
    **_ignored,
) -> Tuple[np.ndarray, float]:
    """
    Implementation of symmetric NMF. 
    
    Factorises ``A ≈ H Hᵀ`` via multiplicative updates with column renormalisation.
    Each iteration applies the standard MU rule from Kuang et al. (2012)::

        H_new <- H * (A H) / (H Hᵀ H + eps)

    and then normalises every column of ``H_new`` to unit L2 norm, storing
    the column scales in ``d``.  Renormalisation prevents columns from
    collapsing to zero (a known stagnation mode of plain MU) and keeps the
    iterates in a bounded region without affecting the monotone decrease of
    the objective.  On convergence, the column scales are restored so that
    the returned ``H`` satisfies ``H @ H.T ≈ A``.

    If the input matrix is not symmetric it is symmetrised as
    ``A = (A + Aᵀ) / 2`` before factorisation and a ``UserWarning`` is
    emitted.

    Any keyword arguments not listed below are silently ignored so that
    ``find_niches`` can forward its ``**kwargs`` without splitting them by
    solver.

    References:
        Kuang, D., Ding, C., & Park, H. (2012). Symmetric nonnegative matrix
        factorization for graph clustering. *Proceedings of the 2012 SIAM
        International Conference on Data Mining (SDM)*, pp. 106–117.
        https://doi.org/10.1137/1.9781611972825.10

        Kuang, D., Yun, S., & Park, H. (2015). SymNMF: Nonnegative low-rank
        approximation of a similarity matrix for graph clustering.
        *Journal of Global Optimization*, 62(3), 545–574.
        https://doi.org/10.1007/s10898-014-0247-2

    Args:
        adjacency_matrix: (n, n) non-negative symmetric matrix (dense or sparse).
            Non-symmetric inputs are automatically symmetrised.
        k (int): Number of components / niches.
        max_iter (int): Maximum number of multiplicative-update iterations.
            Default ``1000``.
        tol (float): Convergence tolerance on the relative change in the
            normalised ``H`` between iterations. Default ``1e-4``.
        random_state (int): Seed for the random initialisation of ``H``.
            Default ``0``.
        **_ignored: Extra keyword arguments are accepted but silently ignored
            so callers can pass standard NMF kwargs without error.

    Returns:
        Tuple ``(H, reconstruction_error)`` where ``H`` has shape (n, k) and
        ``reconstruction_error = ||A - H Hᵀ||_F``.
    """
    import warnings

    A = _dense(adjacency_matrix).astype(float)
    n = A.shape[0]

    # SymNMF requires a symmetric input
    if not np.allclose(A, A.T, atol=1e-8, rtol=1e-5):
        warnings.warn(
            "Input matrix passed to _symnmf is not symmetric! "
            "Symmetrising as A = (A + A.T) / 2 before factorisation.",
            UserWarning,
            stacklevel=3,
        )
        A = 0.5 * (A + A.T)

    rng = np.random.default_rng(random_state)
    H = rng.random((n, k)) + 1e-2

    eps = 1e-12
    # Initialise column scales to 1
    d = np.ones(k, dtype=float)

    for _ in range(max_iter):
        HtH = H.T @ H           # (k, k)
        numerator = A @ H       # (n, k)
        denominator = H @ HtH   # (n, k)
        H_new = H * (numerator / (denominator + eps))

        # Column renormalisation (prevent collapse / unbounded growth)
        col_norms = np.linalg.norm(H_new, axis=0)
        col_norms = np.maximum(col_norms, eps)
        d = d * col_norms                          # accumulate scales
        H_new = H_new / col_norms[np.newaxis, :]   # normalised iterate

        rel_change = np.linalg.norm(H_new - H) / (np.linalg.norm(H) + eps)
        H = H_new
        if rel_change < tol:
            break

    # Restore column scales so that H @ H.T ~= A
    H = H * d[np.newaxis, :]
    reconstruction_error = np.linalg.norm(A - H @ H.T, ord="fro")
    return H, reconstruction_error


def _calculate_reconstruction_errors(
    adjacency_matrix: sp.sparse.csr_matrix,
    max_components: int,
    symmetric: bool = False,
    **kwargs,
) -> Tuple[List[float], List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Calculate the reconstruction errors for different numbers of components in NMF.

    Args:
        adjacency_matrix (sp.sparse.csr_matrix): Aggregated spatial neighbors.
        max_components (int): Maximum number of components for NMF.
        symmetric (bool): Use symmetric NMF instead of standard NMF. Default ``False``.
        **kwargs: Additional keyword arguments passed to the chosen NMF solver.

    Returns:
        Tuple[List[float], List[Tuple[np.ndarray, np.ndarray]]]: List of reconstruction
        errors and ``(W, H)`` model pairs. For symmetric NMF, ``W = H_factor`` and
        ``H = H_factor.T``.
    """
    errors = []
    models = []
    for k in range(1, max_components + 1):
        if symmetric:
            W, reconstruction_error = _symnmf(adjacency_matrix, k, **kwargs)
            H = W.T
        else:
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


def symmetrise_nmf_factors(
    W: np.ndarray,
    H: np.ndarray,
    alpha: float = 0.5,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Combine asymmetric NMF factors ``W`` and ``H`` into a single symmetric
    loading matrix suitable for neighbourhood scoring.

    The two factors are column-normalised and blended::

        S = alpha * col_norm(W) + (1 - alpha) * col_norm(H.T)

    where ``W`` has shape (n_types, n_niches) and ``H`` has shape
    (n_niches, n_types), so ``H.T`` has the same shape as ``W``.

    ``alpha=0.5`` gives equal sender/receiver weight; ``alpha=1.0`` uses
    only ``W`` (sender perspective); ``alpha=0.0`` uses only ``H.T``
    (receiver perspective).

    Args:
        W (np.ndarray): (n_types, n_niches) sender-type loading matrix.
        H (np.ndarray): (n_niches, n_types) receiver-type loading matrix.
        alpha (float): Blending weight in ``[0, 1]``. Default ``0.5``.
        eps (float): Small constant for numerical stability. Default ``1e-12``.

    Returns:
        np.ndarray: (n_types, n_niches) symmetrised loading matrix ``S``.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1]; got {alpha}.")
    Wc = _col_norm(W, eps=eps)
    Hc = _col_norm(H.T, eps=eps)  # H.T: (n_types, n_niches)
    return alpha * Wc + (1.0 - alpha) * Hc


# ---------------------------------------------------------------------------
# Per-cell niche scoring
# ---------------------------------------------------------------------------

def cell_niche_scores(
    adjacency: Mat,
    L: Mat,
    W: Union[np.ndarray, pd.DataFrame],
    H: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    *,
    method: Literal["linear", "geometric"] = "linear",
    alpha: float = 0.5,
    adjacency_mode: Literal["out", "in", "undirected"] = "undirected",
    normalize_adjacency: bool = True,
    normalize_S: bool = True,
    normalize_N: bool = True,
    s_sharpen: float = 1.0,
    score_space: Literal["prob", "log"] = "prob",
    n_smooth_iter: int = 0,
    smooth_margin: float = 1.0,
    min_neighbors: int = 0,
    eps: float = 1e-12,
    cell_index: Optional[pd.Index] = None,
    niche_names: Optional[List[str]] = None,
    add_self_loops: bool = True,
) -> pd.DataFrame:
    """
    Compute per-cell niche scores by aggregating neighbourhood type compositions.

    Two scoring methods are available:

    - ``"linear"``:  ``Z = (P @ L) @ S`` - each cell's score is the linear
      combination of its neighbours' type fractions weighted by the niche profile.
    - ``"geometric"``:  ``Z = exp( log(P @ L + eps) @ S )`` - weighted geometric
      mean (missing niche components strongly suppress the score; AND-like logic).

    When both ``W`` (n_types x n_niches) and ``H`` (n_niches x n_types) are
    provided (i.e. asymmetric NMF output), they are symmetrised before scoring:

    .. math::

        S = \\alpha \\cdot \\text{col\\_norm}(W)
            + (1-\\alpha) \\cdot \\text{col\\_norm}(H^T)

    ``alpha=0.5`` gives equal sender/receiver weight; ``alpha=1.0`` is W-only
    (sender perspective); ``alpha=0.0`` is H.T-only (receiver perspective).
    When ``H=None``, ``S = col_norm(W)`` (or ``W`` if ``normalize_S=False``).

    Args:
        adjacency (Mat): (n_cells, n_cells) adjacency matrix (i→j if ``adjacency[i,j] > 0``).
        adjacency_mode (str): Which adjacency mode to use: ``'out'`` (sender), ``'in'``
            (receiver), or ``'undirected'``. Default ``'undirected'``.
        normalize_adjacency (bool): Row-normalise ``adjacency`` before computing scores.
            Default ``True``.
        L (Mat): (n_cells, n_types) cell-type probabilities; rows should sum to 1.
        W (array-like or DataFrame): (n_types, n_niches) niche-type loading matrix
            (e.g. the membership matrix from :func:`find_niches`).
        H (array-like or DataFrame, optional): (n_niches, n_types) transpose factor
            from NMF. When provided, symmetrised with ``W`` via ``alpha``.
        method (str): ``'linear'`` or ``'geometric'``. Default ``'linear'``.
        alpha (float): Blending weight for W vs H.T when ``H`` is provided.
            Default ``0.5``.
        knn_mode (str): How to build the neighbourhood operator *P*:
            ``'out'`` — row-normalise ``C`` (outgoing edges);
            ``'in'`` — row-normalise ``C.T`` (incoming edges);
            ``'undirected'`` — average of the two. Default ``'undirected'``.
        normalize_knn (bool): Whether to row-normalise the kNN operator.
            Default ``True``.
        normalize_S (bool): Column-normalise the niche-type matrix ``S``
            before scoring. Default ``True``.
        normalize_N (bool): Row-normalise the neighbourhood type composition
            ``N = P @ L`` before scoring (geometric method only). Default ``True``.
        s_sharpen (float): Raise ``S`` to this power before normalisation
            to emphasise core cell types (geometric method only). ``1.0`` is
            no sharpening. Default ``1.0``.
        score_space (str): ``'prob'`` returns ``exp(log_score)``;
            ``'log'`` returns the log-score directly (geometric method only).
            Default ``'prob'``.
        n_smooth_iter (int): Number of conservative majority-vote smoothing
            iterations applied to hard niche labels. ``0`` disables smoothing.
            When ``> 0``, an additional column ``'niche_label'`` with the
            (optionally smoothed) hard assignment is added to the output.
            Default ``0``.
        smooth_margin (float): Minimum score-sum advantage required to flip a
            cell's label during majority-vote smoothing. Default ``1.0``.
        min_neighbors (int): Cells with strictly fewer kNN neighbours than this
            threshold receive ``NaN`` scores. Default ``0`` (no filtering).
        eps (float): Small constant for numerical stability. Default ``1e-12``.
        cell_index (pd.Index, optional): Index for the output DataFrame rows.
        niche_names (list of str, optional): Names for the niche columns.
        add_self_loops (bool): Whether to add self-loops to the connectivity matrix. Default is True.

    Returns:
        pd.DataFrame: (n_cells x n_niches) score matrix. Columns are named by
        ``niche_names`` (inferred from ``W`` if it is a DataFrame). An extra
        ``'niche_label'`` column is appended when ``n_smooth_iter > 0``.
    """
    if add_self_loops:
        # Add self-loops to a copy of the adjacency matrix to avoid side effects
        if sp.sparse.issparse(adjacency):
            adjacency = adjacency.copy() + sp.sparse.eye(adjacency.shape[0], format=adjacency.format)
        else:
            adjacency = adjacency.copy() + np.eye(adjacency.shape[0])

    # resolve W/H to a single S matrix
    W_df = W if isinstance(W, pd.DataFrame) else pd.DataFrame(np.asarray(W, dtype=float))
    Wn = W_df.to_numpy(dtype=float)

    if H is not None:
        H_df = H if isinstance(H, pd.DataFrame) else pd.DataFrame(np.asarray(H, dtype=float))
        Hn = H_df.to_numpy(dtype=float)  # shape (n_niches, n_types)
        if Hn.shape != (Wn.shape[1], Wn.shape[0]):
            raise ValueError(
                f"H must be (n_niches={Wn.shape[1]}, n_types={Wn.shape[0]}); got {Hn.shape}."
            )
        S = symmetrise_nmf_factors(Wn, Hn, alpha=alpha, eps=eps)
    else:
        S = Wn.copy()

    if niche_names is None:
        niche_names = (
            list(W_df.columns)
            if W_df.columns.dtype != "int64"
            else [f"niche_{i}" for i in range(S.shape[1])]
        )
        if len(niche_names) != S.shape[1]:
            niche_names = [f"niche_{i}" for i in range(S.shape[1])]

    if (S < 0).any() and method == "geometric":
        raise ValueError("S_type_niche must be nonneg for method='geometric'.")

    # optionally normalize S
    if normalize_S:
        if method == "geometric" and s_sharpen != 1.0:
            S = np.power(S, s_sharpen)
        S = _col_norm(S, eps=eps)
    elif method == "geometric" and s_sharpen != 1.0:
        S = np.power(S, s_sharpen)

    # build kNN operator P
    if not normalize_adjacency:
        if adjacency_mode == "out":
            P = adjacency
        elif adjacency_mode == "in":
            P = adjacency.T
        else:
            P = 0.5 * (adjacency + adjacency.T)
    else:
        if adjacency_mode == "out":
            P = _row_norm(adjacency, eps=eps)
        elif adjacency_mode == "in":
            P = _row_norm(adjacency.T, eps=eps)
        else:
            P = 0.5 * (_row_norm(adjacency, eps=eps) + _row_norm(adjacency.T, eps=eps))

    # min-neighbour filter mask
    if min_neighbors > 0:
        n_nbrs = np.asarray((adjacency > 0).sum(axis=1)).ravel()
        mask_nan = n_nbrs < min_neighbors
    else:
        mask_nan = None

    # neighbourhood type composition
    N = P @ L
    N = _dense(N)

    # score
    if method == "linear":
        if normalize_N:
            N = N / (N.sum(axis=1, keepdims=True) + eps)
        Z = N @ S
    elif method == "geometric":
        if normalize_N:
            N = N / (N.sum(axis=1, keepdims=True) + eps)
        log_score = np.log(N + eps) @ S
        Z = np.exp(log_score) if score_space == "prob" else log_score
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'linear' or 'geometric'.")

    # apply min-neighbour mask
    if mask_nan is not None:
        Z[mask_nan] = np.nan

    idx = cell_index if cell_index is not None else pd.RangeIndex(Z.shape[0])
    df = pd.DataFrame(Z, index=idx, columns=niche_names)

    # optional majority-vote label smoothing
    if n_smooth_iter > 0:
        # Compute hard labels from argmax (skip NaN rows)
        valid = ~np.isnan(Z).any(axis=1)
        hard_int = np.full(Z.shape[0], -1, dtype=int)
        hard_int[valid] = np.argmax(Z[valid], axis=1)

        if valid.any():
            # Only smooth the valid subset
            C_csr = adjacency.tocsr() if sp.sparse.issparse(adjacency) else sp.sparse.csr_matrix(adjacency)
            smoothed = _conservative_majority(
                C_csr[np.ix_(valid, valid)],
                hard_int[valid],
                n_iter=n_smooth_iter,
                margin=smooth_margin,
            )
            hard_int[valid] = smoothed

        niche_label = np.full(Z.shape[0], None, dtype=object)
        for i in range(Z.shape[0]):
            if hard_int[i] >= 0:
                niche_label[i] = niche_names[hard_int[i]]

        df["niche_label"] = niche_label

    return df


def niche_scores_from_nmf(
    C: Mat,
    L: Mat,
    W: Union[np.ndarray, pd.DataFrame],
    H: Union[np.ndarray, pd.DataFrame],
    *,
    direction: Literal["out", "in", "both"] = "out",
    normalize_C_for_scoring: bool = True,
    return_membership: bool = True,
    eps: float = 1e-12,
    return_dataframes: bool = True,
) -> Tuple[
    Optional[Union[np.ndarray, pd.DataFrame]],
    Optional[Union[np.ndarray, pd.DataFrame]],
    Optional[Union[np.ndarray, pd.DataFrame]],
]:
    """
    Map NMF niches learned on ``A = L^T C L`` back to per-cell niche scores.

    This function implements the pointwise-product scoring that explicitly captures
    sender/receiver roles from directed NMF factorisations. For undirected NMF
    (symmetric input matrix), consider averaging ``W`` and ``H.T`` into a single
    ``S`` matrix and using :func:`cell_niche_scores` instead, or pass
    ``direction="both"`` and use ``S_mem`` as a symmetrised view.

    Args:
        C (Mat): (n_cells, n_cells) directed adjacency (i→j if ``C[i,j] > 0``).
        L (Mat): (n_cells, n_types) cell-type probabilities; rows should sum to 1.
        W (array-like or DataFrame): (n_types, n_niches) — sender-type loadings from NMF.
        H (array-like or DataFrame): (n_niches, n_types) — receiver-type loadings from NMF.
        direction (str): Which role scores to compute: ``'out'`` (sender), ``'in'``
            (receiver), or ``'both'``. Default ``'out'``.
        normalize_C_for_scoring (bool): Row-normalise ``C`` before computing scores.
            Default ``True``.
        return_membership (bool): Whether to compute and return row-normalised
            membership ``S_mem``. Default ``True``.
        eps (float): Small constant for numerical stability. Default ``1e-12``.
        return_dataframes (bool): Return results as DataFrames. Default ``True``.

    Returns:
        Tuple of ``(S_out, S_in, S_mem)`` where each element is ``None`` if not
        requested. ``S_out`` and ``S_in`` are (n_cells, n_niches) score matrices.
        ``S_mem`` is the row-normalised mixture of the requested direction(s).

    Shapes:
        - ``S_out[i, k]``: cell *i* fits niche *k* as a sender.
        - ``S_in[i, k]``:  cell *i* fits niche *k* as a receiver.
        - ``S_mem[i, k]``: soft niche membership (sums to 1 across niches).
    """
    def _as_np(x, name: str) -> np.ndarray:
        a = x.to_numpy() if isinstance(x, pd.DataFrame) else np.asarray(x)
        if a.ndim != 2:
            raise ValueError(f"{name} must be 2D, got {a.shape}.")
        return a

    def _df(X: np.ndarray, cols: List[str]) -> Union[pd.DataFrame, np.ndarray]:
        return pd.DataFrame(X, columns=cols) if return_dataframes else X

    # Convert W/H and derive niche column names
    W_df = W if isinstance(W, pd.DataFrame) else pd.DataFrame(_as_np(W, "W"))
    H_df = H if isinstance(H, pd.DataFrame) else pd.DataFrame(_as_np(H, "H"))
    Wn, Hn = W_df.to_numpy(), H_df.to_numpy()

    niche_cols = (
        list(W_df.columns)
        if W_df.columns.dtype != "int64"
        else [f"niche_{i}" for i in range(Wn.shape[1])]
    )
    if len(niche_cols) != Wn.shape[1]:
        niche_cols = [f"niche_{i}" for i in range(Wn.shape[1])]

    # Shape validation
    if C.shape[0] != C.shape[1]:
        raise ValueError(f"C must be square; got {C.shape}.")
    n = C.shape[0]
    if L.shape[0] != n:
        raise ValueError(f"L rows must match C; got L {L.shape}, C {C.shape}.")
    t = L.shape[1]
    if Wn.shape[0] != t:
        raise ValueError(f"W rows must match n_types={t}; got W {Wn.shape}.")
    k = Wn.shape[1]
    if Hn.shape != (k, t):
        raise ValueError(f"H must be (n_niches, n_types)=({k},{t}); got H {Hn.shape}.")

    # OUT scores: S_out = (L W) * ((P_out @ L) @ H^T)
    S_out = None
    if direction in ("out", "both"):
        P_out = _row_norm(C, eps=eps) if normalize_C_for_scoring else C
        U = _dense(L @ Wn)                   # (n, k)
        V = _dense(P_out @ L) @ Hn.T         # (n, k)
        S_out = U * V

    # IN scores: S_in = (L H^T) * ((P_in^T @ L) @ W)
    S_in = None
    if direction in ("in", "both"):
        P_in = _col_norm(C, eps=eps) if normalize_C_for_scoring else C
        U = _dense(L @ Hn.T)                 # (n, k)
        V = _dense(P_in.T @ L) @ Wn          # (n, k)
        S_in = U * V

    # Membership: row-normalised sum of requested directions
    S_mem = None
    if return_membership:
        if direction == "out":
            base = S_out
        elif direction == "in":
            base = S_in
        else:
            base = (0.0 if S_out is None else S_out) + (0.0 if S_in is None else S_in)
        den = base.sum(axis=1, keepdims=True) + eps
        S_mem = base / den

    return (
        _df(S_out, niche_cols) if S_out is not None else None,
        _df(S_in, niche_cols) if S_in is not None else None,
        _df(S_mem, niche_cols) if S_mem is not None else None,
    )


# ---------------------------------------------------------------------------
# Private majority-vote smoothing helpers
# ---------------------------------------------------------------------------

def _csr_row_argmax(V_csr: sp.sparse.csr_matrix, fallback_idx: np.ndarray) -> np.ndarray:
    """Return per-row argmax of a CSR matrix, using ``fallback_idx`` for empty rows."""
    V = V_csr.tocsr()
    N = V.shape[0]
    best = fallback_idx.copy()
    indptr, indices, data = V.indptr, V.indices, V.data
    for i in range(N):
        s, e = indptr[i], indptr[i + 1]
        if s == e:
            continue
        j = indices[s:e]
        d = data[s:e]
        best[i] = j[np.argmax(d)]
    return best


def _conservative_majority(
    A: sp.sparse.spmatrix,
    y: np.ndarray,
    n_iter: int = 10,
    margin: float = 1.0,
    remove_self_loops: bool = False,
) -> np.ndarray:
    """
    Iterative conservative majority-vote label smoothing on a sparse graph.

    A cell's label is flipped only when a neighbour label accumulates at least
    ``margin`` more votes than the current label, ensuring conservative updates.

    Args:
        A (spmatrix): (n, n) adjacency / kNN weight matrix.
        y (np.ndarray): Integer label vector of length n.
        n_iter (int): Maximum number of smoothing iterations. Default ``10``.
        margin (float): Minimum vote-score improvement required to flip. Default ``1.0``.
        remove_self_loops (bool): Remove diagonal entries from ``A`` before
            smoothing. Useful when the kNN graph includes self-connections.
            Default ``False``.

    Returns:
        np.ndarray: Smoothed integer label vector (same label space as ``y``).
    """
    y = np.asarray(y).copy()
    labs, y = np.unique(y, return_inverse=True)
    K = labs.size
    N = y.size

    A = A.tocsr() if sp.sparse.issparse(A) else sp.sparse.csr_matrix(A)

    if remove_self_loops:
        A = A.copy()
        A.setdiag(0)
        A.eliminate_zeros()

    rows = np.arange(N)

    for _ in range(n_iter):
        Y = sp.sparse.csr_matrix(
            (np.ones(N, dtype=A.dtype), (rows, y)), shape=(N, K)
        )
        V = (A @ Y).tocsr()

        cur_score = np.asarray(V.multiply(Y).sum(axis=1)).ravel()
        best_score = np.asarray(V.max(axis=1).toarray()).ravel()
        best = _csr_row_argmax(V, fallback_idx=y)

        y_new = y.copy()
        change = (best != y) & ((best_score - cur_score) >= margin)
        y_new[change] = best[change]

        if np.array_equal(y_new, y):
            break
        y = y_new

    return labs[y]
