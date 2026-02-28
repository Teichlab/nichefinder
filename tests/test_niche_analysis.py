"""
Tests for niche_analysis.py.

Fixtures replicate the simulated-data workflow from the simulated fig2c
dataset. The two small CSV files are stored in tests/data/ (~1200 simulated
cells, 6 cell types T0-T5).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scanpy as sc

import nichefinder as nf
from nichefinder.niche_analysis import (
    _scale_by_expectation,
    _conservative_majority,
    cell_niche_scores,
    niche_scores_from_nmf,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def simulated_adata():
    """AnnData built from the fig2c simulated CSV files."""
    positions = pd.read_csv(DATA_DIR / "tissue_positions_list.csv", header=None, index_col=0)
    labels = pd.read_csv(DATA_DIR / "save_clusterid_0.csv", header=0, index_col=0)

    ad = sc.AnnData(positions)
    ad.obsm["spatial"] = positions.to_numpy()
    ad.obs["cell_type"] = labels.iloc[:, 0].map(lambda x: f"T{x}")

    cell_type_labels = pd.get_dummies(ad.obs["cell_type"]).columns.tolist()
    ad.obsm["cell_type_prob"] = pd.get_dummies(ad.obs["cell_type"]).astype(float).values
    ad.uns["label_transfer"] = {
        "cell_type": {
            "obsm_key": "cell_type_prob",
            "labels": cell_type_labels,
            "model": None,
            "genes": None,
            "kind": "external",
        }
    }
    sc.pp.neighbors(ad, n_neighbors=5, use_rep="spatial")
    return ad


@pytest.fixture(scope="module")
def niche_matrices(simulated_adata):
    """Aggregated neighbor matrix + NMF membership & features."""
    agg = nf.aggregate_neighbors(simulated_adata, label="cell_type", scale="expected")
    result = nf.find_niches(agg, n_clusters=3, max_clusters=5)
    return simulated_adata, agg, result.membership, result.features


# ---------------------------------------------------------------------------
# _scale_by_expectation
# ---------------------------------------------------------------------------

class TestScaleByExpectation:
    def _simple_agg(self):
        rng = np.random.default_rng(0)
        A = rng.random((4, 4))
        A = A + A.T  # symmetric
        return A

    def test_no_exclusion_returns_finite(self):
        A = self._simple_agg()
        out = _scale_by_expectation(A, exclude_diagonal=0.0)
        assert np.all(np.isfinite(out))

    def test_full_exclusion_returns_finite(self):
        A = self._simple_agg()
        out = _scale_by_expectation(A, exclude_diagonal=1.0)
        assert np.all(np.isfinite(out))

    def test_interpolation_between_extremes(self):
        A = self._simple_agg()
        out0 = _scale_by_expectation(A, exclude_diagonal=0.0)
        out1 = _scale_by_expectation(A, exclude_diagonal=1.0)
        out5 = _scale_by_expectation(A, exclude_diagonal=0.5)
        # off-diagonal values at 0.5 should lie between the two extremes
        mask = ~np.eye(4, dtype=bool)
        lo = np.minimum(out0[mask], out1[mask])
        hi = np.maximum(out0[mask], out1[mask])
        assert np.all(out5[mask] >= lo - 1e-9)
        assert np.all(out5[mask] <= hi + 1e-9)

    def test_bool_true_equals_float_one(self):
        A = self._simple_agg()
        out_bool = _scale_by_expectation(A, exclude_diagonal=True)
        out_float = _scale_by_expectation(A, exclude_diagonal=1.0)
        np.testing.assert_allclose(out_bool, out_float)

    def test_bool_false_equals_float_zero(self):
        A = self._simple_agg()
        out_bool = _scale_by_expectation(A, exclude_diagonal=False)
        out_float = _scale_by_expectation(A, exclude_diagonal=0.0)
        np.testing.assert_allclose(out_bool, out_float)

    def test_out_of_range_raises(self):
        A = self._simple_agg()
        with pytest.raises(ValueError, match="between 0 and 1"):
            _scale_by_expectation(A, exclude_diagonal=1.5)
        with pytest.raises(ValueError, match="between 0 and 1"):
            _scale_by_expectation(A, exclude_diagonal=-0.1)

    def test_does_not_modify_input(self):
        A = self._simple_agg()
        A_copy = A.copy()
        _scale_by_expectation(A, exclude_diagonal=0.5)
        np.testing.assert_array_equal(A, A_copy)


# ---------------------------------------------------------------------------
# aggregate_neighbors (scale parameter)
# ---------------------------------------------------------------------------

class TestAggregateNeighbors:
    def test_scale_float_passthrough(self, simulated_adata):
        agg05 = nf.aggregate_neighbors(simulated_adata, label="cell_type", scale=0.5)
        assert isinstance(agg05, pd.DataFrame)
        assert agg05.shape[0] == agg05.shape[1]
        assert np.all(np.isfinite(agg05.values))

    def test_scale_none(self, simulated_adata):
        agg = nf.aggregate_neighbors(simulated_adata, label="cell_type", scale=None)
        assert isinstance(agg, pd.DataFrame)
        assert np.all(agg.values >= 0)

    def test_scale_expected_and_expected_no_diag_differ(self, simulated_adata):
        agg_e = nf.aggregate_neighbors(simulated_adata, label="cell_type", scale="expected")
        agg_nd = nf.aggregate_neighbors(simulated_adata, label="cell_type", scale="expected_no_diag")
        # They should not be identical
        assert not np.allclose(agg_e.values, agg_nd.values)


# ---------------------------------------------------------------------------
# find_niches – symmetric NMF
# ---------------------------------------------------------------------------

class TestFindNichesSymmetric:
    def test_returns_nichematch_namedtuple(self, simulated_adata):
        agg = nf.aggregate_neighbors(simulated_adata, label="cell_type", scale="expected")
        result = nf.find_niches(agg, n_clusters=3, symmetric=True)
        assert hasattr(result, "n") and hasattr(result, "membership") and hasattr(result, "features")
        assert result.n == 3

    def test_W_and_H_are_transposes(self, simulated_adata):
        """For symmetric NMF, features == membership.T (up to floating point)."""
        agg = nf.aggregate_neighbors(simulated_adata, label="cell_type", scale="expected")
        result = nf.find_niches(agg, n_clusters=3, symmetric=True)
        W = result.membership.to_numpy()
        H = result.features.to_numpy()
        np.testing.assert_allclose(W.T, H, atol=1e-10)

    def test_shapes(self, simulated_adata):
        agg = nf.aggregate_neighbors(simulated_adata, label="cell_type", scale="expected")
        n_types = agg.shape[0]
        result = nf.find_niches(agg, n_clusters=4, symmetric=True)
        assert result.membership.shape == (n_types, 4)
        assert result.features.shape == (4, n_types)

    def test_reconstruction_is_nonneg(self, simulated_adata):
        """H H^T should be non-negative since H >= 0."""
        agg = nf.aggregate_neighbors(simulated_adata, label="cell_type", scale="expected")
        result = nf.find_niches(agg, n_clusters=3, symmetric=True)
        W = result.membership.to_numpy()
        recon = W @ W.T
        assert np.all(recon >= -1e-9)

    def test_elbow_selection_symmetric(self, simulated_adata):
        """Automatic cluster selection should work with symmetric=True."""
        agg = nf.aggregate_neighbors(simulated_adata, label="cell_type", scale="expected")
        result = nf.find_niches(agg, max_clusters=6, symmetric=True)
        assert 1 <= result.n <= 6

    def test_standard_and_symmetric_differ(self, simulated_adata):
        """Standard and symmetric NMF should give different factorisations."""
        agg = nf.aggregate_neighbors(simulated_adata, label="cell_type", scale="expected")
        r_std = nf.find_niches(agg, n_clusters=3, symmetric=False)
        r_sym = nf.find_niches(agg, n_clusters=3, symmetric=True)
        # Standard NMF W and H are not necessarily transposes; symmetric are
        W_std = r_std.membership.to_numpy()
        H_std = r_std.features.to_numpy()
        assert not np.allclose(W_std.T, H_std, atol=1e-3), (
            "Standard NMF should not produce W.T == H (would indicate accidental symmetry)"
        )


# ---------------------------------------------------------------------------
# niche_scores_from_nmf
# ---------------------------------------------------------------------------

class TestNicheScoresFromNmf:
    def test_shapes_out(self, niche_matrices):
        ad, _, W, H = niche_matrices
        C = ad.obsp["connectivities"]
        L = ad.obsm["cell_type_prob"]
        S_out, S_in, S_mem = niche_scores_from_nmf(C, L, W, H, direction="out")
        n = C.shape[0]
        k = W.shape[1]
        assert S_out.shape == (n, k)
        assert S_in is None
        assert S_mem.shape == (n, k)

    def test_shapes_both(self, niche_matrices):
        ad, _, W, H = niche_matrices
        C = ad.obsp["connectivities"]
        L = ad.obsm["cell_type_prob"]
        S_out, S_in, S_mem = niche_scores_from_nmf(C, L, W, H, direction="both")
        n = C.shape[0]
        k = W.shape[1]
        assert S_out.shape == (n, k)
        assert S_in.shape == (n, k)
        assert S_mem.shape == (n, k)

    def test_membership_rows_sum_to_one(self, niche_matrices):
        ad, _, W, H = niche_matrices
        C = ad.obsp["connectivities"]
        L = ad.obsm["cell_type_prob"]
        _, _, S_mem = niche_scores_from_nmf(C, L, W, H, direction="out")
        row_sums = S_mem.sum(axis=1).values
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_nonnegative_scores(self, niche_matrices):
        ad, _, W, H = niche_matrices
        C = ad.obsp["connectivities"]
        L = ad.obsm["cell_type_prob"]
        S_out, _, _ = niche_scores_from_nmf(C, L, W, H, direction="out")
        assert np.all(S_out.values >= -1e-9)

    def test_bad_H_shape_raises(self, niche_matrices):
        ad, _, W, H = niche_matrices
        C = ad.obsp["connectivities"]
        L = ad.obsm["cell_type_prob"]
        with pytest.raises(ValueError, match="H must be"):
            niche_scores_from_nmf(C, L, W, H.T)  # wrong orientation


# ---------------------------------------------------------------------------
# cell_niche_scores
# ---------------------------------------------------------------------------

class TestCellNicheScores:
    def test_linear_shape(self, niche_matrices):
        ad, _, W, H = niche_matrices
        C = ad.obsp["connectivities"]
        L = ad.obsm["cell_type_prob"]
        df = cell_niche_scores(C, L, W, method="linear")
        assert df.shape == (C.shape[0], W.shape[1])
        # Columns are auto-named niche_0, niche_1, etc. when W has integer columns
        assert len(df.columns) == W.shape[1]

    def test_geometric_shape(self, niche_matrices):
        ad, _, W, H = niche_matrices
        C = ad.obsp["connectivities"]
        L = ad.obsm["cell_type_prob"]
        df = cell_niche_scores(C, L, W, method="geometric")
        assert df.shape == (C.shape[0], W.shape[1])

    def test_symmetrisation_with_H(self, niche_matrices):
        ad, _, W, H = niche_matrices
        C = ad.obsp["connectivities"]
        L = ad.obsm["cell_type_prob"]
        df_W_only = cell_niche_scores(C, L, W, H=None, method="linear")
        df_sym = cell_niche_scores(C, L, W, H=H, alpha=0.5, method="linear")
        # Different inputs should produce different scores
        assert not np.allclose(df_W_only.values, df_sym.values)

    def test_alpha_endpoints(self, niche_matrices):
        ad, _, W, H = niche_matrices
        C = ad.obsp["connectivities"]
        L = ad.obsm["cell_type_prob"]
        df_w = cell_niche_scores(C, L, W, H=H, alpha=1.0, method="linear")
        df_h = cell_niche_scores(C, L, W, H=H, alpha=0.0, method="linear")
        assert not np.allclose(df_w.values, df_h.values)

    def test_knn_modes_differ(self, niche_matrices):
        """On a symmetric graph out==in, but undirected should equal both.
        Use a manually asymmetric adjacency to check that out != in."""
        ad, _, W, H = niche_matrices
        import scipy.sparse
        # Build an asymmetric version: upper triangle only
        C_sym = ad.obsp["connectivities"]
        C_asym = scipy.sparse.triu(C_sym, k=1).tocsr()
        L = ad.obsm["cell_type_prob"]
        df_out = cell_niche_scores(C_asym, L, W, knn_mode="out", method="linear")
        df_in = cell_niche_scores(C_asym, L, W, knn_mode="in", method="linear")
        df_un = cell_niche_scores(C_asym, L, W, knn_mode="undirected", method="linear")
        assert not np.allclose(df_out.values, df_in.values)
        assert not np.allclose(df_out.values, df_un.values)

    def test_min_neighbors_filter(self, niche_matrices):
        """Cells with no outgoing edges should get NaN rows when min_neighbors > 0."""
        ad, _, W, _ = niche_matrices
        C = ad.obsp["connectivities"]
        L = ad.obsm["cell_type_prob"]
        # Force a handful of cells to look isolated by zeroing their rows
        import scipy.sparse
        C_mod = C.copy().tolil()
        C_mod[0, :] = 0
        C_mod[1, :] = 0
        C_mod = C_mod.tocsr()
        df = cell_niche_scores(C_mod, L, W, min_neighbors=1, method="linear")
        # Rows 0 and 1 should be NaN
        assert np.all(np.isnan(df.iloc[0].values))
        assert np.all(np.isnan(df.iloc[1].values))
        # Other rows should be finite
        assert np.all(np.isfinite(df.iloc[2:].values))

    def test_smoothing_adds_niche_label_column(self, niche_matrices):
        ad, _, W, _ = niche_matrices
        C = ad.obsp["connectivities"]
        L = ad.obsm["cell_type_prob"]
        df = cell_niche_scores(C, L, W, method="linear", n_smooth_iter=3)
        assert "niche_label" in df.columns
        # niche_label values must come from the score columns (not the raw W columns)
        score_cols = set(df.columns.drop("niche_label").astype(str))
        assert set(df["niche_label"].dropna().unique()).issubset(score_cols)

    def test_no_smoothing_no_niche_label_column(self, niche_matrices):
        ad, _, W, _ = niche_matrices
        C = ad.obsp["connectivities"]
        L = ad.obsm["cell_type_prob"]
        df = cell_niche_scores(C, L, W, method="linear", n_smooth_iter=0)
        assert "niche_label" not in df.columns

    def test_geometric_prob_space_nonnegative(self, niche_matrices):
        ad, _, W, _ = niche_matrices
        C = ad.obsp["connectivities"]
        L = ad.obsm["cell_type_prob"]
        df = cell_niche_scores(C, L, W, method="geometric", score_space="prob")
        assert np.all(df.values >= -1e-9)

    def test_cell_index_passthrough(self, niche_matrices):
        ad, _, W, _ = niche_matrices
        C = ad.obsp["connectivities"]
        L = ad.obsm["cell_type_prob"]
        df = cell_niche_scores(C, L, W, method="linear", cell_index=ad.obs_names)
        assert list(df.index) == list(ad.obs_names)


# ---------------------------------------------------------------------------
# _conservative_majority
# ---------------------------------------------------------------------------

class TestConservativeMajority:
    def _ring_graph(self, n=10):
        """Simple ring adjacency for predictable majority-vote behaviour."""
        import scipy.sparse
        rows = np.arange(n)
        cols = (np.arange(n) + 1) % n
        data = np.ones(n)
        A = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
        return A + A.T

    def test_converges(self):
        A = self._ring_graph(10)
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        result = _conservative_majority(A, y, n_iter=20)
        assert result.shape == y.shape

    def test_output_labels_subset_of_input(self):
        A = self._ring_graph(10)
        y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 0])
        result = _conservative_majority(A, y, n_iter=5)
        assert set(np.unique(result)).issubset({0, 1, 2})

    def test_no_change_when_stable(self):
        """All cells already agree with majority — should be unchanged."""
        import scipy.sparse
        # Fully connected 4-node graph, all label 0
        A = scipy.sparse.csr_matrix(np.ones((4, 4)) - np.eye(4))
        y = np.array([0, 0, 0, 0])
        result = _conservative_majority(A, y, n_iter=10)
        np.testing.assert_array_equal(result, y)
