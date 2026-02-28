__version__ = "0.1.0"
__author__ = "J. Patrick Pett"

from .gene_selection import select_genes
from .label_transfer import transfer_labels, add_prob_to_obs
from .niche_analysis import (
    aggregate_neighbors,
    find_niches,
    plot_niches,
    plot_aggregated_neighbors,
    niche_scores_from_nmf,
    cell_niche_scores,
    symmetrise_nmf_factors,
)
