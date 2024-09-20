from typing import Literal, List
import scanpy as sc


def select_genes(
    suspension: sc.AnnData, spatial: sc.AnnData, kind: Literal["all", "hvg"], **kwargs
) -> List[str]:
    """
    Select genes from spatial and suspension datasets.
    The overlap of genes from both datasets is selected.
    In addition, genes are sub-selected based on the choice of `kind`.

    Args:
        spatial (sc.AnnData): Spatial dataset containing gene expression data.
        suspension (sc.AnnData): Suspension dataset containing gene expression data.
        kind (Literal["all", "hvg"]): Type of gene selection. "all" selects all overlapping genes,
                                      "hvg" selects highly variable genes (HVG) from both datasets.
        **kwargs: Additional keyword arguments passed to downstream functions, depending on `kind`.

    Returns:
        List[str]: List of selected gene names.

    Raises:
        TypeError: If an unknown keyword argument is passed depending on `kind`.
    """
    # get overlapping genes
    genes = set(spatial.var_names.intersection(suspension.var_names).tolist())

    susp = suspension[:, genes]
    spat = spatial[:, genes]

    # sub-select genes
    try:
        if kind == "hvg":
            hvg_suspension = set(
                sc.pp.highly_variable_genes(
                    susp, inplace=False, **kwargs
                ).index.tolist()
            )
            hvg_spatial = set(
                sc.pp.highly_variable_genes(
                    spat, inplace=False, **kwargs
                ).index.tolist()
            )
            genes &= hvg_suspension & hvg_spatial
    except TypeError as e:
        raise TypeError(f"wrong keyword argument for kind='{kind}': {e}")

    return list(genes)
