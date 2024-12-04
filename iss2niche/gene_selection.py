from typing import Literal, List
import logging
import scanpy as sc


def select_genes(
    suspension: sc.AnnData,
    spatial: sc.AnnData,
    kind: Literal["all", "hvg"],
    log: logging.Logger = logging.getLogger(__name__),
    **kwargs,
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

    # sub-select genes
    log.info(f"select {len(genes)} overlapping genes")
    susp = suspension[:, list(genes)]
    spat = spatial[:, list(genes)]

    # sub-select genes
    try:
        if kind == "hvg":
            log.info("select highly variable genes")
            hvg_suspension = set(
                sc.pp.highly_variable_genes(susp, inplace=False, **kwargs)
                .query("highly_variable == True")
                .index.tolist()
            )
            log.info(f"  > number of HVG in suspension: {len(hvg_suspension)}")
            hvg_spatial = set(
                sc.pp.highly_variable_genes(spat, inplace=False, **kwargs)
                .query("highly_variable == True")
                .index.tolist()
            )
            log.info(f"  > number of HVG in spatial: {len(hvg_spatial)}")
            genes &= hvg_suspension & hvg_spatial
            log.info(f"  > number of HVG in overlap: {len(genes)}")
    except TypeError as e:
        e_str = f"wrong keyword argument for kind='{kind}': {e}"
        log.error(e_str)
        raise TypeError(e_str)

    return list(genes)
