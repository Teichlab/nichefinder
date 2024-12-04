from typing import Literal, List, Optional
import logging
import scanpy as sc


def select_genes(
    suspension: sc.AnnData,
    spatial: sc.AnnData,
    kind: Literal["all", "hvg"],
    kind_susp: Optional[Literal["all", "hvg"]] = None,
    kind_spat: Optional[Literal["all", "hvg"]] = None,
    log: logging.Logger = logging.getLogger(__name__),
    kwargs_susp: Optional[dict] = None,
    kwargs_spat: Optional[dict] = None,
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
                                      Used as default for `kind_susp` and `kind_spat` if they are None.
        kind_susp (Optional[Literal["all", "hvg"]]): Type of gene selection for suspension data.
        kind_spat (Optional[Literal["all", "hvg"]]): Type of gene selection for spatial data.
        log (logging.Logger): Logger object for logging messages.
        kwargs_susp: Additional keyword arguments passed to downstream functions, depending on `kind_susp`.
        kwargs_spat: Additional keyword arguments passed to downstream functions, depending on `kind_spat`.

    Returns:
        List[str]: List of selected gene names.

    Raises:
        TypeError: If an unknown keyword argument is passed depending on `kind`.
    """
    # set default kind for suspension and spatial data
    kind_susp = kind_susp or kind
    kind_spat = kind_spat or kind
    kwargs_susp = kwargs_susp or {}
    kwargs_spat = kwargs_spat or {}

    # get overlapping genes
    genes = set(spatial.var_names.intersection(suspension.var_names).tolist())

    # sub-select genes
    log.info(f"select {len(genes)} overlapping genes")
    susp = suspension[:, list(genes)]
    spat = spatial[:, list(genes)]

    # sub-select genes in suspension data
    try:
        if kind_susp == "hvg":
            log.info("select highly variable genes for suspension data")
            hvg_suspension = set(
                sc.pp.highly_variable_genes(susp, inplace=False, **kwargs_susp)
                .query("highly_variable == True")
                .index.tolist()
            )
            log.info(f"  > number of HVG in suspension: {len(hvg_suspension)}")
            genes &= hvg_suspension
    except TypeError as e:
        e_str = f"wrong keyword argument for kind_spat='{kind_susp}': {e}"
        log.error(e_str)
        raise TypeError(e_str)

    # sub-select genes in spatial data
    try:
        if kind_spat == "hvg":
            log.info("select highly variable genes for spatial data")
            hvg_spatial = set(
                sc.pp.highly_variable_genes(spat, inplace=False, **kwargs_spat)
                .query("highly_variable == True")
                .index.tolist()
            )
            log.info(f"  > number of HVG in spatial: {len(hvg_spatial)}")
            genes &= hvg_spatial
    except TypeError as e:
        e_str = f"wrong keyword argument for kind_spat='{kind_spat}': {e}"
        log.error(e_str)
        raise TypeError(e_str)

    log.info(f"number of selected genes: {len(genes)}")

    return list(genes)
