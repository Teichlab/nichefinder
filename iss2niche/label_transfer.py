import os
import logging
from copy import deepcopy
from pathlib import Path
from typing import Literal, Optional, List, Union
import joblib
import numpy as np
import numpy_groupies as npg
import scipy.sparse as sp
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.base import ClassifierMixin

import scanpy as sc


def transfer_labels(
    suspension: sc.AnnData,
    spatial: sc.AnnData,
    genes: Optional[List[str]] = None,
    labels: Union[str, List[str]] = "cell_type",
    kind: Literal["logreg"] = "logreg",
    log: logging.Logger = logging.getLogger(__name__),
    train_kwargs: Optional[dict] = None,
    predict_kwargs: Optional[dict] = None,
) -> sc.AnnData:
    """
    Transfer labels from suspension to spatial dataset using a logistic regression model.

    Args:
        suspension (sc.AnnData): Suspension dataset containing cell labels.
        spatial (sc.AnnData): Spatial dataset containing gene expression data.
        genes (Optional[List[str]]): List of genes to use for label transfer.
        labels (Union[str, List[str]]): Name of the column in `suspension.obs` containing labels.
        kind (Literal['logreg']): Type of label transfer method.
        log (logging.Logger): Logger object.
        train_kwargs (Optional[dict]): Keyword arguments passed to training function.
        predict_kwargs (Optional[dict]): Keyword arguments passed to prediction function.

    Returns:
        sc.AnnData: Spatial dataset with transferred labels.

    Raises:
        TypeError: If an unknown keyword argument is passed depending on `kind`.
    """
    if isinstance(labels, str):
        labels = [labels]
    if train_kwargs is None:
        train_kwargs = {}
    if predict_kwargs is None:
        predict_kwargs = {}

    if genes is not None:
        log.info(f"subsetting to {len(genes)} genes")
        susp = suspension[:, genes].copy()
        spat = spatial[:, genes].copy()
    else:
        susp = suspension.copy()
        spat = spatial.copy()

    if "label_transfer" not in spatial.uns:
        spatial.uns["label_transfer"] = {}

    try:
        if kind == "logreg":
            for label in labels:
                log.info(f"training logistic regression model for {label}")
                model = _logreg_train(susp, label, **train_kwargs)

                log.info(f"predicting labels for {label}")
                ret = _logreg_predict(
                    spat, model, key_added=label, return_predict=True, **predict_kwargs
                )

                # add predicted labels to spatial dataset
                log.debug(f"results: {ret}")
                log.info(f"adding predicted labels for {label} to spatial dataset")
                spatial.obs[label] = spatial.obs_names.map(spat.obs[label].to_dict())
                spatial.obsm[f"{label}_prob"] = (
                    ret["prob"].loc[spatial.obs_names].to_numpy()
                )
                spatial.uns["label_transfer"][label] = {
                    "obsm_key": f"{label}_prob",
                    "labels": ret["prob"].columns.tolist(),
                    "model": model,
                    "genes": genes,
                    "kind": kind,
                    **train_kwargs,
                    **predict_kwargs,
                }
    except TypeError as e:
        raise TypeError(f"wrong keyword argument for kind='{kind}': {e}")

    return spatial


@ignore_warnings(category=ConvergenceWarning)
def _logreg_train(
    adata: sc.AnnData,
    groupby: str,
    use_rep: str = "raw",
    use_hvg: bool = False,
    use_pseudobulk: bool = False,
    max_pass: int = 20,
    save: Optional[os.PathLike] = None,
    model: Optional[LogisticRegression] = None,
    log: logging.Logger = logging.getLogger(__name__),
    **kwargs,
) -> LogisticRegression:
    """
    Train a logistic regression model to predict labels based on gene expression data.

    Args:
        adata (sc.AnnData): Annotated data matrix.
        groupby (str): Key in `adata.obs` containing labels.
        use_rep (str): Representation of gene expression data to use.
        use_hvg (bool): Use highly variable genes.
        use_pseudobulk (bool): Use pseudobulk data.
        max_pass (int): Maximum number of passes for model convergence.
        save (Optional[os.PathLike]): Path to save trained model.
        model (Optional[LogisticRegression]): Pre-trained model.
        log (logging.Logger): Logger object.
        **kwargs: Additional keyword arguments passed to `LogisticRegression`.

    Returns:
        LogisticRegression: Trained logistic regression model.
    """
    groupby_var = adata.obs[groupby].cat.remove_unused_categories()
    Y = groupby_var.astype(str)

    if use_rep == "raw":
        # use raw data attribute
        X = adata.raw.X
        features = adata.raw.var_names.values
    elif use_rep == "X":
        # use X data attribute
        X = adata.X
        features = adata.var_names.values
        if use_hvg and "highly_variable" in adata.var.keys():
            k_hvg = adata.var["highly_variable"].values
            X = X[:, k_hvg]
            features = features[k_hvg]
    elif use_rep in adata.obsm.keys():
        # use obsm attribute
        X = adata.obsm[use_rep]
        features = np.array([f"V{i+1}" for i in range(X.shape[1])])
    else:
        raise KeyError(f"invalid value for `use_rep`: {use_rep}")

    if use_pseudobulk:
        if sp.issparse(X):
            summarised = np.zeros((Y.unique().size, X.shape[1]))
            for i, grp in enumerate(groupby_var.cat.categories):
                k_grp = np.where(Y == grp)[0]
                summarised[i] = np.mean(X[k_grp, :], axis=0)
            X = summarised
        else:
            X = npg.aggregate(groupby_var.cat.codes, X, axis=0, func="mean")
        Y = groupby_var.cat.categories.values

    if model is not None:
        log.debug("using provided LR model")
        lr = model
    else:
        log.debug("creating new LR model")
        lr = LogisticRegression(
            penalty="l2", C=0.1, solver="saga", warm_start=True, n_jobs=-1, **kwargs
        )

    n_pass = 0
    while n_pass < max_pass:
        lr.fit(X, Y)
        n_pass += 1
        if lr.n_iter_ < 100:
            log.debug(f"converged after {n_pass} passes")
            break

    if lr.n_iter_ >= 100:
        log.error("LR model failed to converge")

    lr.features = features
    if save:
        log.info(f"saving LR model to {save}")
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(lr, save)

    return lr


def _logreg_predict(
    adata: sc.AnnData,
    model: Union[os.PathLike, ClassifierMixin],
    use_rep: str = "raw",
    use_pseudobulk: Optional[bool] = False,
    groupby: Optional[str] = None,
    feature: Optional[str] = None,
    key_added: Optional[str] = None,
    ground_truth: Optional[str] = None,
    return_predict: Optional[bool] = False,
    log: logging.Logger = logging.getLogger(__name__),
):
    """
    Predict labels using a logistic regression model.

    Args:
        adata (sc.AnnData): Annotated data matrix.
        model (Union[os.PathLike, ClassifierMixin]): Path to trained model or pre-trained model.
        use_rep (str): Representation of gene expression data to use.
        use_pseudobulk (Optional[bool]): Use pseudobulk data.
        groupby (Optional[str]): Key in `adata.obs` containing labels. Only used if `use_pseudobulk` is True.
        feature (Optional[str]): Name of feature in `adata.var` to use.
        key_added (Optional[str]): Key to add to `adata.obs`.
        ground_truth (Optional[str]): Key in `adata.obs` containing ground truth labels.
        return_predict (Optional[bool]): Return prediction results.
        log (logging.Logger): Logger object.

    Returns:
        Optional[Dict]: Prediction results.
    """
    if use_rep == "raw":
        # use raw data attribute
        X = adata.raw.X
        features = adata.raw.var_names if feature is None else adata.raw.var[feature]
    elif use_rep == "X":
        # use X data attribute
        X = adata.X
        features = adata.var_names if feature is None else adata.var[feature]
    elif use_rep in adata.obsm.keys():
        # use obsm attribute
        X = adata.obsm[use_rep]
        features = np.array([f"V{i+1}" for i in range(X.shape[1])])
    else:
        raise KeyError(f"invalid value for `use_rep`: {use_rep}")
    features = pd.Series(features)

    if isinstance(model, os.PathLike) and Path(model).is_file():
        lr = joblib.load(model)
    elif isinstance(model, ClassifierMixin):
        lr = deepcopy(model)
    else:
        raise ValueError(f"Invalid LR model: {model}")

    if getattr(lr, "features", None) is None:
        if lr.n_features_in_ == features.size:
            lr.features = features.values
        else:
            raise ValueError(
                f"logistic regression model has no feature names "
                f"and unmatched size: {lr.n_features_in_} != {features.size}"
            )

    # find features used in training
    k_x = features.isin(list(lr.features))
    log.info(f"{k_x.sum()} features used for prediction")
    k_x_idx = np.where(k_x)[0]

    # subset X to only include features used in training
    X = X[:, k_x_idx]
    features = features[k_x]

    ad_ft = (
        pd.DataFrame(features.values, columns=["ad_features"])
        .reset_index()
        .rename(columns={"index": "ad_idx"})
    )
    lr_ft = (
        pd.DataFrame(lr.features, columns=["lr_features"])
        .reset_index()
        .rename(columns={"index": "lr_idx"})
    )
    lr_idx = (
        lr_ft.merge(ad_ft, left_on="lr_features", right_on="ad_features")
        .sort_values(by="ad_idx")
        .lr_idx.values
    )

    lr.n_features_in_ = lr_idx.size
    lr.features = lr.features[lr_idx]
    lr.coef_ = lr.coef_[:, lr_idx]

    if use_pseudobulk:
        if not groupby or groupby not in adata.obs.columns:
            raise ValueError("missing or invalid `groupby`")
        groupby_var = adata.obs[groupby].cat.remove_unused_categories()
        summarised = np.zeros((groupby_var.cat.categories.size, X.shape[1]))

        for i, grp in enumerate(groupby_var.cat.categories):
            k_grp = np.where(groupby_var == grp)[0]
            if sp.issparse(X):
                summarised[i] = np.mean(X[k_grp, :], axis=0)
            else:
                summarised[i] = np.mean(X[k_grp, :], axis=0, keepdims=True)
        X = summarised

    Y_predict = lr.predict(X)
    Y_prob = lr.predict_proba(X)
    max_Y_prob = Y_prob.max(axis=1)

    if use_pseudobulk:
        tmp_groupby = adata.obs[groupby].astype(str)
        tmp_prob = np.zeros(tmp_groupby.size)
        tmp_predict = tmp_prob.astype(str)
        for i, ct in enumerate(adata.obs[groupby].cat.categories):
            tmp_prob[tmp_groupby == ct] = max_Y_prob[i]
            tmp_predict[tmp_groupby == ct] = Y_predict[i]
        max_Y_prob = tmp_prob
        Y_predict = tmp_predict

    if key_added:
        adata.obs[key_added] = Y_predict
        adata.obs[key_added] = adata.obs[key_added].astype("category")
        adata.obs[f"{key_added}_prob"] = max_Y_prob

    log.info("compiling results")
    use_index = adata.obs[groupby].cat.categories if use_pseudobulk else adata.obs_names
    results = {
        "label": Y_predict,
        "prob": pd.DataFrame(
            Y_prob,
            index=use_index,
            columns=lr.classes_,
        ),
    }

    if ground_truth:
        log.info(f"computing metrics using ground truth: {ground_truth}")
        Y_ground_truth = adata.obs[ground_truth].astype(str)
        results["accuracy"] = (Y_predict == Y_ground_truth).sum() / Y_predict.size
        results["adjusted_rand_score"] = adjusted_rand_score(Y_ground_truth, Y_predict)

    if return_predict:
        return results


def add_prob_to_obs(adata: sc.AnnData, key: str) -> sc.AnnData:
    """
    Add probabilities saved in `adata.obsm` to `adata.obs`.

    Args:
        adata (sc.AnnData): AnnData object.
        key (str): Key in `adata.uns["label_transfer"]`.
    """
    entry = adata.uns["label_transfer"][key]
    prob_df = pd.DataFrame(
        adata.obsm[entry["obsm_key"]],
        index=adata.obs_names,
        columns=entry["labels"],
    )
    adata.obs = pd.merge(
        adata.obs,
        prob_df,
        left_index=True,
        right_index=True,
    )
    return adata
