import numpy as np
import pandas as pd


def normalize_weights(wght):
    """Recompute weights in a vector of weights to make them valid weights.

    Parameters
    ----------
    wght : pandas.DataFrame or pandas.Series
        if DataFrame, it is assumed weights are in rows

    Returns
    -------
    res : pandas.DataFrame or pandas.Series

    """
    # if a DataFrame -> apply to each row
    if isinstance(wght, pd.DataFrame):
        return wght.apply(normalize_weights, axis=1)

    # if all are zero, return
    if (wght == 0).all():
        return wght

    elif (wght < 0).any() & (wght > 0).any():
        # if there are both positive and negative weights, it is a zero-cost
        #   position, so the short and long leg needs to sum up to 1 in
        #   magnitude separately
        short_leg = wght.where(wght < 0)
        long_leg = wght.where(wght >= 0)

        short_leg = short_leg / np.abs(short_leg).sum()
        long_leg = long_leg / np.abs(long_leg).sum()

        res = short_leg.fillna(long_leg)

    else:
        # else the usual stuff
        res = wght / np.abs(wght).sum()

    return res


def trim_symmetric_df(covmat):
    """Get rid of rows/columns with NA in a symmetric matrix.

    Remove rows and columns simulataneously thus preserving the symmetry
    until no NAs are present. If all values are NA, empty DataFrame is
    returned.

    Parameters
    ----------
    covmat : pandas.DataFrame
        symmetric matrix

    Returns
    -------
    covmat_trim : pandas.DataFrame
        with certain rows kicked out

    """
    covmat_trim = covmat.copy()

    # init count of nans
    nan_count_total = pd.isnull(covmat_trim).sum().sum()

    # while there are nans in covmat, remove columns with max no. of nans
    while nan_count_total > 0:
        # detect rows where number of nans is less than maximum
        nan_number = pd.concat({
            1: pd.isnull(covmat_trim).sum(axis=0),
            0: pd.isnull(covmat_trim).sum(axis=1)
        }, axis=1)
        nan_max_axis = nan_number.max().idxmax()
        nan_max_loc = nan_number[nan_max_axis].idxmax()

        nan_max_idx = max([(p, q) for q, p in enumerate(nan_number)])[1]

        covmat_trim = covmat_trim\
            .drop(nan_max_loc, axis=0)\
            .drop(nan_max_loc, axis=1)

        # new count of nans
        nan_count_total = pd.isnull(covmat_trim).sum().sum()

    # make sure index is not messed up
    covmat_trim = covmat_trim.loc[:, covmat_trim.index]

    return covmat_trim


def quadratic_form(covmat, weights, dropna=False, trim=False, reweight=False):
    """Compute quadratic form from symmetric matrix and vector of weights.

    Parameters
    ----------
    covmat : pandas.DataFrame
        symmetric DataFrame, needs to be indexed and columned equally
    weights : pandas.Series
    dropna : bool
        True to drop na in `weight` covmat will be reindexed
        accordingly
    trim : bool
        True to trim a covmat whenever there are NA, which would results in
        not all assets entering the calculation, but the covmat being not
        discarded altogether
    reweight : bool
        True to reweight `weights` (e.g. after trimming `covmat`)

    Returns
    -------
    res : float

    """
    assert covmat.index.equals(covmat.columns)

    if covmat.empty:
        return np.nan

    # those values where weight is 0 do not contribute to variance
    no_use_stuff = weights == 0.0

    new_w = weights.loc[~no_use_stuff]
    new_vcv = covmat.loc[~no_use_stuff, ~no_use_stuff]

    if dropna:
        new_w = new_w.dropna()
        new_vcv = new_vcv.loc[new_w.index, new_w.index]

    # trim if necessary (consider reweighting after this!)
    if trim:
        new_vcv = trim_symmetric_df(new_vcv)

        # reweight now
        new_w = new_w.reindex(index=new_vcv.columns)

    # skip this matrix if all na
    if new_w.empty | new_vcv.empty:
        return np.nan

    if reweight:
        # normalize weights (long-short taken care of)
        new_w = normalize_weights(new_w)

    # calculate the quadratic form
    res = new_w.dot(new_vcv.dot(new_w))

    return res
