import numpy as np
import pandas as pd
import itertools


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


def fx_covmat_from_variances(variances, counter_currency):
    """Compute covariance matrix of FX log-returns by no-arbitrage argument.

    Given variances of log-changes of all possible cross-exchange rates
    between N currencies, computes the (N-1)x(N-1) covariance matrix of
    log-returns of N-1 currencies against a common `counter_currency`.

    Parameters
    ----------
    variances : pandas.Series or pandas.DataFrame
        of variances of appreciation rates, in (frac of 1), columned with a
        MultiIndex of (base, counter) currencies
    counter_currency : str
        3-letter iso e.g. 'usd', denoting the counter currency of returns
        that the covariance matrix is calcualted for

    Returns
    -------
    res : pandas.DataFrame
        symmetric covariance dataframe if `variances` is a Series,
        or stacked such dataframes multiindexed by (date, currency) if
        `variances` is a DataFrame

    """
    assert isinstance(variances, (pd.Series, pd.DataFrame))

    # if `variances` is a DataFrame, loop over rows
    if isinstance(variances, pd.DataFrame):
        # allocate space
        covmat_dict = dict()

        # loop over time
        for t, row in variances.iterrows():
            covmat_dict[t] = fx_covmat_from_variances(row, counter_currency)

        # concat to a DataFrame with MultiIndex
        res = pd.concat(covmat_dict, axis=0, names=["date", "currency"])\
            .sort_index(axis=0, level=0, sort_remaining=False)

        return res

    # `variances` is a Series from here -------------------------------------
    if not isinstance(variances.index, pd.MultiIndex):
        raise ValueError("Index of `variances` must be of two levels: base "
                         "and counter currency.")

    # collect all possible currencies (unique, obviously)
    currencies = list(set(
        variances.index.levels[0].tolist() +
        variances.index.levels[1].tolist()))

    # kick out the counter currency
    currencies = [c for c in currencies if c != counter_currency]

    # init storage for resulting covariance matrix
    res = pd.DataFrame(np.nan, index=currencies, columns=currencies)

    # for the sake of code brevity
    c = counter_currency

    # loop over currencies
    for a in currencies:
        # variance of returns of that currency vs. the coutner currency
        var_ac = variances.get((a, c), variances.get((c, a), np.nan))

        # loop over the other currencies
        for b in currencies:
            if b == a:
                # if its the same as xxx, just save the above variance
                res.loc[a, a] = var_ac
                continue

            # find "xxxyyy" or "yyyxxx"
            var_ab = variances.get((a, b), variances.get((b, a), np.nan))

            # find "yyyusd" or "usdyyy"
            var_cb = variances.get((c, b), variances.get((b, c), np.nan))

            # calculate covariance
            cov_ac_bc = var_triplet_to_cov(var_ab=var_ab,
                                           var_ac=var_ac,
                                           var_cb=var_cb)

            # store symmetric covariances
            res.loc[a, b] = cov_ac_bc
            res.loc[b, a] = cov_ac_bc

    return res


def var_triplet_to_cov(var_ab, var_ac, var_cb):
    """Calculate covariance and correlation implied by three variances.

    Builds on var[AB] = var[AC] + var[CC] + 2*cov[AC,CB] if
    AB = AC + CB to extract covariance and correlation between AC and BC (
    note the change of order).

    Parameters
    ----------
    var_ab : float
        variance of currency pair consisting of base currency A and counter
        currency B (units of B for one unit of A)
    var_ac : float
        variance of currency pair consisting of base currency A and
        the common counter currency C (units of C for one unit of A)
    var_cb : float
        variance of currency pair consisting of base currency B and the
        common counter currency C (units of B for one unit of C)

    Returns
    -------
    res : float
        implied covariance between AC and BC
    """

    res = -0.5 * (var_ab - var_ac - var_cb)

    return res


def vectorize(func):
    """Unpack lists in arguments, parsing the content value-by-value."""
    def wrapper(*args, **kwargs):
        # transform each argument to list for easier powersetting
        args_l = ([a] if not isinstance(a, (list, tuple)) else a for a in args)

        # powerset
        new_args = list(itertools.product(*args_l))

        # transform each keyword argument to list for easier powersetting
        kwargs_l = dict()
        for k, v in kwargs.items():
            kwargs_l[k] = v if isinstance(v, (list, tuple)) else [v]

        # powerset
        new_kwargs = [
            {kk: vv for kk, vv in zip(kwargs_l.keys(), v)}
            for v in itertools.product(*kwargs_l.values())
        ]

        # all together
        new_args_kwargs = itertools.product(new_args, new_kwargs)

        # evaluate function
        for arg, kwarg in new_args_kwargs:
            func(*arg, **kwarg)

    return wrapper


def maturity_str_to_float(mat, to_freq='Y'):
    """Convert maturity to fractions of a period.

    Parameters
    ----------
    mat: str
    to_freq : str
        character, pandas frequency

    Returns
    -------
    res : float

    """
    if (to_freq.upper() == 'B') and (mat[-1].upper() == 'M'):
        map_dict = {"1m": 22, "2m": 44, "3m": 64, "4m": 84, "5m": 104,
                    "6m": 126, "7m": 148, "8m": 170, "9m": 190, "10m": 212,
                    "11m": 232, "12m": 254}
        return map_dict[mat]

    scale_matrix = pd.DataFrame(
        index=['D', 'B', 'Y'],
        columns=['W', 'M', 'Y'],
        data=np.array([[1/7, 1/30, 1/365],
                       [1/5, 1/22, 1/254],
                       [52, 12, 1]], dtype=float))

    int_part = int(mat[:-1])
    scale = scale_matrix.loc[to_freq, mat[-1].upper()]

    res = int_part / scale

    return res
