import pandas as pd

from helpers import fx_covmat_from_variances
from core import ResearchData

path_to_hangar = "c:/Users/Igor/Documents/projects/" + \
    "option_implied_covs/oibp_hangar_mini.h5"
hangar = ResearchData(path_to_hangar)


def compute_mficov(counter_currency, horizon=1):
    """Compute MFICov matrix from fetched varainces.

    MFIV must be stored

    Parameters
    ----------
    counter_currency : str or list-like
        3-letter ISO of the counter currency
    horizon : int or list-like
        horizon, in months

    Returns
    -------
    None

    """
    if isinstance(counter_currency, (list, tuple)):
        for c in counter_currency:
            compute_mficov(c, horizon)

    if isinstance(horizon, (list, tuple)):
        for h in horizon:
            compute_mficov(counter_currency, h)

    horizon_s = str(horizon) + "m"

    # fetch MFIV
    mfiv = hangar.get("mfiv/m" + horizon_s)

    # calculate
    mficov = fx_covmat_from_variances(mfiv, counter_currency)

    # sort currencies alphabetically
    mficov = mficov\
        .sort_index(axis=0, level=0, sort_remaining=True)\
        .sort_index(axis=1)

    # store as e.g. 'mficov/aud/m3m'
    mficov_key = "mficov/{}/m{}".format(counter_currency, horizon_s)
    hangar.store({mficov_key: mficov})


if __name__ == "__main__":
    compute_mficov(["usd", "aud"], [1, 3])
