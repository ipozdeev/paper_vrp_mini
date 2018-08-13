import paper_vrp_mini as research
from paper_vrp_mini.helpers import vectorize
from paper_vrp_mini import settings

r_data = research.ResearchData(path_to_hangar=settings["path_to_hangar"])


@vectorize
def compute_mficov(counter_currency, horizon):
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
    horizon_s = str(horizon) + "m"

    # fetch MFIV
    mfiv = r_data.get("mfiv/m" + horizon_s)

    # calculate
    mficov = research.fx_covmat_from_variances(mfiv, counter_currency)

    # sort currencies alphabetically
    mficov = mficov\
        .sort_index(axis=0, level=0, sort_remaining=True)\
        .sort_index(axis=1)

    # store as e.g. 'mficov/aud/m3m'
    mficov_key = "mficov/{}/m{}".format(counter_currency, horizon_s)
    r_data.store({mficov_key: mficov})


@vectorize
def compute_mfiv_of_currency_index(counter_currency, horizon):
    """Compute MFIV of a bunch of currency indexes."""
    # init research universe with that counter currency
    r_uni = research.ResearchUniverse(research_data=r_data,
                                      currencies=settings["sample"],
                                      counter_currency=counter_currency,
                                      s_dt=settings["s_dt"],
                                      e_dt=settings["e_dt"])

    currency_idx_strategy = r_uni.construct_strategy_currency_index()

    # compute all the quadratic forms
    mficov = r_uni.get_mficov(horizon, check_determinant=True)
    idx_mfiv = mficov.quadratic_form(currency_idx_strategy.positions,
                                     dropna=True, trim=True, reweight=True)


if __name__ == "__main__":
    compute_mfiv_of_currency_index(["usd", "aud"], [1, 3])
