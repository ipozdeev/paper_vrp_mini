import paper_vrp_mini as research
from paper_vrp_mini.helpers import vectorize
from paper_vrp_mini import settings

hangar = research.ResearchData(
    "c:/Users/Igor/Documents/projects/option_implied_covs" +
    "/vrp_paper_hangar_mini.h5")


@vectorize
def compute_mficov(counter_currency, horizon):
    """Compute MFICov matrix from fetched variances.

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
    mfiv = hangar.get("mfiv/m" + horizon_s)

    # calculate
    mficov = research.fx_covmat_from_variances(mfiv, counter_currency)

    # sort currencies alphabetically
    mficov = mficov\
        .sort_index(axis=0, level=0, sort_remaining=True)\
        .sort_index(axis=1)

    # store as e.g. 'mficov/aud/m3m'
    mficov_key = "mficov/{}/m{}".format(counter_currency, horizon_s)
    hangar.store({mficov_key: mficov})


@vectorize
def compute_mfiv_of_currency_index(counter_currency, horizon):
    """Compute MFIV of a bunch of currency indexes."""
    # init research universe with that counter currency
    r_uni = research.ResearchUniverse(research_data=hangar,
                                      currencies=settings["sample"],
                                      counter_currency=counter_currency,
                                      s_dt=settings["s_dt"],
                                      e_dt=settings["e_dt"])

    # construct currency index
    currency_idx_strategy = r_uni.construct_strategy_currency_index()

    # compute all the quadratic forms
    mficov = r_uni.get_mficov(horizon, check_determinant=True)
    idx_mfiv = mficov.quadratic_form(currency_idx_strategy.positions,
                                     dropna=True, trim=True, reweight=True)

    # store
    idx_mfiv_key = currency_idx_strategy.get_hangar_key() + \
        "mfiv/m{}m".format(horizon)

    r_uni.store({idx_mfiv_key: idx_mfiv})


if __name__ == "__main__":
    pass
    # compute_mfiv_of_currency_index(research.settings["sample"],
    #                                [1, 2, 3, 4, 6, 9, 12])
    # import pandas as pd
    # mficov = hangar.get("mficov/usd/m3m").xs("2015-12-14", axis=0)
    # mficov_c = research.CovarianceDataFrame(mficov)
    # portfolio = pd.Series({"aud": 1 / 3, "nzd": 1 / 3, "cad": 1 / 3,
    #                        "jpy": -1 / 3, "chf": -1 / 3, "eur": -1 / 3})
    # portfolio = portfolio.reindex(mficov.columns, fill_value=0.0)
    # mficov_c.quadratic_form(portfolio, trim=True)
