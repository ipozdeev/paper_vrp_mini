import pandas as pd
import numpy as np

from paper_vrp_mini.covariance import CovarianceDataFrame
from paper_vrp_mini.strategy import *
from paper_vrp_mini.database import ResearchData


class ResearchUniverse:
    """Research universe representation.

    For a particular sample of currencies and a particular counter currency.

    Parameters
    ----------
    research_data : ResearchData
        valid data hangar
    currencies : list-like
        of 3-letter ISO, e.g. ['aud', 'chf']
    counter_currency : str
        3-letter ISO, e.g. 'aud'
    s_dt : str
        date to start the sample period at
    e_dt : str
        date to end the sample period at

    """
    def __init__(self, research_data, currencies, counter_currency,
                 s_dt, e_dt):
        # args
        self.research_data = research_data
        self.counter_currency = counter_currency

        self.currencies = sorted(
            [c.lower() for c in currencies if c != counter_currency])

        self.s_dt = s_dt
        self.e_dt = e_dt

        # self.universe = "uni_" + '_'.join(self.currencies)

        # cache
        self.cache = dict()

    def get(self, *args, **kwargs):
        """Get data from hangar (pass to ResearchData.get)."""
        return self.research_data.get(*args, **kwargs)

    def store(self, *args, **kwargs):
        """Store data to hangar (pass to ResearchData.get)."""
        return self.research_data.store(*args, **kwargs)

    def get_mfiv(self, horizon):
        """Fetch MFIV from the HDFStore, retaining only certain currencies.

        Parameters
        ----------
        horizon : int
            horizon, in months

        Returns
        -------

        """
        horizon_s = "{:d}m".format(horizon)

        # mfiv
        mfiv_key = "mfiv/m" + horizon_s

        mfiv_full_one = self.get(mfiv_key)
        mfiv_full_two = mfiv_full_one.swaplevel(axis=1)
        mfiv_full_two.columns.names = ["base", "counter"]

        mfiv_full = pd.concat((mfiv_full_one, mfiv_full_two), axis=1)

        mfiv = mfiv_full.reindex(
            columns=pd.MultiIndex.from_product(
                [self.currencies, [self.counter_currency]],
                names=["base", "counter"]
            )
        )

        return mfiv

    def get_mficov(self, horizon, check_determinant=False):
        """Fetch time series of model-free implied covariance matrices.

        etch from HDFStore + reindex with `self.currencies` + (optional) check
        if the determinant is non-negative everywhere

        Parameters
        ----------
        horizon : int or str
        check_determinant : bool
            True to exclude those dates when the determinant was non-positive

        Returns
        -------

        """
        if isinstance(horizon, int):
            horizon_s = "{:d}m".format(horizon)
        else:
            horizon_s = horizon

        cached = self.cache.get("mficov_{}".format(horizon_s), None)
        if cached is not None:
            return cached

        # fetch from HDFStore
        mficov_key = "mficov/{}/m{}".format(self.counter_currency, horizon_s)
        mficov_df = self.get(mficov_key)

        # reindex
        mficov = CovarianceDataFrame(mficov_df).reindex(assets=self.currencies)

        # exclude dates when the determinant was non-positive
        if check_determinant:
            det = mficov.get_det()
            bad_dt = det.loc[det.le(0.0)].index

            mficov.df.loc[(bad_dt, slice(None)), :] *= np.nan

        # to cache
        self.cache["mficov_{}".format(horizon_s)] = mficov

        return mficov

    def construct_strategy_currency_index(self, rebalancing='B'):
        """Construct equally-weighted portfolio of currencies in the universe.

        Takes currencies in `self.universe` that are not the
        `counter_currency` and constructs a static equally-weighted portfolio
        thereof. Result is an instance of TimeSeriesStrategy with
        `positions` being a Series with the same equal value for each currency.

        Parameters
        ----------
        rebalancing : str
            rebalancing frequency of the strategy (for naming mostly)

        Returns
        -------
        res : TimeSeriesStrategy
            with pandas.Series of 1/len(universe) for positions

        """
        res = TimeSeriesStrategy.currency_index(
            currency=self.counter_currency,
            universe=self.currencies,
            rebalancing=rebalancing)

        return res


if __name__ == "__main__":
    h = "c:/Users/Igor/Documents/projects/option_implied_covs" + \
        "/vrp_paper_hangar_mini.h5"
    hangar = ResearchData(h)

    currencies = ["aud", "cad", "chf", "eur", "gbp", "jpy", "nzd", "usd"]
    counter_currency = "usd"
    runi = ResearchUniverse(hangar, currencies, counter_currency,
                            "2006", "2018")
    mficov = runi.get_mficov(horizon=1, check_determinant=True)
