import pandas as pd
import numpy as np
import re

from paper_vrp_mini.covariance import CovarianceDataFrame
from paper_vrp_mini.strategy import *


class ResearchData:
    """Storage tailored to the needs of a research project.

    Parameters
    ----------
    path_to_hangar : str
        path to the .h5 file with data

    """
    def __init__(self, path_to_hangar):
        self.path_to_hangar = path_to_hangar

        # this is too slow
        with pd.HDFStore(self.path_to_hangar, mode='r') as hangar:
            self.keys = hangar.keys()

    @staticmethod
    def format_key(key_to_format):
        """Make key name conformable with that of other keys in HDFStore."""
        if not key_to_format.startswith("/"):
            key_to_format = "/" + key_to_format
        if key_to_format.endswith("/"):
            key_to_format = key_to_format[:-1]

        return key_to_format

    def add_key(self, new_key):
        """Add a new key to the snapshot of the list of keys in the HDFStore.

        Adds a new key to the local environment, not the the store itself.
        Mostly used when saving new stuff and then having to call the keys
        again.

        Parameters
        ----------
        new_key : str

        Returns
        -------
        None

        """
        if new_key not in self.keys:
            self.keys.append(self.format_key(new_key))

        return

    def remove_key(self, bad_key):
        """Delete a key from the list of keys in the HDFStore."""
        self.keys.remove(self.format_key(bad_key))

        return

    def search_in_keys(self, regexpr=None):
        """Get all keys in the HDFStore satisfying a condition.

        Now, only single-string expression is implemented (no double
        conditions etc.)

        Parameters
        ----------
        regexpr : str
            valid pattern to search for

        Returns
        -------
        res : list
            of strings, keys matching the pattern
        """
        if regexpr is not None:
            res = [p for p in self.keys if re.search(regexpr, p) is not None]
        else:
            res = self.keys

        return res

    def get(self, what=None, get_all=True):
        """Fetch value  from the HDFStore.

        Either the key or the whole group as a multiindexed DataFrame.

        Parameters
        ----------
        what : str
        get_all : bool
            True to try down the tree from `what`; if many items found,
            e.g. 'what/subwhat_1', 'what/subwhat_2' etc., these are
            concatenated under a MultiIndex

        Returns
        -------
        res : pandas.DataFrame

        """
        what = self.format_key(what) + "/"

        # try fetch as is, if not found in hangar, search down the tree
        try:
            res = pd.read_hdf(self.path_to_hangar, what)

        except TypeError as err:
            if not get_all:
                raise err

            # get all keys that start with `what`
            valid_keys = [k for k in self.keys if k.startswith(what)]

            # of those, split the remaining name by '/', concat
            tmp_res = {
                tuple(re.sub(what, '', k).split('/')):
                self.get(k, get_all=True) for k in valid_keys
            }

            res = pd.concat(tmp_res, axis=1)

            # forget about MultiIndex if it contains only one level
            if isinstance(res.columns, pd.MultiIndex) and \
                    res.columns.nlevels < 2:
                res.columns = res.columns.levels[0]

        return res

    def remove(self, what):
        """Remove a key from the hangar."""
        with pd.HDFStore(self.path_to_hangar, mode='a') as hngr:
            hngr.remove(what)

        self.remove_key(bad_key=what)

        return

    def store(self, what):
        """Store stuff to the local hangar.

        Parameters
        ----------
        what : dict
            {key : value}
        """
        # iterate through the (key, value) pairs, store each separately
        with pd.HDFStore(self.path_to_hangar, mode='a') as hngr:
            for k, v in what.items():
                hngr.put(k, v, format="table")
                self.add_key(k)


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

        # exclude dates when the determinant was non-positiv
        if check_determinant:
            det = mficov.get_det()
            bad_dt = det.loc[det.le(0.0)].index

            mficov.df.loc[(bad_dt, slice(None)), :] *= np.nan

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
        "/oibp_hangar_mini.h5"
    hangar = ResearchData(h)

    currencies = ["aud", "cad", "chf", "eur", "gbp", "jpy", "nzd", "usd"]
    counter_currency = "usd"
    runi = ResearchUniverse(hangar, currencies, counter_currency,
                            "2006", "2018")
    mficov = runi.get_mficov(horizon=1, check_determinant=True)
