import abc
import copy


class TradingStrategy:
    """Generic strategy.

    Parameters
    ----------
    name : str
        name of the strategy, e.g. 'carry'
    rebalancing : str
        frequency of rebalancing (recalculation of signals), e.g. 'B' or 'M',
        may mimicks pandas frequencies
    positions : pandas.DataFrame
        of position weights

    """
    def __init__(self, name, rebalancing, positions, **kwargs):
        """
        """
        self.name = name.lower()
        self.rebalancing = rebalancing
        self.positions = positions.dropna(how="all").fillna(0.0)

        self.universe = sorted(self.positions.columns)

        # the rest
        for k, v in kwargs.items():
            self.__setattr__(k, v)

    @abc.abstractmethod
    def get_hangar_key(self):
        pass

    @classmethod
    def from_fxtradingstrategy(cls, name, legsize, rebalancing, fxts):
        """

        Parameters
        ----------
        name : str
        legsize : int
        rebalancing : str
        fxts : FXTradingStrategy

        Returns
        -------
        res : TradingStrategy

        """
        pos = fxts.position_weights
        unv = pos.columns
        res = cls(name, unv, legsize, rebalancing, pos)

        return res

    def shift(self, lag):
        """pandas.DataFrame.shift() for positions."""
        if self.positions is None:
            pass
        else:
            self.positions = self.positions.shift(lag)

        return self

    def upsample(self, freq, **kwargs):
        """Switch frequency of `positions` to a higher frequency.

        Fills the newly missing values backwards.

        Parameters
        ----------
        freq : str
        kwargs : dict

        Returns
        -------
        new_self : TradingStrategy

        """
        # new index, from the first to the last date, by the new frequency
        new_idx = pd.date_range(self.positions.index[0],
                                self.positions.index[-1],
                                freq=freq)

        # reindex + bfill
        new_positions = self.positions.reindex(new_idx, method="bfill",
                                               **kwargs)

        # copy, substitute positions, return
        new_self = copy.deepcopy(self)
        new_self.positions = new_positions

        return new_self

    def downsample(self, freq, **kwargs):
        """Switch frequency of `positions` to a lower frequency.

        Resamples by the last value.

        Parameters
        ----------
        freq : str
        kwargs : dict

        Returns
        -------
        new_self : TradingStrategy

        """
        # resample
        new_positions = self.positions.resample(freq, **kwargs).last()

        # copy, substitute positions, return
        new_self = copy.deepcopy(self)
        new_self.positions = new_positions

        return new_self

    def __mul__(self, other):
        """Get strategy returns given a cross-section.

        Overloads multiplication to yield portfolio returns.

        Parameters
        ----------
        other : pandas.DataFrame

        Returns
        -------
        res : pandas.DataFrame

        """
        res = self.positions.mul(other, axis=0).sum(axis=1, skipna=True)

        return res

    def __repr__(self):
        """

        Returns
        -------

        """
        res = "Trading strategy: " + self.name + "\n" + repr(self.positions)
        return res


class TimeSeriesStrategy(TradingStrategy):
    """Time series strategy."""
    def get_hangar_key(self):
        """"""
        universe_str = '_'.join(self.universe)

        k = "{}/uni_{}/reb_{}".format(self.name, universe_str,
                                      self.rebalancing)

        return k


class LongShortStrategy(TradingStrategy):
    """Representation of a long-short strategy.

    Long-short strategies have an additional parameter, namely, the number
    of assets in each leg.

    Parameters
    ----------
    name : str
        name of the strategy, e.g. 'carry'
    rebalancing : str
        frequency of rebalancing (recalculation of signals), e.g. 'B' or 'M',
        may mimicks pandas frequencies
    positions : pandas.DataFrame
        of position weights
    legsize : int
        number of assert in the long and short portfolio

    """
    def __init__(self, name, rebalancing, positions, legsize):
        """
        """
        super().__init__(name, rebalancing, positions, legsize=legsize)

    @abc.abstractmethod
    def get_hangar_key(self):
        """Get the key used to fetch related stuff from the HDFStore."""
        universe_str = '_'.join(self.universe)

        k = "{}/uni_{}/legsize_{}/reb_{}".format(self.name, universe_str,
                                                 self.legsize,
                                                 self.rebalancing)

        return k


if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    w = pd.DataFrame(np.random.random(size=(10, 3)),
                     columns=["aud", "chf", "nzd"],
                     index=pd.date_range("2001-01-31", periods=10, freq='M'))
    strat = LongShortStrategy(name="pseudo_carry", rebalancing='M',
                              positions=w, legsize=1)
    print(strat)
    strat.get_hangar_key()