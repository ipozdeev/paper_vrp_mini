import pandas as pd
import numpy as np
from helpers import quadratic_form, trim_symmetric_df


class CovarianceDataFrame:
    """Time series of of covariance matrices.

    Representation of time series of symmetric covariance matrices, based on
    `pandas.DataFrame`. Time is in the 0th level of the index.

    Parameters
    ----------
    df : pandas.DataFrame
        with pandas.MultiIndex (date, asset)

    """
    def __init__(self, df):
        """
        """
        assert isinstance(df.index, pd.MultiIndex)

        df.index.names = ["date", "asset"]
        df.columns.name = "asset"

        # sort
        df = df.sort_index(axis=0, sort_remaining=False)

        # unique dates
        dt_idx = df.index.get_level_values("date").unique()

        self.df = df
        self.date_index = dt_idx
        self.assets = df.columns

        # legacy of `self.df`
        self.index = self.df.index
        self.columns = self.df.columns

    def __getattr__(self, item):
        # Redefine to be able to get attributes from the underlying DataFrame
        res = getattr(self.df, item)

        if callable(res):
            res = to_covariancedataframe(res)

        return res

    def get_variances(self):
        """Get the time series of diagonal elements (variances).

        Returns
        -------
        res : pandas.DataFrame
            of variances, indexed with level 'date' of the original df

        """
        def aux_diag_getter(x):
            aux_res = pd.Series(np.diag(x), index=x.columns)
            return aux_res

        res = self.groupby_time().apply(aux_diag_getter)

        return res

    def __repr__(self):
        repr_str = "DataFrame of covariance matrices:\n" + repr(self.df)
        return repr_str

    def reindex(self, date_index=None, assets=None, **kwargs):
        """Reindex dataframe.

        Parameters
        ----------
        date_index : list-like
            (optional)
        assets : list-like
            (optional)
        kwargs : any
            arguments to `pandas.reindex`

        Returns
        -------
        res : CovarianceDataFrame

        """
        df_reix = self.df.copy()

        if assets is not None:
            df_reix = df_reix\
                .reindex(assets, axis=0, level="asset", **kwargs)\
                .reindex(columns=assets, **kwargs)

        if date_index is not None:
            date_index = pd.MultiIndex.from_product(
                iterables=[date_index, self.assets],
                names=["date", "asset"])

            df_reix = df_reix.reindex(index=date_index, **kwargs)

        res = CovarianceDataFrame(df_reix)

        return res

    def groupby_time(self):
        """Group by the time index.

        Implements an analogue of `pandas.DataFrame.iterrows()` for looping
        over date rows only; wrapper around `pandas.groupby()`.

        Watch out that in pandas.groupby(level=0), MultiIndex is not dropped
        for individual grous!

        Returns
        -------
        res : pandas.core.groupby.DataFrameGroupBy

        """
        res = self.df.groupby(axis=0, level="date")

        return res

    def dropna(self, how="all"):
        """Drop (date, matrix) pairs where the matrix only contains NA.

        Parameters
        ----------
        how : str

        Returns
        -------
        res : CovarianceDataFrame

        """
        # function to get NA-only matrices
        def na_func(x):
            if how == "all":
                all_na_flag = x.isnull().all().all()
            else:
                all_na_flag = x.isnull().any().any()
            return all_na_flag

        # find NA-only matrices
        na_bool_idx = self.groupby_time().apply(na_func)

        # broadcast to be able to .loc on the MultiIndex
        na_bool_idx = na_bool_idx.reindex(index=self.df.index, level="date")

        # .loc
        res_df = self.df.loc[~na_bool_idx]
        res = CovarianceDataFrame(res_df)

        return res

    def group_apply(self, func, level="asset"):
        """Apply function to the dataframe grouped by along the index axis.

        Applies to 'slices' of the dataframe, e.g. when resampling. Underlies
        most other convenience functions.

        Parameters
        ----------
        func : callable
        level : str
            'date' to apply stuff such as drop missing obs, or
            'asset' to apply stuff such as rolling/expanding

        Returns
        -------
        res : CovarianceDataFrame

        """
        res = dict()

        for n, grp in self.df.groupby(axis=0, level=level):
            res[n] = func(grp.xs(n, axis=0, level=level, drop_level=True))

        res_df = pd.concat(res, axis=0)

        # restores date, asset order
        if level == "asset":
            res_df = res_df.swaplevel(axis=0)

        # back to
        res = CovarianceDataFrame(res_df)

        return res

    def rolling_apply(self, func2d, **kwargs):
        """Rolling trasformation along the time dimension.

        Whenever a function needs to be applied to rolling windows of the
        same asset, `rolling_apply` provides functionality to do so.
        Examples include smoothing covariances of asset_1 with all other
        assets over time.

        Parameters
        ----------
        func2d : callable
        kwargs : dict
            arguments to pandas.rolling

        Returns
        -------
        res : CovarianceDataFrame

        """
        # auxiliary function to evoke
        def aux_func2d(x):
            aux_res = x.rolling(**kwargs).apply(func2d)
            return aux_res

        # loop over assets, do the rolling magic
        res = self.group_apply(aux_func2d, level="asset")

        return res

    def resample(self, func2d, **kwargs):
        """Convenient implementation of `pandas.DataFrame.resample()`.

        Use rule='M' in `kwargs` to resample monthly.

        Parameters
        ----------
        func2d : callable or str
            if str, can only be 'last'
        kwargs : any

        Returns
        -------
        res : CovarianceDataFrame

        """
        # auxiliary function
        def aux_func2d(x):
            aux_res = x.resample(**kwargs, axis=0, level="date")
            if func2d == "last":
                aux_res = aux_res.last()
            else:
                aux_res = aux_res.apply(func2d)
            return aux_res

        res = self.group_apply(aux_func2d, level="asset")

        return res

    def quadratic_form(self, other, **kwargs):
        """Compute time series of quadratic forms.

        Parameters
        ----------
        other : pandas.Series or pandas.DataFrame
            of the 'bread' of the sandwich; if Series, it will be used for all
            dates, if DataFrame, dates will be matched (index of `other` will
            be the index of the result)
        kwargs : dict
            keyword arguments to `quadratic_form()`, such as 'trim'

        Returns
        -------
        res : pandas.Series
            of quadratic forms

        """
        if isinstance(other, pd.Series):
            other = pd.concat({t: other for t in self.date_index}, axis=1).T

        # reindex, get only the pandas.DataFrame for the sake of speed
        vcv = self.reindex(date_index=other.index, assets=other.columns).df

        # do the dot product
        res = pd.Series(index=other.index)

        for t, row in other.iterrows():
            res.loc[t] = quadratic_form(vcv.loc[t], row, **kwargs)

        return res

    def get_det(self, trim=False):
        """Calculate determinant for each date.

        Returns
        -------
        res : pandas.Series

        """
        # auxiliary function to drop date level (otherwise errors) and use
        # with `self.groupby_time()`
        def det_fun(df):
            df_tmp = df.copy()
            df_tmp.index = df_tmp.index.droplevel(0)

            if trim:
                df_tmp = trim_symmetric_df(df_tmp)

            if df_tmp.empty:
                aux_res = np.nan
            else:
                aux_res = np.linalg.det(df_tmp.values)

            return aux_res

        res = self.groupby_time().apply(det_fun)

        return res


def to_covariancedataframe(func):
    """Transform generic output of `func` to CovarianceDataFrame.

    Returns
    -------
    wrapper : callable
    """
    def wrapper(*args, **kwargs):
        res = func(*args, **kwargs)
        return CovarianceDataFrame(res)

    return wrapper


if __name__ == "__main__":
    df = pd.concat({t: pd.DataFrame(np.eye(4))*t.day
                    for t in pd.date_range("2001-01-01", periods=10)},
                   axis=0)

    df.iloc[5, 3] = np.nan
    cv_df = CovarianceDataFrame(df)

    # w = pd.Series(np.ones(shape=(4,)) / 4)
    # cv_df.quadratic_form(other=w)

    cv_df.get_det(trim=True)
