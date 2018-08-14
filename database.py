import pandas as pd
import re


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
