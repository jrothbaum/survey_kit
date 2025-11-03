from __future__ import annotations
import numpy as np
import random
import polars as pl
from datetime import date, datetime

from .compress import compress_df
from .. import logger


def set_seed(seed: int = 0):
    if seed > 0:
        random.seed(seed)


def get_random_state():
    return random.getstate()


def set_random_state(value):
    random.setstate(value)


def RandomNumberGenerator() -> np.random.Generator:
    return np.random.default_rng(random.randint(1, 2**63 - 1))


def generate_seed(power_of_2_limit: int = 32):
    rng = RandomNumberGenerator()
    return int(rng.integers(1, 2**power_of_2_limit - 1, 1)[0])


#   TODO - convert this to an IO plugin so I can be lazy?
class RandomData:
    """
    Generate random data, in a slightly easier way

    args:
        n_rows : int
            Number of rows in the data to be generated
        seed : int, optional
            For replicability, set the seed
            Default is 0 (which does not set the seed)

    Example
    --------
    Create a dataframe with random variables:
    >>> nRows = 10000
    >>> # Create the data
    >>> df = (RandomData(n_rows=nRows,
    ...                  seed=89465551)
    ...       .index("index")
    ...       .integer("year",
    ...                lower=2015,
    ...                upper=2020)
    ...       .float("var1", 0, 100000)
    ...       .integer("var2",1,100000)
    ...       .integer("var3",1,50)
    ...       .float("var4", 0, 1)
    ...       .date("date", date(2020,1,1), date(2025,12,31))
    ...       .datetime("datetime", date(2020,1,1), date(2025,12,31))
    ...       .np_distribution("v_normal", "normal", dict(loc=1,
    ...                                                   scale=2))
    ...       .np_distribution("v_lognormal", "lognormal", dict(mean=1,
    ...                                                         sigma=2))
    ...       .to_df()
    ...      )

    """

    def __init__(self, n_rows: int, seed: int = 0):
        if seed > 0:
            set_seed(seed)

        self.rng = RandomNumberGenerator()
        self.n_rows = n_rows
        self._data = {}

    def index(self, name: str) -> RandomData:
        """
        Create an index column (i.e. 0-n_rows-1)

        Parameters
        ----------
        name : str
            column name

        Returns
        -------
        RandomData object (so you can chain 's)
        """
        self._data[name] = range(0, self.n_rows)

        return self

    def boolean(self, name: str) -> RandomData:
        """
        Create an boolean column

        Parameters
        ----------
        name : str
            column name

        Returns
        -------
        RandomData object (so you can chain 's)
        """

        self._data[name] = np.ceil(self.rng.uniform(low=-1, high=1, size=self.n_rows))

        return self

    def integer(self, name: str, lower: int, upper: int) -> RandomData:
        """
        Create an integer column in [lower,upper]

        Parameters
        ----------
        name : str
            column name
        lower : int
            lower bound (inclusive)
        upper : int
            upper bound (inclusive)

        Returns
        -------
        RandomData object (so you can chain 's)
        """

        self._data[name] = np.ceil(
            self.rng.uniform(low=lower - 1, high=upper, size=self.n_rows)
        )

        return self

    def float(self, name: str, lower: float, upper: float) -> RandomData:
        """
        Create a float64 column in (lower,upper)

        Parameters
        ----------
        name : str
            column name
        lower : float
            lower bound
        upper : float
            upper bound

        Returns
        -------
        RandomData object (so you can chain 's)
        """

        self._data[name] = self.rng.uniform(low=lower, high=upper, size=self.n_rows)

        return self

    def date(self, name: str, start: date, end: date) -> RandomData:
        """
        Create a date column in [start,end]

        Parameters
        ----------
        name : str
            column name
        start : date
            lower bound
        end : date
            upper bound

        Returns
        -------
        RandomData object (so you can chain 's)
        """
        self._data[name] = pl.date_range(start, end, "1d", eager=True).sample(
            n=self.n_rows, with_replacement=True
        )

        return self

    def datetime(
        self, name: str, start: datetime | date, end: datetime | date
    ) -> RandomData:
        """
        Create a datetime column in [start,end]

        Parameters
        ----------
        name : str
            column name
        start : datetime | date
            lower bound
        end : datetime | date
            upper bound

        Returns
        -------
        RandomData object (so you can chain 's)
        """

        self._data[name] = pl.datetime_range(start, end, "1m", eager=True).sample(
            n=self.n_rows, with_replacement=True
        )

        return self

    def np_distribution(self, name: str, distribution: str, **kwargs) -> RandomData:
        if hasattr(self.rng, distribution):
            generator = getattr(self.rng, distribution)
            self._data[name] = generator(size=self.n_rows, **kwargs)
        else:
            message = f"Numpy generator does not have the function '{distribution}'"
            logger.error(message)
            raise Exception(message)

        return self

    def to_df(self, compress: bool = True) -> pl.DataFrame:
        df = pl.DataFrame(self._data)
        if compress:
            return compress_df(df)
        else:
            return df
