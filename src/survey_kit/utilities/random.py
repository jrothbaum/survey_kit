from __future__ import annotations
import numpy as np
import random
import polars as pl
from datetime import date, datetime

from .compress import compress_df
from .. import logger


def set_seed(seed: int = 0):
    """
    Set the random seed for reproducibility.

    Sets the seed for Python's random module, which is used by the
    random number generator in this module.

    Parameters
    ----------
    seed : int, optional
        Random seed. If 0 or less, does not set seed. Default is 0.

    Examples
    --------
    >>> from survey_kit.utilities.random import set_seed, generate_seed
    >>> set_seed(12345)
    >>> seed1 = generate_seed()
    >>> set_seed(12345)
    >>> seed2 = generate_seed()
    >>> assert seed1 == seed2  # Same seed produces same sequence
    """

    if seed > 0:
        random.seed(seed)


def get_random_state():
    """
    Get the current state of the random number generator.

    Returns
    -------
    tuple
        Internal state of Python's random module.

    Examples
    --------
    >>> from survey_kit.utilities.random import get_random_state, set_random_state
    >>> state = get_random_state()
    >>> # ... do some random operations ...
    >>> set_random_state(state)  # Restore to previous state
    """
    return random.getstate()


def set_random_state(value):
    """
    Set the random number generator to a specific state.

    Parameters
    ----------
    value : tuple
        State tuple from get_random_state().

    Examples
    --------
    >>> state = get_random_state()
    >>> # ... do some random operations ...
    >>> set_random_state(state)  # Restore to previous state
    """
    random.setstate(value)


def RandomNumberGenerator() -> np.random.Generator:
    """
    Create a new numpy random number generator with random seed.

    Returns
    -------
    np.random.Generator
        Numpy random number generator initialized with a random seed
        from Python's random module.

    Examples
    --------
    >>> from survey_kit.utilities.random import RandomNumberGenerator
    >>> rng = RandomNumberGenerator()
    >>> random_values = rng.normal(loc=0, scale=1, size=100)
    """
    return np.random.default_rng(random.randint(1, 2**63 - 1))


def generate_seed(power_of_2_limit: int = 32):
    """
    Generate a random seed value.

    Parameters
    ----------
    power_of_2_limit : int, optional
        Upper bound is 2^power_of_2_limit. Default is 32.

    Returns
    -------
    int
        Random integer between 1 and 2^power_of_2_limit - 1.

    Examples
    --------
    >>> from survey_kit.utilities.random import generate_seed, set_seed
    >>> seed = generate_seed()
    >>> set_seed(seed)  # Use for reproducibility
    """
    rng = RandomNumberGenerator()
    return int(rng.integers(1, 2**power_of_2_limit - 1, 1)[0])


#   TODO - convert this to an IO plugin so I can be lazy?
class RandomData:
    """
    Generate random data, in a slightly easier way.

    Parameters
    ----------
    n_rows : int
        Number of rows in the data to be generated
    seed : int, optional
        For replicability, set the seed.
        Default is 0 (which does not set the seed)

    Examples
    --------
    Create a dataframe with random variables:

    >>> nRows = 10000
    >>> df = (RandomData(n_rows=nRows, seed=89465551)
    ...       .index("index")
    ...       .integer("year", lower=2015, upper=2020)
    ...       .float("var1", 0, 100000)
    ...       .integer("var2", 1, 100000)
    ...       .integer("var3", 1, 50)
    ...       .float("var4", 0, 1)
    ...       .date("date", date(2020, 1, 1), date(2025, 12, 31))
    ...       .datetime("datetime", date(2020, 1, 1), date(2025, 12, 31))
    ...       .np_distribution("v_normal", "normal", dict(loc=1, scale=2))
    ...       .np_distribution("v_lognormal", "lognormal", dict(mean=1, sigma=2))
    ...       .to_df()
    ... )
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
        """
        Create a column from a numpy distribution.

        Parameters
        ----------
        name : str
            Column name.
        distribution : str
            Name of numpy random distribution (e.g., "normal", "lognormal", "exponential").
            Must be a valid method of np.random.Generator.
        **kwargs
            Keyword arguments passed to the distribution function (e.g., loc, scale for normal).

        Returns
        -------
        RandomData
            Self for method chaining.

        Raises
        ------
        Exception
            If distribution is not a valid numpy random distribution.

        Examples
        --------
        >>> df = (RandomData(n_rows=1000, seed=123)
        ...       .np_distribution("normal_var", "normal", loc=0, scale=1)
        ...       .np_distribution("lognormal_var", "lognormal", mean=1, sigma=2)
        ...       .np_distribution("exponential_var", "exponential", scale=1.5)
        ...       .to_df())
        """

        if hasattr(self.rng, distribution):
            generator = getattr(self.rng, distribution)
            self._data[name] = generator(size=self.n_rows, **kwargs)
        else:
            message = f"Numpy generator does not have the function '{distribution}'"
            logger.error(message)
            raise Exception(message)

        return self

    def to_df(self, compress: bool = True) -> pl.DataFrame:
        """
        Convert accumulated random data to a Polars DataFrame.

        Parameters
        ----------
        compress : bool, optional
            Apply compression to reduce memory usage by downcasting numeric types.
            Default is True.

        Returns
        -------
        pl.DataFrame
            DataFrame containing all generated columns.

        Examples
        --------
        >>> df = (RandomData(n_rows=1000, seed=123)
        ...       .integer("id", 1, 1000)
        ...       .float("value", 0, 100)
        ...       .to_df())
        >>> print(df)

        >>> # Without compression
        >>> df_uncompressed = (RandomData(n_rows=1000, seed=123)
        ...                    .integer("id", 1, 1000)
        ...                    .to_df(compress=False))
        """

        df = pl.DataFrame(self._data)
        if compress:
            return compress_df(df)
        else:
            return df
