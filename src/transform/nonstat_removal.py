"""This module provides transformations that aim at removing
nonstationarity from series retaining the information, so that
the analysis of the series is more efficient
"""

from typing import Callable
import numpy as np
import pandas as pd
from sktime.transformations.series.difference import Differencer
from sktime.transformations.series.boxcox import BoxCoxTransformer
from wotan import flatten

from base import TimeSeriesWithAnoms, MapTransformation
from utils import check_anom_df_consistency, adjust_invalid_anoms


class RollingMeanSmoother(MapTransformation):
    """Applies rolling mean on values of a time series

    x_out[i] = mean(x_in[i - window_len + 1], x_in[i - window_len + 2], ..., x_in[i])

    The first (window_len - 1) values are undefined (there are cut off)
    """

    def __init__(self, window_len: int):
        super().__init__(name=f"rollingmean_{window_len}")
        self.window_len: int = window_len

    def _transform_series(self, ts: TimeSeriesWithAnoms) -> TimeSeriesWithAnoms:

        # get rolling means with pandas
        rolling_mean_series = (
            pd.Series(ts.values).rolling(window=self.window_len).mean()
        )

        # first (self.window_len - 1) values are NaNs
        values_transformed = rolling_mean_series.values[(self.window_len - 1) :]

        # get new anoms df - shift indices
        anoms_transformed = ts.anoms - (self.window_len - 1)

        # adjust anoms that were in the cut off part
        anoms_transformed = adjust_invalid_anoms(anoms_transformed, ts.name)
        if anoms_transformed.shape[0] == 0:
            raise RuntimeError("No anomalies has left after removing invalid ones")

        # check if new anoms indices are consistent
        if not check_anom_df_consistency(anoms_transformed):
            raise RuntimeError(
                "Tranformation produced inconsistent anomalies indices; check if there are no "
                "anomalies in the part that is cut off in the transformation"
            )

        return TimeSeriesWithAnoms(
            values=values_transformed, anoms=anoms_transformed, name=ts.name
        )


class Differencing(MapTransformation):
    """Performs differencing on series values

    x_out[i] = x_in[i] - x_in[i - lag]

    The first `lag` values are undefined and are cut off
    (the result is shorter)

    Attributes:
        lag: if int: the lag value to difference the data,
            if callable: function taking TimeSeriesWithAnoms object as argument
            and returning the lag
    """

    def __init__(self, lag: int | Callable[[TimeSeriesWithAnoms], int]):
        super().__init__(
            name=f"differencing_{(lag if isinstance(lag, int) else 'dynamic')}"
        )
        self.lag: int | Callable[[TimeSeriesWithAnoms], int] = lag

    def _transform_series(self, ts: TimeSeriesWithAnoms) -> TimeSeriesWithAnoms:

        lag = self.lag if isinstance(self.lag, int) else self.lag(ts)
        diff = Differencer(lags=lag, na_handling="drop_na")

        # transform values (differencing)
        values_transformed = diff.fit_transform(ts.values).reshape((-1,))

        # transform anom inds (shift by lag)
        anoms_transformed = ts.anoms - lag

        # adjust anoms that were in the cut off part
        anoms_transformed = adjust_invalid_anoms(anoms_transformed, ts.name)
        if anoms_transformed.shape[0] == 0:
            raise RuntimeError("No anomalies has left after removing invalid ones")

        # check if new anoms indices are consistent
        if not check_anom_df_consistency(anoms_transformed):
            raise RuntimeError(
                "Tranformation produced inconsistent anomalies indices; check if there are no "
                "anomalies in the part that is cut off in the transformation"
            )

        return TimeSeriesWithAnoms(
            values=values_transformed, anoms=anoms_transformed, name=ts.name
        )


class RobustDifferencing(MapTransformation):
    """Performs robust differencing:

    x_out[i] = (
        median({x_in[i - j*lag], j=1,2,..,k}) if i >= k*lag
        else x_in[i - lag]
    )

    Attributes:
        lag: if int: the lag value to difference the data,
            if callable: function taking TimeSeriesWithAnoms object as argument
            and returning the lag

    """

    def __init__(self, lag: int | Callable[[TimeSeriesWithAnoms], int], k: int):
        super().__init__(
            name=f"robustdifferencing_{(lag if isinstance(lag, int) else 'dynamic')}_{k}"
        )
        self.lag: int | Callable[[TimeSeriesWithAnoms], int] = lag
        self.k = k

    def _transform_series(self, ts: TimeSeriesWithAnoms) -> TimeSeriesWithAnoms:
        lag = self.lag if isinstance(self.lag, int) else self.lag(ts)

        # transform values
        values_transformed = self.__robust_diff(ts.values, lag)

        # transform anoms (shift by lag)
        anoms_transformed = ts.anoms - lag

        # adjust anoms that were in the cut off part
        anoms_transformed = adjust_invalid_anoms(anoms_transformed, ts.name)
        if anoms_transformed.shape[0] == 0:
            raise RuntimeError("No anomalies has left after removing invalid ones")

        # check if new anoms indices are consistent
        if not check_anom_df_consistency(anoms_transformed):
            raise RuntimeError(
                "Tranformation produced inconsistent anomalies indices; check if there are no "
                "anomalies in the part that is cut off in the transformation"
            )

        return TimeSeriesWithAnoms(
            values=values_transformed, anoms=anoms_transformed, name=ts.name
        )

    def __robust_diff(self, values: np.ndarray, lag: int) -> np.ndarray:
        unrobust_len = lag * self.k
        unrobust_part = values[lag:unrobust_len] - values[: (unrobust_len - lag)]
        ilags = [
            values[unrobust_len:] - values[(unrobust_len - i * lag) : -i * lag]
            for i in range(1, self.k + 1)
        ]
        robust_part = np.median(np.stack(ilags), axis=0)
        return np.concatenate((unrobust_part, robust_part))


class Detrending(MapTransformation):
    """Removes trend from series values

    x_out[i] = x_in[i] - trend[i]

    where trend[i] is extracted trend value at timestamp i

    """

    def __init__(self, window_len_getter: Callable[[TimeSeriesWithAnoms], int]):
        super().__init__(name=f"detrending")
        self.window_len_getter: Callable[[TimeSeriesWithAnoms], int] = window_len_getter

    def _transform_series(self, ts: TimeSeriesWithAnoms) -> TimeSeriesWithAnoms:

        detrend_window_len = self.window_len_getter(ts)
        values_transformed = self.__detrend(ts.values, window_len=detrend_window_len)

        return TimeSeriesWithAnoms(
            values=values_transformed, anoms=ts.anoms, name=ts.name
        )

    @staticmethod
    def __detrend(vals: np.ndarray, window_len: int) -> np.ndarray:
        """Removes trend component from given time series values
        with LOWESS

        Args:
            vals: 1D array with time series values
            window_len: the length of the filter window in LOWESS detrending
                (in timesteps)

        Returns:
            array: 1D array with time series values with trend removed
        """
        x = np.arange(vals.shape[0])

        flatten_lc, trend_lc = flatten(
            x,  # Array of time values
            vals,  # Array of flux values
            method="lowess",
            window_length=window_len,  # The length of the filter window in units of ``time``
            return_trend=True,  # Return trend and flattened light curve,
            cval=3.0,
        )

        return vals - trend_lc


class BoxCox(MapTransformation):
    """Performs Box-Cox transformation on series
    values

    If minimal value of series is less than `eps`, then series values
    are shifted so that `eps` is the new minimal value
    """

    def __init__(self, eps: float = 1e-3):
        super().__init__(name="boxcox")
        self.boxcox = BoxCoxTransformer()
        self.eps = eps

    def _transform_series(self, ts: TimeSeriesWithAnoms) -> TimeSeriesWithAnoms:
        # shift series if necessary
        series_min = np.amin(ts.values)
        if series_min < self.eps:
            values_to_transform = ts.values - series_min + self.eps
        else:
            values_to_transform = ts.values

        values_transformed = self.boxcox.fit_transform(values_to_transform).reshape(
            (-1,)
        )
        return TimeSeriesWithAnoms(
            values=values_transformed, anoms=ts.anoms, name=ts.name
        )
