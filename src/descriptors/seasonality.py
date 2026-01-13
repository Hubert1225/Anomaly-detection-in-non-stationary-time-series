"""This module provides descriptor to examine
time series seasonality
"""

from typing import Any
import numpy as np
from wotan import flatten
from sktime.transformations.series.acf import AutoCorrelationTransformer

from base import TimeSeriesWithAnoms, TimeSeriesDescriptor


class Seasonality(TimeSeriesDescriptor):
    """Descriptor to examine time series seasonality

    At first, the time series is detrended using LOWESS,
    to roughly remove the trend so that seasonal patterns can be
    detected more easily. Next, ACF is computed. The lag with the greatest
    ACF value is assumed to be the length of the seasonal period.
    It is return along with the greatest ACF value

    Attributes:
        detrend_window_len: the length of the filter window in LOWESS detrending
            (in timesteps)
        min_season: minimal possible length of the season considered
        max_season: maximum possible length of the season considered
    """

    def __init__(self, detrend_window_len: int, min_season: int, max_season: int):
        self.detrend_window_len: int = detrend_window_len
        self.min_season: int = min_season
        self.max_season: int = max_season

    def get_value_for_ts(self, ts: TimeSeriesWithAnoms) -> dict[str, Any]:

        features: dict[str, Any] = {}
        season_len, season_autocorr = self.__get_season_len(ts)
        features["Season length"] = season_len
        features["Season autocorrelation"] = season_autocorr
        return features

    def __get_season_len(self, ts: TimeSeriesWithAnoms) -> tuple[int, float]:
        """Obtains length of the seasonality and associated ACF value
        using Autocorrelation function (ACF) of the series

        Args:
            ts: time series object

        Returns:
            tuple: (season_len, max_acf)
        """
        detrended_vals = self.__detrend(ts.values)
        vals_acf = self.__acf(detrended_vals, n_lags=self.max_season)
        vals_acf = vals_acf[self.min_season :]
        max_acf = np.max(vals_acf)
        season_len = np.argmax(vals_acf) + self.min_season
        return season_len, max_acf

    def __detrend(self, vals: np.ndarray) -> np.ndarray:
        """Removes trend component from given time series values
        with LOWESS

        Args:
            vals: 1D array with time series values

        Returns:
            array: 1D array with time series values with trend removed
        """
        x = np.arange(vals.shape[0])

        flatten_lc, trend_lc = flatten(
            x,  # Array of time values
            vals,  # Array of flux values
            method="lowess",
            window_length=self.detrend_window_len,  # The length of the filter window in units of ``time``
            return_trend=True,  # Return trend and flattened light curve,
            cval=3.0,
        )

        return vals - trend_lc

    @staticmethod
    def __acf(vals: np.ndarray, n_lags: int) -> np.ndarray:
        """Computes Autocorrelation function for lags 0, ..., n_lags
        for given time series values
        """
        acf = AutoCorrelationTransformer(n_lags=n_lags)
        return acf.fit_transform(vals).reshape(-1)
