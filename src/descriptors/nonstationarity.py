"""This module provides descriptor for examining nonstationarity
presence and types in time series
"""

from typing import Any

import numpy as np
from arch.unitroot import KPSS
import statsmodels.api as sm
from sktime.param_est.stationarity import StationarityADFArch

from base import TimeSeriesDescriptor, TimeSeriesWithAnoms


class NonstationarityTypes(TimeSeriesDescriptor):
    """Descriptor to examine time series (non)stationarity

    It performs statistical tests to check time series
    stationarity, trend stationarity and
    heteroscedasticity.

    Attributes:
        sign_level: significance level used in statistical tests;
            common values are e.g. 0.05, 0.01
        adf_max_lag: The maximum number of lags to use when selecting lag length
            in ADF test
    """

    def __init__(self, sign_level: float, adf_max_lag: int):
        self.sign_level: float = sign_level
        self.adf_max_lag: int = adf_max_lag

    def get_value_for_ts(self, ts: TimeSeriesWithAnoms) -> dict[str, Any]:

        features: dict[str, Any] = {}

        is_stationary = self.__adf_test(ts)
        features["Stationary"] = is_stationary

        features["Trend stationary"] = self.__trend_test(ts)

        is_het = self.__het_test(ts)
        features["Heteroscedasticity"] = is_het

        return features

    def __trend_test(self, ts: TimeSeriesWithAnoms) -> bool:
        """Performs KPSS test to determine if the series is trend stationary
        (i.e. stationary around the deterministic trend)

        Args:
            ts: time series object

        Returns:
            tuple: bool
                whether the series is trend stationary
        """
        vals = ts.values
        kpss_result = KPSS(vals, trend="ct").pvalue
        return kpss_result > self.sign_level

    def __het_test(self, ts: TimeSeriesWithAnoms) -> bool:
        """Performs Goldfeld-Quandt homoskedasticity test
        with two-sided alternative

        Args:
            ts: tested time series object

        Returns:
            `True` if heteroscedasticity is detected, `False` otherwise
        """
        x = np.arange(ts.values.shape[0]).reshape((-1, 1))
        _, pval, _ = sm.stats.diagnostic.het_goldfeldquandt(
            y=ts.values, x=x, alternative="two-sided"
        )
        if pval <= self.sign_level:
            return True
        return False

    def __adf_test(self, ts: TimeSeriesWithAnoms) -> bool:
        """Performs test for stationarity via the
        Augmented Dickey-Fuller Unit Root Test

        Args:
            ts: time series object to be tested

        Returns:
            bool: whether the series in fit is stationary according to the test;
                more precisely, whether the null of the ADF test is rejected
        """
        sty_est = StationarityADFArch(
            p_threshold=self.sign_level, max_lags=self.adf_max_lag
        )
        sty_est.fit(ts.values)
        return sty_est.get_fitted_params()["stationary"]
